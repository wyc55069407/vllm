/* Standalone test: minimal DPAS QK accumulation loop to detect spills.
 * Compiles directly with icpx — no PyTorch dependency.
 * Measures TFLOPS for QK + softmax + VS in the same structure as sdp_paged.
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;
namespace ens = sycl::ext::intel::experimental::esimd;

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr uint32_t HD = 256;
static constexpr uint32_t HD_BLKS = 16;
static constexpr uint32_t WG_Q_ROWS = 128;
static constexpr uint32_t KV_CHUNK = 128;
static constexpr uint32_t KV_PER_SG = 16;
static constexpr uint32_t KV_BLKS = 8;
static constexpr uint32_t Q_ROWS = 8;
static constexpr uint32_t Q_PAIRS = 2;
static constexpr uint32_t Q_TILES = 8;
static constexpr uint32_t Q_GRPS = 4;
static constexpr uint32_t D_BLKS_PER_SG = 2;

static constexpr uint32_t Q_SLM_BASE = 0;
static constexpr uint32_t S_SLM_BASE = 0x10000;
static constexpr uint32_t MAX_SLM_BASE = 0x18000;
static constexpr uint32_t SUM_SLM_BASE = 0x19000;
static constexpr float FP32_MIN = -1e30f;

/* Simplified: contiguous K/V (no paging) but same computation structure */
ESIMD_INLINE void test_prefill_kernel(
    const unsigned short* Q_ptr,   // [q_len, num_heads, 256] bf16
    const unsigned short* K_ptr,   // [kv_len, num_kv_heads, 256] bf16
    const unsigned short* V_ptr,   // [kv_len, num_kv_heads, 256] bf16
    unsigned short* O_ptr,         // [q_len, num_heads, 256] bf16
    int q_len, int kv_len, int num_heads, int num_kv_heads,
    float attn_scale, nd_item<1> ndi)
{
    uint32_t wg_id = ndi.get_group(0);
    uint32_t tid = ndi.get_local_id(0);
    uint32_t sg_i = tid & 7;
    uint32_t sg_j = tid >> 3;

    slm_init(0x1A000);
    __esimd_nbarrier_init(1);

    int group_size = num_heads / num_kv_heads;
    uint32_t num_q_tiles = (q_len + WG_Q_ROWS - 1) / WG_Q_ROWS;
    uint32_t head_idx = wg_id / num_q_tiles;
    uint32_t q_tile_idx = wg_id % num_q_tiles;
    uint32_t kv_head = head_idx / group_size;

    int q_global_start = q_tile_idx * WG_Q_ROWS;
    int actual_q_rows = q_len - q_global_start;
    if (actual_q_rows > (int)WG_Q_ROWS) actual_q_rows = WG_Q_ROWS;
    if (actual_q_rows <= 0) return;

    float attnScoreMul = attn_scale * 1.44269504f;

    // Load Q to SLM — standard cooperative load
    uint32_t kv_surf_w = num_kv_heads * HD * 2 - 1;
    uint32_t kv_surf_h = kv_len - 1;
    uint32_t kv_head_offset = kv_head * HD;
    uint32_t kv_x_k = kv_head_offset;

    uint32_t q_surf_w = num_heads * HD * 2 - 1;
    uint32_t q_surf_h = q_len - 1;

    // Q tile loading — each thread loads 4 tiles
    for (uint32_t t = 0; t < 4; t++) {
        uint32_t tile_id = tid * 4 + t;
        if (tile_id >= HD_BLKS * Q_TILES) continue;

        uint32_t d_blk = tile_id / Q_TILES;
        uint32_t q_sub = tile_id % Q_TILES;

        uint32_t q_row = q_global_start + q_sub * 16;
        uint32_t q_col = head_idx * HD + d_blk * 16;

        ens::config_2d_mem_access<fp16, 16, 16, 1> payloadQ(
            (fp16*)Q_ptr, q_surf_w, q_surf_h, q_surf_w, q_col, q_row);
        simd<fp16, 256> qTile_fp16 = ens::lsc_load_2d<fp16, 16, 16, 1, false, true,
            ens::cache_hint::cached, ens::cache_hint::cached>(payloadQ);
        auto qTile = qTile_fp16.template bit_cast_view<uint32_t>();

        uint32_t slm_off = Q_SLM_BASE + tile_id * 512;
        slm_block_store<uint32_t, 64>(slm_off, qTile.select<64, 1>(0));
        slm_block_store<uint32_t, 64>(slm_off + 256, qTile.select<64, 1>(64));
    }

    barrier();

    simd<float, 1024> A_tile = 0;
    simd<float, 512> ST_tile;
    simd<float, 32> fp32_max = FP32_MIN;
    simd<float, 32> fp32_sum = 0;
    simd<float, 32> delta;

    int32_t max_kv_end = q_global_start + actual_q_rows;
    if (max_kv_end > kv_len) max_kv_end = kv_len;
    int32_t kvOuterLoops = (max_kv_end + KV_CHUNK - 1) / KV_CHUNK;
    if (kvOuterLoops <= 0) kvOuterLoops = 1;

    for (int32_t outerIter = 0; outerIter < kvOuterLoops; outerIter++) {
        uint32_t kv_start = outerIter * KV_CHUNK;

        // ===== QK PHASE =====
        {
            int32_t kv_pos = kv_start + sg_i * KV_PER_SG;
            ST_tile = 0;

            // Contiguous K surface (no paging)
            ens::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
                (fp16*)K_ptr, kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, kv_pos);

            simd<fp16, 256> K_both = ens::lsc_load_2d<fp16, 16, 16, 1, false, false,
                ens::cache_hint::cached, ens::cache_hint::cached>(payloadK);

            for (int d = 0; d < (int)HD_BLKS; d++) {
                auto K_us = K_both.template bit_cast_view<unsigned short>();
                simd<bf16, 128> K_sb0(K_us.select<128, 1>(0).template bit_cast_view<bf16>().data());
                simd<bf16, 128> K_sb1(K_us.select<128, 1>(128).template bit_cast_view<bf16>().data());

                uint32_t q_slm_off0 = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + 0) * 512;
                uint32_t q_slm_off1 = Q_SLM_BASE + (d * Q_TILES + sg_j * Q_PAIRS + 1) * 512;

                simd<bf16, 256> Q_vnni0, Q_vnni1;
                Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(0) =
                    slm_block_load<uint32_t, 64>(q_slm_off0);
                Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(64) =
                    slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
                Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(0) =
                    slm_block_load<uint32_t, 64>(q_slm_off1);
                Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(64) =
                    slm_block_load<uint32_t, 64>(q_slm_off1 + 256);

                { auto acc = ST_tile.select<128, 1>(0);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni0.data()), K_sb0); }
                { auto acc = ST_tile.select<128, 1>(128);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni0.data()), K_sb1); }
                { auto acc = ST_tile.select<128, 1>(256);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni1.data()), K_sb0); }
                { auto acc = ST_tile.select<128, 1>(384);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni1.data()), K_sb1); }

                if (d < (int)HD_BLKS - 1) {
                    payloadK.set_x(kv_x_k + (d + 1) * 16);
                    K_both = ens::lsc_load_2d<fp16, 16, 16, 1, false, false,
                        ens::cache_hint::cached, ens::cache_hint::cached>(payloadK);
                }
            }
        }

        // ===== SOFTMAX FIRST HALF =====
        ST_tile *= attnScoreMul;

        // Causal mask
        #pragma unroll
        for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
            #pragma unroll
            for (int qr = 0; qr < 16; qr++) {
                int q_local = sg_j * 32 + qp * 16 + qr;
                int q_abs = q_global_start + q_local;
                int kv_end_for_q = q_abs + 1;
                uint32_t kv_base_sg = kv_start + sg_i * KV_PER_SG;
                #pragma unroll
                for (int kv = 0; kv < 8; kv++) {
                    if ((int)(kv_base_sg + kv) >= kv_end_for_q)
                        ST_tile[qp * 256 + kv * 16 + qr] = FP32_MIN;
                    if ((int)(kv_base_sg + 8 + kv) >= kv_end_for_q)
                        ST_tile[qp * 256 + 128 + kv * 16 + qr] = FP32_MIN;
                }
            }
        }

        // Out-of-range Q masking
        #pragma unroll
        for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
            #pragma unroll
            for (int qr = 0; qr < 16; qr++) {
                int q_local = sg_j * 32 + qp * 16 + qr;
                if (q_local >= actual_q_rows) {
                    #pragma unroll
                    for (int kv = 0; kv < 16; kv++)
                        ST_tile[qp * 256 + kv * 16 + qr] = FP32_MIN;
                }
            }
        }

        // Local max
        simd<float, 32> local_max;
        #pragma unroll
        for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
            local_max.select<16, 1>(qp * 16) = ST_tile.select<16, 1>(qp * 256);
            #pragma unroll
            for (int kv = 1; kv < 8; kv++)
                local_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    local_max.select<16, 1>(qp * 16),
                    ST_tile.select<16, 1>(qp * 256 + kv * 16));
            #pragma unroll
            for (int kv = 0; kv < 8; kv++)
                local_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    local_max.select<16, 1>(qp * 16),
                    ST_tile.select<16, 1>(qp * 256 + 128 + kv * 16));
        }

        // Store max to SLM
        #pragma unroll
        for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
            uint32_t q_base = sg_j * 32 + qp * 16;
            slm_block_store<float, 16>(MAX_SLM_BASE + (sg_i * 128 + q_base) * 4,
                local_max.select<16, 1>(qp * 16));
        }

        barrier();

        // ===== SOFTMAX SECOND HALF =====
        simd<float, 32> global_max = FP32_MIN;
        #pragma unroll
        for (int si = 0; si < 8; si++) {
            #pragma unroll
            for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
                uint32_t q_base = sg_j * 32 + qp * 16;
                simd<float, 16> m = slm_block_load<float, 16>(MAX_SLM_BASE + (si * 128 + q_base) * 4);
                global_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    global_max.select<16, 1>(qp * 16), m);
            }
        }
        global_max = __ESIMD_NS::max<float, 32, float>(global_max, fp32_max);
        delta = __ESIMD_NS::exp2<float, 32, float>(fp32_max - global_max);
        fp32_max = global_max;

        simd<float, 32> local_sum = 0;
        simd<bf16, 256> ST_bf16_0, ST_bf16_1;
        {
            simd<float, 16> gm = global_max.select<16, 1>(0);
            #pragma unroll
            for (int kv = 0; kv < 16; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>((kv < 8 ? 0 : 128) + (kv % 8) * 16) - gm);
                ST_bf16_0.select<16, 1>(kv * 16) = s;
                local_sum.select<16, 1>(0) += s;
            }
        }
        {
            simd<float, 16> gm = global_max.select<16, 1>(16);
            #pragma unroll
            for (int kv = 0; kv < 16; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + (kv < 8 ? 0 : 128) + (kv % 8) * 16) - gm);
                ST_bf16_1.select<16, 1>(kv * 16) = s;
                local_sum.select<16, 1>(16) += s;
            }
        }

        // ===== S SCATTER TO SLM =====
        #pragma unroll
        for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
            auto& ST_bf16 = (qp == 0) ? ST_bf16_0 : ST_bf16_1;
            simd<uint16_t, 256> ST_u16 = ST_bf16.template bit_cast_view<uint16_t>();

            uint32_t ta0 = S_SLM_BASE + (sg_i * 16 + sg_j * Q_GRPS + qp * 2 + 0) * 256;
            uint32_t ta1 = S_SLM_BASE + (sg_i * 16 + sg_j * Q_GRPS + qp * 2 + 1) * 256;

            simd<uint32_t, 16> q_offsets;
            #pragma unroll
            for (int q = 0; q < 8; q++) q_offsets[q] = ta0 + q * 32;
            #pragma unroll
            for (int q = 0; q < 8; q++) q_offsets[8 + q] = ta1 + q * 32;

            #pragma unroll
            for (int kg = 0; kg < 4; kg++) {
                int kb = kg * 4;
                simd<uint32_t, 16> p0 =
                    simd<uint32_t, 16>(ST_u16.select<16, 1>((kb + 0) * 16)) |
                    (simd<uint32_t, 16>(ST_u16.select<16, 1>((kb + 1) * 16)) << 16);
                simd<uint32_t, 16> p1 =
                    simd<uint32_t, 16>(ST_u16.select<16, 1>((kb + 2) * 16)) |
                    (simd<uint32_t, 16>(ST_u16.select<16, 1>((kb + 3) * 16)) << 16);
                simd<uint32_t, 32> data;
                data.select<16, 1>(0) = p0;
                data.select<16, 1>(16) = p1;
                ens::lsc_slm_scatter<uint32_t, 2, ens::lsc_data_size::u32, 16>(
                    q_offsets + kb * 2, data);
            }
        }

        // ===== BARRIER B =====
        __esimd_nbarrier_arrive(0, 0, 32, 32);

        fp32_sum = fp32_sum * delta + local_sum;

        // V load from contiguous memory
        int32_t v_kv_pos_0 = kv_start;
        ens::config_2d_mem_access<fp16, 16, 16, 1> payloadV(
            (fp16*)V_ptr, kv_surf_w, kv_surf_h, kv_surf_w,
            (uint32_t)(kv_head_offset + sg_i * 32), v_kv_pos_0);

        simd<fp16, 256> V_vnni0 = ens::lsc_load_2d<fp16, 16, 16, 1, false, true,
            ens::cache_hint::cached, ens::cache_hint::cached>(payloadV);
        payloadV.set_x((uint32_t)(kv_head_offset + sg_i * 32 + 16));
        simd<fp16, 256> V_vnni1 = ens::lsc_load_2d<fp16, 16, 16, 1, false, true,
            ens::cache_hint::cached, ens::cache_hint::cached>(payloadV);

        // Compensation
        #pragma unroll
        for (int qg = 0; qg < (int)Q_GRPS; qg++) {
            #pragma unroll
            for (int db = 0; db < (int)D_BLKS_PER_SG; db++) {
                #pragma unroll
                for (int q = 0; q < (int)Q_ROWS; q++) {
                    float d_val = delta[qg * Q_ROWS + q];
                    A_tile.select<16, 1>((qg * D_BLKS_PER_SG + db) * 128 + q * 16) *= d_val;
                }
            }
        }

        __esimd_nbarrier(0, 0, 32);

        // ===== VS PHASE =====
        for (int kv_blk = 0; kv_blk < (int)KV_BLKS; kv_blk++) {
            if (kv_blk > 0) {
                int32_t v_kv_pos = kv_start + kv_blk * 16;
                payloadV.set_x((uint32_t)(kv_head_offset + sg_i * 32));
                payloadV.set_y(v_kv_pos);
                V_vnni0 = ens::lsc_load_2d<fp16, 16, 16, 1, false, true,
                    ens::cache_hint::cached, ens::cache_hint::cached>(payloadV);
                payloadV.set_x((uint32_t)(kv_head_offset + sg_i * 32 + 16));
                V_vnni1 = ens::lsc_load_2d<fp16, 16, 16, 1, false, true,
                    ens::cache_hint::cached, ens::cache_hint::cached>(payloadV);
            }

            uint32_t s_base = S_SLM_BASE + (kv_blk * 16 + sg_j * Q_GRPS) * 256;
            simd<bf16, 128> S0, S1, S2, S3;
            S0.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base);
            S1.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 256);
            S2.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 512);
            S3.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 768);

            auto V_bf16_0 = V_vnni0.template bit_cast_view<bf16>();
            auto V_bf16_1 = V_vnni1.template bit_cast_view<bf16>();

            { auto acc = A_tile.select<128, 1>(0);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_0.data()), S0); }
            { auto acc = A_tile.select<128, 1>(1 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_1.data()), S0); }
            { auto acc = A_tile.select<128, 1>(2 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_0.data()), S1); }
            { auto acc = A_tile.select<128, 1>(3 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_1.data()), S1); }
            { auto acc = A_tile.select<128, 1>(4 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_0.data()), S2); }
            { auto acc = A_tile.select<128, 1>(5 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_1.data()), S2); }
            { auto acc = A_tile.select<128, 1>(6 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_0.data()), S3); }
            { auto acc = A_tile.select<128, 1>(7 * 128);
              acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(V_bf16_1.data()), S3); }
        }
    }

    // ===== OUTPUT =====
    #pragma unroll
    for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
        uint32_t q_base = sg_j * 32 + qp * 16;
        slm_block_store<float, 16>(SUM_SLM_BASE + (sg_i * 128 + q_base) * 4,
            fp32_sum.select<16, 1>(qp * 16));
    }
    barrier();

    simd<float, 32> total_sum = 0;
    #pragma unroll
    for (int si = 0; si < 8; si++) {
        #pragma unroll
        for (int qp = 0; qp < (int)Q_PAIRS; qp++) {
            uint32_t q_base = sg_j * 32 + qp * 16;
            total_sum.select<16, 1>(qp * 16) += slm_block_load<float, 16>(
                SUM_SLM_BASE + (si * 128 + q_base) * 4);
        }
    }

    simd<float, 32> inv_sum;
    inv_sum.select<16, 1>(0) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(0));
    inv_sum.select<16, 1>(16) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(16));

    uint32_t d_start = sg_i * D_BLKS_PER_SG * 16;
    uint32_t outW = num_heads * HD * 2 - 1;
    uint32_t outH = q_len - 1;
    ens::config_2d_mem_access<fp16, 16, 8, 1> payloadO(
        (fp16*)O_ptr, outW, outH, outW, 0, 0);

    #pragma unroll
    for (int qg = 0; qg < (int)Q_GRPS; qg++) {
        #pragma unroll
        for (int db = 0; db < (int)D_BLKS_PER_SG; db++) {
            simd<fp16, 128> fOut;
            #pragma unroll
            for (int q = 0; q < (int)Q_ROWS; q++) {
                simd<float, 16> f32_out = A_tile.select<16, 1>((qg * D_BLKS_PER_SG + db) * 128 + q * 16);
                float inv = inv_sum[qg * Q_ROWS + q];
                f32_out *= inv;
                simd<bf16, 16> bf16_out = f32_out;
                fOut.select<16, 1>(q * 16) = bf16_out.template bit_cast_view<fp16>();
            }

            int q_row_in_tile = sg_j * 32 + qg * Q_ROWS;
            int q_global_row = q_global_start + q_row_in_tile;

            if (q_row_in_tile + Q_ROWS <= (uint32_t)actual_q_rows) {
                payloadO.set_x(head_idx * HD + d_start + db * 16);
                payloadO.set_y(q_global_row);
                ens::lsc_store_2d<fp16, 16, 8, 1,
                    ens::cache_hint::write_back, ens::cache_hint::write_back>(payloadO, fOut);
            }
        }
    }
}


int main() {
    queue q(gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    int num_heads = 16, num_kv_heads = 4;
    int head_dim = 256;
    float scale = 1.0f / sqrtf(256.0f);

    for (int sz : {4096, 8192}) {
        int q_len = sz, kv_len = sz;

        auto Q = malloc_device<unsigned short>(q_len * num_heads * head_dim, q);
        auto K = malloc_device<unsigned short>(kv_len * num_kv_heads * head_dim, q);
        auto V = malloc_device<unsigned short>(kv_len * num_kv_heads * head_dim, q);
        auto O = malloc_device<unsigned short>(q_len * num_heads * head_dim, q);

        // Init with random data via host
        auto hQ = malloc_host<unsigned short>(q_len * num_heads * head_dim, q);
        srand(42);
        for (int i = 0; i < q_len * num_heads * head_dim; i++)
            hQ[i] = 0x3C00; // 1.0 in fp16
        q.memcpy(Q, hQ, q_len * num_heads * head_dim * 2);
        q.memcpy(K, hQ, kv_len * num_kv_heads * head_dim * 2);  // reuse
        q.memcpy(V, hQ, kv_len * num_kv_heads * head_dim * 2);
        q.wait();
        free(hQ, q);

        int num_q_tiles = (q_len + WG_Q_ROWS - 1) / WG_Q_ROWS;
        int total_wgs = num_heads * num_q_tiles;

        auto run = [&]() {
            q.submit([&](handler& cgh) {
                cgh.parallel_for(nd_range<1>(range<1>(total_wgs * 32), range<1>(32)),
                    [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                        test_prefill_kernel(Q, K, V, O, q_len, kv_len,
                            num_heads, num_kv_heads, scale, ndi);
                    });
            });
        };

        // Warmup
        for (int i = 0; i < 5; i++) run();
        q.wait();

        auto t0 = std::chrono::high_resolution_clock::now();
        int iters = 20;
        for (int i = 0; i < iters; i++) run();
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count() / iters;

        double flops = 2.0 * q_len * kv_len * head_dim * num_heads * 2;
        double tflops = flops / sec / 1e12;

        std::cout << sz << "x" << sz << ": " << sec * 1e3 << " ms, " << tflops << " TFLOPS" << std::endl;

        free(Q, q); free(K, q); free(V, q); free(O, q);
    }

    return 0;
}
