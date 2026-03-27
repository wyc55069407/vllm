#pragma once
// moe_prefill_lgrf.h — doubleGRF GGEMV kernels for MoE prefill
// Ported from moe_prefill_v29dll.cpp (Windows DLL)
// Build: common_ops_lgrf extension (doubleGRF)
//
// GRF budget (doubleGRF=256): MS=4, NS=4, N=32
//   acc:    4*4*128 floats = 8192B = 128 GRF
//   b_tile: 4*256   halfs  = 2048B =  32 GRF
//   in_off: 64      u32    =  256B =   4 GRF
//   scl_byte_off: 32 u32   =  128B =   2 GRF
//   w_tile+scl+temps        ≈ 1152B =  18 GRF
//   Total ≈ 184 GRF < 256 — fits doubleGRF

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx_ns = sycl::ext::intel::esimd::xmx;

using fp16 = sycl::half;

struct MoePrefillChunkInfo { int eid; int t0; int nt; };

// ======================== Fused Gate+Up GGEMV (N=32, doubleGRF) ========================
inline sycl::event moe_up_forward_v29(
    sycl::queue& q,
    const fp16* expert_states, const uint8_t* guw_weight, const fp16* guw_scale,
    fp16* gate_buf, const MoePrefillChunkInfo* chunks, int num_chunks,
    int hidden_size, int intermediate_size)
{
    constexpr int BS = 128;
    constexpr int MAX_M = 64;
    constexpr int N = 32;
    constexpr int MS = MAX_M / 16;   // 4
    constexpr int NS = N / 8;        // 4
    constexpr int ACC_SZ = MS * NS * 128;  // 2048

    const int fused_im = 2 * intermediate_size;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, fused_im / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int nblocks = hidden_size / BS;
            const int cid = (int)id[0];
            const int tid = (int)id[1];
            const int r_base = tid * N;

            const int eid = chunks[cid].eid;
            const int t0  = chunks[cid].t0;
            const int nt  = chunks[cid].nt;
            if (nt <= 0) return;

            simd<uint32_t, MAX_M> in_off =
                min(simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)t0,
                    simd<uint32_t, MAX_M>((uint32_t)(t0 + nt - 1)))
                * (uint32_t)(hidden_size * sizeof(fp16));

            const size_t row_base = (size_t)eid * fused_im + r_base;
            const uint8_t* w_ptr = guw_weight + row_base * (hidden_size / 2);

            simd<float, ACC_SZ> acc(0.f);

            const simd<uint32_t, N> scl_byte_off =
                simd<uint32_t, N>(0u, 1u) * (uint32_t)(nblocks * sizeof(fp16));
            const fp16* s_base = guw_scale + row_base * nblocks;

            for (int blk = 0; blk < nblocks; blk++) {
                simd<fp16, N> scl = gather<fp16, N>(s_base + blk, scl_byte_off);

                for (int k_base = 0; k_base < BS / 2; k_base += 8) {
                    const uint32_t k_off = (uint32_t)(blk * BS + k_base * 2) * (uint32_t)sizeof(fp16);
                    simd<fp16, MS * 256> b_tile;
                    #pragma unroll
                    for (int ms = 0; ms < MS; ms++) {
                        b_tile.template select<256, 1>(ms * 256).template bit_cast_view<uint32_t>() =
                            xesimd::lsc_gather<uint32_t, 8,
                                xesimd::lsc_data_size::u32,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                                16, uint32_t>(
                                reinterpret_cast<const uint32_t*>(expert_states),
                                in_off.template select<16, 1>(ms * 16).read() + k_off);
                    }

                    const int x_byte = blk * (BS / 2) + k_base;
                    auto w_raw_all = xesimd::lsc_load_2d<uint8_t, 8, 32, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        w_ptr, (unsigned)(hidden_size/2-1), (unsigned)(N-1),
                        (unsigned)(hidden_size/2-1), x_byte, 0);

                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++) {
                        simd<uint8_t, 64> w_raw = w_raw_all.template select<64, 1>(ns * 64);
                        simd<uint8_t, 64> lo = w_raw & (uint8_t)0x0F;
                        simd<uint8_t, 64> hi = w_raw >> 4;
                        simd<fp16, 64> lo_val = lo.template bit_cast_view<int8_t>() - (int8_t)8;
                        simd<fp16, 64> hi_val = hi.template bit_cast_view<int8_t>() - (int8_t)8;
                        simd<fp16, 128> w_tile;
                        w_tile.template select<64, 2>(0) = lo_val;
                        w_tile.template select<64, 2>(1) = hi_val;
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            fp16 ws = scl[ns*8+r];
                            w_tile.template select<16, 1>(r*16) *= ws;
                        }

                        #pragma unroll
                        for (int ms = 0; ms < MS; ms++) {
                            const int idx = ms * NS * 128 + ns * 128;
                            simd<fp16, 256> bv = b_tile.template select<256, 1>(ms * 256);
                            simd<float, 128> a = acc.template select<128, 1>(idx);
                            a = xmx_ns::dpas<8, 8, float, float, fp16, fp16>(a, bv, w_tile);
                            acc.template select<128, 1>(idx) = a;
                        }
                    }
                }
            }

            #pragma unroll
            for (int ms = 0; ms < MS; ms++) {
                #pragma unroll
                for (int m = 0; m < 16; m++) {
                    int tok = (ms * 16 + m < nt) ? (t0 + ms * 16 + m) : (t0 + nt - 1);
                    simd<float, N> row_f;
                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++)
                        row_f.template select<8, 1>(ns * 8) =
                            acc.template select<8, 16>(ms * NS * 128 + ns * 128 + m);
                    block_store<fp16, N>(gate_buf + (size_t)tok * fused_im + r_base,
                                         convert<fp16>(row_f));
                }
            }
        });
    });
}

// ======================== SiLU+Mul for chunked GGEMV output ========================
inline sycl::event moe_silu_mul_v29(
    sycl::queue& q,
    const fp16* gate_buf, fp16* intermediate,
    const MoePrefillChunkInfo* chunks, int num_chunks,
    int intermediate_size)
{
    const int fused_im = 2 * intermediate_size;
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, intermediate_size / 32),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
                const int cid = (int)id[0];
                const int tid = (int)id[1];
                const int r_base = tid * 32;
                const int t0 = chunks[cid].t0;
                const int nt = chunks[cid].nt;
                if (nt <= 0) return;

                for (int i = 0; i < nt; i++) {
                    const size_t row = (size_t)(t0 + i);
                    simd<fp16, 32> gate = block_load<fp16, 32>(gate_buf + row * fused_im + r_base);
                    simd<fp16, 32> up   = block_load<fp16, 32>(gate_buf + row * fused_im + intermediate_size + r_base);
                    simd<float, 32> gf = convert<float>(gate);
                    simd<float, 32> uf = convert<float>(up);
                    simd<float, 32> silu = gf / (1.f + sycl::ext::intel::esimd::exp(-gf));
                    block_store<fp16, 32>(intermediate + row * intermediate_size + r_base,
                                          convert<fp16>(silu * uf));
                }
            });
    });
}

// ======================== Down GGEMV (N=32, doubleGRF) ========================
inline sycl::event moe_down_forward_v29(
    sycl::queue& q,
    const fp16* intermediate, const uint8_t* down_weight, const fp16* down_scale,
    fp16* expert_output, const MoePrefillChunkInfo* chunks, int num_chunks,
    int intermediate_size, int hidden_size)
{
    constexpr int BS = 128;
    constexpr int MAX_M = 64;
    constexpr int N = 32;
    constexpr int MS = MAX_M / 16;   // 4
    constexpr int NS = N / 8;        // 4
    constexpr int ACC_SZ = MS * NS * 128;  // 2048

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<2>(num_chunks, hidden_size / N),
            [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {

            const int nblocks = intermediate_size / BS;
            const int cid = (int)id[0];
            const int tid = (int)id[1];
            const int d_base = tid * N;

            const int eid = chunks[cid].eid;
            const int t0  = chunks[cid].t0;
            const int nt  = chunks[cid].nt;
            if (nt <= 0) return;

            simd<uint32_t, MAX_M> in_off =
                min(simd<uint32_t, MAX_M>(0u, 1u) + (uint32_t)t0,
                    simd<uint32_t, MAX_M>((uint32_t)(t0 + nt - 1)))
                * (uint32_t)(intermediate_size * sizeof(fp16));

            const size_t row_base = (size_t)eid * hidden_size + d_base;
            const uint8_t* down_w_ptr = down_weight + row_base * (intermediate_size / 2);

            simd<float, ACC_SZ> acc(0.f);

            const simd<uint32_t, N> scl_byte_off =
                simd<uint32_t, N>(0u, 1u) * (uint32_t)(nblocks * sizeof(fp16));
            const fp16* down_s_base = down_scale + row_base * nblocks;

            for (int blk = 0; blk < nblocks; blk++) {
                simd<fp16, N> down_scl = gather<fp16, N>(down_s_base + blk, scl_byte_off);
                for (int k_base = 0; k_base < BS / 2; k_base += 8) {
                    const uint32_t k_off = (uint32_t)(blk * BS + k_base * 2) * (uint32_t)sizeof(fp16);
                    simd<fp16, MS * 256> b_tile;
                    #pragma unroll
                    for (int ms = 0; ms < MS; ms++) {
                        b_tile.template select<256, 1>(ms * 256).template bit_cast_view<uint32_t>() =
                            xesimd::lsc_gather<uint32_t, 8,
                                xesimd::lsc_data_size::u32,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                                16, uint32_t>(
                                reinterpret_cast<const uint32_t*>(intermediate),
                                in_off.template select<16, 1>(ms * 16).read() + k_off);
                    }
                    const int x_byte = blk * (BS / 2) + k_base;
                    auto dw_raw_all = xesimd::lsc_load_2d<uint8_t, 8, 32, 1,
                        false, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        down_w_ptr, (unsigned)(intermediate_size/2-1), (unsigned)(N-1),
                        (unsigned)(intermediate_size/2-1), x_byte, 0);

                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++) {
                        simd<uint8_t, 64> dw_raw = dw_raw_all.template select<64, 1>(ns * 64);
                        simd<uint8_t, 64> d_lo = dw_raw & (uint8_t)0x0F;
                        simd<uint8_t, 64> d_hi = dw_raw >> 4;
                        simd<fp16, 64> d_lo_val = d_lo.template bit_cast_view<int8_t>() - (int8_t)8;
                        simd<fp16, 64> d_hi_val = d_hi.template bit_cast_view<int8_t>() - (int8_t)8;
                        simd<fp16, 128> down_tile;
                        down_tile.template select<64, 2>(0) = d_lo_val;
                        down_tile.template select<64, 2>(1) = d_hi_val;
                        #pragma unroll
                        for (int r = 0; r < 8; r++) {
                            fp16 ds = down_scl[ns*8+r];
                            down_tile.template select<16, 1>(r*16) *= ds;
                        }
                        #pragma unroll
                        for (int ms = 0; ms < MS; ms++) {
                            const int idx = ms * NS * 128 + ns * 128;
                            simd<fp16, 256> bv = b_tile.template select<256, 1>(ms * 256);
                            simd<float, 128> a = acc.template select<128, 1>(idx);
                            a = xmx_ns::dpas<8, 8, float, float, fp16, fp16>(a, bv, down_tile);
                            acc.template select<128, 1>(idx) = a;
                        }
                    }
                }
            }

            #pragma unroll
            for (int ms = 0; ms < MS; ms++) {
                #pragma unroll
                for (int m = 0; m < 16; m++) {
                    int tok = (ms * 16 + m < nt) ? (t0 + ms * 16 + m) : (t0 + nt - 1);
                    simd<float, N> row_f;
                    #pragma unroll
                    for (int ns = 0; ns < NS; ns++)
                        row_f.template select<8, 1>(ns * 8) =
                            acc.template select<8, 16>(ms * NS * 128 + ns * 128 + m);
                    block_store<fp16, N>(expert_output + (size_t)tok * hidden_size + d_base,
                                         convert<fp16>(row_f));
                }
            }
        });
    });
}
