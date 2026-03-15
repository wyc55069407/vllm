/* sdp_paged.h — Optimized paged SDP kernels (HD=256, bf16io).
 *
 * Four kernels:
 *   1. sdp_paged_kernel_scalar    — Scalar fallback for decode/prefill (all HD)
 *   2. sdp_paged_decode_phase1    — Per-chunk partial softmax for decode
 *   3. sdp_paged_decode_phase2    — Cross-chunk log-sum-exp reduction
 *   4. sdp_paged_prefill_dpas     — DPAS-based prefill (HD=256, 32-thread WG)
 *
 * KV cache layout: [2, num_blocks, block_size, num_kv_heads, head_dim] bf16 (NHD).
 *   kv_cache[0] = key cache, kv_cache[1] = value cache.
 *   Stride-aware: handles non-contiguous layouts from vLLM permute.
 */

#include "utils.h"

#ifndef FP32_MIN
#define FP32_MIN (-3.402823466e+38f)
#endif

using bf16 = sycl::ext::oneapi::bfloat16;

/* ============================================================
 * Constants for decode kernel
 * ============================================================ */
static constexpr int DEC_CHUNK_SIZE = 128;  // KV tokens per chunk in phase1
static constexpr int DEC_SCRATCH_PER_CHUNK = 528;  // bytes: 4+4+512 padded to 16B

/* ============================================================
 * ESIMD scalar math helpers
 * ============================================================ */
ESIMD_INLINE float sdp_esimd_expf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::exp(v);
    return v[0];
}

/* ============================================================
 * Load / store helpers
 * ============================================================ */

/* Load 64 bf16 values as f32. */
ESIMD_INLINE simd<float, 64> sdp_load_bf16_64(const unsigned short* ptr) {
    simd<unsigned short, 64> raw = block_load<unsigned short, 64>(ptr);
    simd<unsigned int, 64> bits = raw;
    bits <<= 16;
    return bits.template bit_cast_view<float>();
}

/* Dot product of two f32-64 vectors via tree reduction. */
ESIMD_INLINE float sdp_dot64(simd<float, 64> a, simd<float, 64> b) {
    simd<float, 64> p = a * b;
    p.select<32,1>(0) += p.select<32,1>(32);
    p.select<16,1>(0) += p.select<16,1>(16);
    p.select<8,1>(0) += p.select<8,1>(8);
    p.select<4,1>(0) += p.select<4,1>(4);
    p.select<2,1>(0) += p.select<2,1>(2);
    return p[0] + p[1];
}

/* Dot product of split-256 f32 vectors (4 x 64). */
ESIMD_INLINE float sdp_dot256(
    simd<float, 64> a0, simd<float, 64> a1,
    simd<float, 64> a2, simd<float, 64> a3,
    simd<float, 64> b0, simd<float, 64> b1,
    simd<float, 64> b2, simd<float, 64> b3) {
    return sdp_dot64(a0, b0) + sdp_dot64(a1, b1) +
           sdp_dot64(a2, b2) + sdp_dot64(a3, b3);
}

/* f32 -> bf16 store: 64 elements with rounding. */
ESIMD_INLINE void sdp_store_bf16_64(unsigned short* ptr, simd<float, 64> val) {
    simd<unsigned int, 64> bits = val.template bit_cast_view<unsigned int>();
    simd<unsigned int, 64> rounding = ((bits >> 16) & 1u) + 0x7FFFu;
    bits += rounding;
    simd<unsigned short, 64> bf16_bits = (bits >> 16);
    block_store<unsigned short, 64>(ptr, bf16_bits);
}


/* ============================================================
 * SCALAR FALLBACK KERNEL — decode + prefill
 * One ESIMD thread per (request, head) or per (token, head).
 * Supports HD=128 and HD=256.
 * ============================================================ */
ESIMD_INLINE void sdp_paged_kernel_scalar(
    const unsigned short* __restrict__ query_ptr,
    const unsigned short* __restrict__ kv_cache_ptr,
    unsigned short* __restrict__ output_ptr,
    const int* __restrict__ block_table_ptr,
    const int* __restrict__ seq_lens_ptr,
    const int* __restrict__ query_start_loc_ptr,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_split, int64_t kv_stride_block,
    int64_t kv_stride_pos, int64_t kv_stride_head,
    float attn_scale,
    int causal,
    int max_query_len,
    nd_item<1>& ndi)
{
    const int tid = ndi.get_global_id(0);

    int req_idx, q_offset_in_req, head_idx;

    if (max_query_len == 1) {
        req_idx = tid / num_heads;
        head_idx = tid % num_heads;
        q_offset_in_req = 0;
    } else {
        int token_linear_idx = tid / num_heads;
        head_idx = tid % num_heads;
        req_idx = 0;
        for (int r = 0; ; r++) {
            int next_start = query_start_loc_ptr[r + 1];
            if (token_linear_idx < next_start) {
                req_idx = r;
                q_offset_in_req = token_linear_idx - query_start_loc_ptr[r];
                break;
            }
        }
    }

    const int seq_len = seq_lens_ptr[req_idx];
    if (seq_len <= 0) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    int query_len;
    if (max_query_len == 1) {
        query_len = 1;
    } else {
        query_len = query_start_loc_ptr[req_idx + 1] - query_start_loc_ptr[req_idx];
    }
    const int q_abs_pos = (seq_len - query_len) + q_offset_in_req;

    int token_idx;
    if (max_query_len == 1) {
        token_idx = req_idx;
    } else {
        token_idx = query_start_loc_ptr[req_idx] + q_offset_in_req;
    }

    const unsigned short* q_row = query_ptr +
        (int64_t)token_idx * num_heads * head_dim +
        (int64_t)head_idx * head_dim;

    simd<float, 64> q0 = sdp_load_bf16_64(q_row);
    simd<float, 64> q1 = sdp_load_bf16_64(q_row + 64);
    simd<float, 64> q2, q3;
    const bool hd256 = (head_dim == 256);
    if (hd256) {
        q2 = sdp_load_bf16_64(q_row + 128);
        q3 = sdp_load_bf16_64(q_row + 192);
    }

    float max_score = FP32_MIN;
    float sum_exp = 0.0f;
    simd<float, 64> acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);

    const int64_t kv_head_offset = (int64_t)kv_head_idx * kv_stride_head;
    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    int kv_end = causal ? (q_abs_pos + 1) : seq_len;

    for (int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
        int block_idx = kv_pos / block_size;
        int block_offset = kv_pos % block_size;
        int block_num = block_table_row[block_idx];

        int64_t kv_base = (int64_t)block_num * kv_stride_block +
                          (int64_t)block_offset * kv_stride_pos +
                          kv_head_offset;

        const unsigned short* k_ptr = kv_cache_ptr + kv_base;
        simd<float, 64> k0 = sdp_load_bf16_64(k_ptr);
        simd<float, 64> k1 = sdp_load_bf16_64(k_ptr + 64);

        float score;
        if (hd256) {
            simd<float, 64> k2 = sdp_load_bf16_64(k_ptr + 128);
            simd<float, 64> k3 = sdp_load_bf16_64(k_ptr + 192);
            score = sdp_dot256(q0, q1, q2, q3, k0, k1, k2, k3) * attn_scale;
        } else {
            score = (sdp_dot64(q0, k0) + sdp_dot64(q1, k1)) * attn_scale;
        }

        float new_max = (score > max_score) ? score : max_score;
        float correction = sdp_esimd_expf(max_score - new_max);
        acc0 *= correction; acc1 *= correction;
        if (hd256) { acc2 *= correction; acc3 *= correction; }
        sum_exp *= correction;

        float w = sdp_esimd_expf(score - new_max);
        sum_exp += w;
        max_score = new_max;

        const unsigned short* v_ptr = kv_cache_ptr + kv_base + kv_stride_split;
        acc0 += w * sdp_load_bf16_64(v_ptr);
        acc1 += w * sdp_load_bf16_64(v_ptr + 64);
        if (hd256) {
            acc2 += w * sdp_load_bf16_64(v_ptr + 128);
            acc3 += w * sdp_load_bf16_64(v_ptr + 192);
        }
    }

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        acc0 *= inv_sum; acc1 *= inv_sum;
        if (hd256) { acc2 *= inv_sum; acc3 *= inv_sum; }
    }

    unsigned short* out_row = output_ptr +
        (int64_t)token_idx * num_heads * head_dim +
        (int64_t)head_idx * head_dim;

    sdp_store_bf16_64(out_row, acc0);
    sdp_store_bf16_64(out_row + 64, acc1);
    if (hd256) {
        sdp_store_bf16_64(out_row + 128, acc2);
        sdp_store_bf16_64(out_row + 192, acc3);
    }
}


/* ============================================================
 * DECODE PHASE 1 — Per-chunk partial softmax + output
 *
 * One ESIMD thread per (request, head, chunk).
 * Processes DEC_CHUNK_SIZE KV tokens, computes partial
 * (max_score, sum_exp, output[head_dim]) and stores to scratch.
 *
 * Scratch layout per chunk (DEC_SCRATCH_PER_CHUNK bytes):
 *   float max_score;        // offset 0
 *   float sum_exp;          // offset 4
 *   bf16  output[256];      // offset 8, 512 bytes
 *   Total: 520B, padded to 528 for alignment
 * ============================================================ */
ESIMD_INLINE void sdp_paged_decode_phase1(
    const unsigned short* __restrict__ query_ptr,
    const unsigned short* __restrict__ kv_cache_ptr,
    float* __restrict__ scratch_ptr,
    const int* __restrict__ block_table_ptr,
    const int* __restrict__ seq_lens_ptr,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_split, int64_t kv_stride_block,
    int64_t kv_stride_pos, int64_t kv_stride_head,
    float attn_scale,
    int num_chunks_per_seq,
    nd_item<1>& ndi)
{
    const int global_id = ndi.get_global_id(0);
    const int chunk_id = global_id % num_chunks_per_seq;
    const int head_idx = (global_id / num_chunks_per_seq) % num_heads;
    const int req_idx = global_id / (num_chunks_per_seq * num_heads);

    const int seq_len = seq_lens_ptr[req_idx];

    // Chunk bounds
    int kv_start = chunk_id * DEC_CHUNK_SIZE;
    if (kv_start >= seq_len) {
        // This chunk is beyond the sequence — write sentinel
        int scratch_idx = (req_idx * num_heads + head_idx) * num_chunks_per_seq + chunk_id;
        int scratch_offset = scratch_idx * (DEC_SCRATCH_PER_CHUNK / 4);
        scratch_ptr[scratch_offset] = FP32_MIN;  // max
        scratch_ptr[scratch_offset + 1] = 0.0f;  // sum
        return;
    }
    int kv_end = kv_start + DEC_CHUNK_SIZE;
    if (kv_end > seq_len) kv_end = seq_len;

    const int group_size = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / group_size;
    const int64_t kv_head_offset = (int64_t)kv_head_idx * kv_stride_head;
    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;

    // Load query [1, head_dim=256] as f32
    const unsigned short* q_row = query_ptr +
        (int64_t)req_idx * num_heads * head_dim +
        (int64_t)head_idx * head_dim;

    simd<float, 64> q0 = sdp_load_bf16_64(q_row);
    simd<float, 64> q1 = sdp_load_bf16_64(q_row + 64);
    simd<float, 64> q2 = sdp_load_bf16_64(q_row + 128);
    simd<float, 64> q3 = sdp_load_bf16_64(q_row + 192);

    // Online softmax state
    float max_score = FP32_MIN;
    float sum_exp = 0.0f;
    simd<float, 64> acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);

    for (int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        int block_idx = kv_pos / block_size;
        int block_offset = kv_pos & (block_size - 1);
        int block_num = block_table_row[block_idx];

        int64_t kv_base = (int64_t)block_num * kv_stride_block +
                          (int64_t)block_offset * kv_stride_pos +
                          kv_head_offset;

        // Load K
        const unsigned short* k_ptr = kv_cache_ptr + kv_base;
        simd<float, 64> k0 = sdp_load_bf16_64(k_ptr);
        simd<float, 64> k1 = sdp_load_bf16_64(k_ptr + 64);
        simd<float, 64> k2 = sdp_load_bf16_64(k_ptr + 128);
        simd<float, 64> k3 = sdp_load_bf16_64(k_ptr + 192);

        float score = sdp_dot256(q0, q1, q2, q3, k0, k1, k2, k3) * attn_scale;

        // Online softmax
        float new_max = (score > max_score) ? score : max_score;
        float correction = sdp_esimd_expf(max_score - new_max);
        acc0 *= correction; acc1 *= correction;
        acc2 *= correction; acc3 *= correction;
        sum_exp *= correction;

        float w = sdp_esimd_expf(score - new_max);
        sum_exp += w;
        max_score = new_max;

        // Load V
        const unsigned short* v_ptr = kv_cache_ptr + kv_base + kv_stride_split;
        acc0 += w * sdp_load_bf16_64(v_ptr);
        acc1 += w * sdp_load_bf16_64(v_ptr + 64);
        acc2 += w * sdp_load_bf16_64(v_ptr + 128);
        acc3 += w * sdp_load_bf16_64(v_ptr + 192);
    }

    // Store partials to scratch
    int scratch_idx = (req_idx * num_heads + head_idx) * num_chunks_per_seq + chunk_id;
    float* scratch_base = scratch_ptr + scratch_idx * (DEC_SCRATCH_PER_CHUNK / 4);

    // Store max and sum as scalar writes
    scratch_base[0] = max_score;
    scratch_base[1] = sum_exp;

    // Store output as bf16 (at offset 2 floats = 8 bytes)
    unsigned short* out_bf16 = reinterpret_cast<unsigned short*>(scratch_base + 2);
    sdp_store_bf16_64(out_bf16, acc0);
    sdp_store_bf16_64(out_bf16 + 64, acc1);
    sdp_store_bf16_64(out_bf16 + 128, acc2);
    sdp_store_bf16_64(out_bf16 + 192, acc3);
}


/* ============================================================
 * DECODE PHASE 2 — Cross-chunk reduction
 *
 * One ESIMD thread per (request, head).
 * Reads all chunk partials, does log-sum-exp correction, stores final output.
 * ============================================================ */
ESIMD_INLINE void sdp_paged_decode_phase2(
    float* __restrict__ scratch_ptr,
    unsigned short* __restrict__ output_ptr,
    const int* __restrict__ seq_lens_ptr,
    int num_heads, int head_dim,
    int num_chunks_per_seq,
    nd_item<1>& ndi)
{
    const int global_id = ndi.get_global_id(0);
    const int head_idx = global_id % num_heads;
    const int req_idx = global_id / num_heads;

    const int seq_len = seq_lens_ptr[req_idx];
    if (seq_len <= 0) return;

    int actual_chunks = (seq_len + DEC_CHUNK_SIZE - 1) / DEC_CHUNK_SIZE;

    int base_scratch_idx = (req_idx * num_heads + head_idx) * num_chunks_per_seq;

    // Read first chunk
    float* s0 = scratch_ptr + base_scratch_idx * (DEC_SCRATCH_PER_CHUNK / 4);
    float global_max = s0[0];
    float global_sum = s0[1];

    const unsigned short* o0_bf16 = reinterpret_cast<const unsigned short*>(s0 + 2);
    simd<float, 64> acc0 = sdp_load_bf16_64(o0_bf16);
    simd<float, 64> acc1 = sdp_load_bf16_64(o0_bf16 + 64);
    simd<float, 64> acc2 = sdp_load_bf16_64(o0_bf16 + 128);
    simd<float, 64> acc3 = sdp_load_bf16_64(o0_bf16 + 192);

    // Merge remaining chunks
    for (int c = 1; c < actual_chunks; c++) {
        float* sc = scratch_ptr + (base_scratch_idx + c) * (DEC_SCRATCH_PER_CHUNK / 4);
        float chunk_max = sc[0];
        float chunk_sum = sc[1];

        const unsigned short* oc_bf16 = reinterpret_cast<const unsigned short*>(sc + 2);
        simd<float, 64> c0 = sdp_load_bf16_64(oc_bf16);
        simd<float, 64> c1 = sdp_load_bf16_64(oc_bf16 + 64);
        simd<float, 64> c2 = sdp_load_bf16_64(oc_bf16 + 128);
        simd<float, 64> c3 = sdp_load_bf16_64(oc_bf16 + 192);

        float new_max = (chunk_max > global_max) ? chunk_max : global_max;
        float corr_old = sdp_esimd_expf(global_max - new_max);
        float corr_new = sdp_esimd_expf(chunk_max - new_max);

        acc0 = acc0 * corr_old + c0 * corr_new;
        acc1 = acc1 * corr_old + c1 * corr_new;
        acc2 = acc2 * corr_old + c2 * corr_new;
        acc3 = acc3 * corr_old + c3 * corr_new;

        global_sum = global_sum * corr_old + chunk_sum * corr_new;
        global_max = new_max;
    }

    // Normalize
    if (global_sum > 0.0f) {
        float inv = 1.0f / global_sum;
        acc0 *= inv; acc1 *= inv; acc2 *= inv; acc3 *= inv;
    }

    // Store final output as bf16
    unsigned short* out_row = output_ptr +
        (int64_t)req_idx * num_heads * head_dim +
        (int64_t)head_idx * head_dim;

    sdp_store_bf16_64(out_row, acc0);
    sdp_store_bf16_64(out_row + 64, acc1);
    sdp_store_bf16_64(out_row + 128, acc2);
    sdp_store_bf16_64(out_row + 192, acc3);
}


/* ============================================================
 * DPAS-BASED PREFILL KERNEL — HD=256, bf16io, 32-thread WG
 *
 * Adapted from the 88 TFLOPS S^T architecture reference kernel.
 * 32-thread WG (8 sg_i × 4 sg_j).
 *   sg_i tiles KV tokens (16 per sg_i) and D dimension (32D per sg_i in VS).
 *   sg_j tiles Q rows (32 per sg_j).
 *
 * bf16io strategy:
 *   QK DPAS: dpas<8,8,float,float,bf16,bf16> — bf16 inputs, fp32 accum
 *   SxV DPAS: dpas<8,8,fp16,fp16,fp16,fp16> — fp16 inputs, fp16 accum
 *   V conversion: bf16→fp16 interleaved with SxV DPAS
 *
 * Paged K/V: per-token page table translation + 1D block_load.
 *   All 16 KV tokens per sg_i guaranteed within same block
 *   (block_size >= 64, KV_PER_SG=16, kv_pos always 16-aligned).
 *
 * Causal masking: per-Q-token kv_end applied after QK scores.
 * ============================================================ */

/* Constants for prefill DPAS kernel */
static constexpr uint32_t PF_HD = 256;
static constexpr uint32_t PF_HD_BLKS = 16;       // 256 / 16
static constexpr uint32_t PF_WG_Q_ROWS = 128;
static constexpr uint32_t PF_KV_CHUNK = 128;      // 8 sg_i × 16
static constexpr uint32_t PF_KV_PER_SG = 16;
static constexpr uint32_t PF_KV_BLKS = 8;
static constexpr uint32_t PF_Q_ROWS = 8;
static constexpr uint32_t PF_Q_PAIRS = 2;
static constexpr uint32_t PF_Q_TILES = 8;
static constexpr uint32_t PF_Q_GRPS = 4;
static constexpr uint32_t PF_D_BLKS_PER_SG = 2;

static constexpr uint32_t PF_Q_SLM_BASE   = 0x00000;  // 64 KB
static constexpr uint32_t PF_S_SLM_BASE   = 0x10000;  // 32 KB
static constexpr uint32_t PF_MAX_SLM_BASE = 0x18000;  // 4 KB
static constexpr uint32_t PF_SUM_SLM_BASE = 0x19000;  // 4 KB
static constexpr uint32_t PF_TOTAL_SLM    = 0x1A000;  // 104 KB total


ESIMD_INLINE void sdp_paged_prefill_dpas(
    const unsigned short* __restrict__ query_ptr,
    const unsigned short* __restrict__ kv_cache_ptr,
    unsigned short* __restrict__ output_ptr,
    const int* __restrict__ block_table_ptr,
    const int* __restrict__ seq_lens_ptr,
    const int* __restrict__ query_start_loc_ptr,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_split, int64_t kv_stride_block,
    int64_t kv_stride_pos, int64_t kv_stride_head,
    float attn_scale,
    int causal,
    int num_tokens,
    int max_q_tiles_per_req,
    int batch,
    nd_item<1>& ndi)
{
    constexpr float LOG2E = sycl::ext::intel::esimd::detail::log2e;
    const float attnScoreMul = attn_scale * LOG2E;

    __ESIMD_NS::slm_init(PF_TOTAL_SLM);
    __esimd_nbarrier_init(1);

    int32_t tid = ndi.get_local_id(0);
    int32_t sg_i = tid & 7;
    int32_t sg_j = tid >> 3;
    int32_t wg_id = ndi.get_group(0);

    int32_t head_idx = wg_id % num_heads;
    int32_t temp = wg_id / num_heads;
    int32_t req_idx = temp / max_q_tiles_per_req;
    int32_t q_tile_idx = temp % max_q_tiles_per_req;

    if (req_idx >= batch) return;

    int32_t req_q_start = query_start_loc_ptr[req_idx];
    int32_t req_q_end = query_start_loc_ptr[req_idx + 1];
    int32_t req_query_len = req_q_end - req_q_start;
    int32_t q_offset = q_tile_idx * PF_WG_Q_ROWS;

    if (q_offset >= req_query_len) return;

    int32_t actual_q_rows = req_query_len - q_offset;
    if (actual_q_rows > (int)PF_WG_Q_ROWS) actual_q_rows = PF_WG_Q_ROWS;

    int32_t seq_len = seq_lens_ptr[req_idx];
    if (seq_len <= 0) return;

    int32_t group_size = num_heads / num_kv_heads;
    int32_t kv_head_idx = head_idx / group_size;
    int64_t kv_head_offset = (int64_t)kv_head_idx * kv_stride_head;
    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    int32_t max_valid_blk_idx = (seq_len - 1) / block_size;

    int32_t q_global_start = req_q_start + q_offset;
    int32_t q_abs_base = (seq_len - req_query_len) + q_offset;

    // 2D surface parameters for per-block KV access
    // Within a block, positions are contiguous with stride kv_stride_pos
    uint32_t kv_row_bytes = (uint32_t)(kv_stride_pos * 2);  // bytes per KV position row
    uint32_t kv_surf_w = kv_row_bytes - 1;
    uint32_t kv_surf_h = (uint32_t)(block_size - 1);
    uint32_t kv_x_k = (uint32_t)kv_head_offset;  // X offset for K loads (in fp16/bf16 elements)

    // ============================================================
    // COOPERATIVE Q LOAD TO SLM
    // ============================================================
    {
        uint32_t widthInByteQ = num_heads * PF_HD * sizeof(bf16) - 1;
        uint32_t heightQ = num_tokens - 1;

        __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
            (uint32_t*)query_ptr, widthInByteQ, heightQ, widthInByteQ, 0, 0);

        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = tid * 4 + t;
            int d_blk = tile_id >> 3;
            int q_tile = tile_id & 7;

            payloadQ.set_x((head_idx * (int)(PF_HD / 2)) + d_blk * 8);
            payloadQ.set_y(q_global_start + q_tile * 16);

            simd<uint32_t, 128> qTile = __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);

            uint32_t slm_off = PF_Q_SLM_BASE + tile_id * 512;
            slm_block_store<uint32_t, 64>(slm_off, qTile.select<64, 1>(0));
            slm_block_store<uint32_t, 64>(slm_off + 256, qTile.select<64, 1>(64));
        }
    }

    barrier();

    // ============================================================
    // REGISTER DECLARATIONS (matching reference: ST_next for pipelining)
    // ============================================================
    simd<float, 1024> A_tile = 0;
    simd<float, 512> ST_tile;
    simd<float, 512> ST_next;
    simd<float, 32> fp32_max = FP32_MIN;
    simd<float, 32> fp32_sum = 0;
    simd<float, 32> delta;

    int32_t max_kv_end;
    if (causal) {
        max_kv_end = q_abs_base + actual_q_rows;
        if (max_kv_end > seq_len) max_kv_end = seq_len;
    } else {
        max_kv_end = seq_len;
    }
    int32_t kvOuterLoops = (max_kv_end + PF_KV_CHUNK - 1) / PF_KV_CHUNK;
    if (kvOuterLoops <= 0) kvOuterLoops = 1;

    // Persistent K payload for paged block loads
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
        (fp16*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, 0);

    // ============================================================
    // PROLOGUE: QK[0] -> ST_tile
    // ============================================================
    {
        int32_t kv_pos = sg_i * PF_KV_PER_SG;
        int32_t blk_idx = kv_pos / block_size;
        if (blk_idx > max_valid_blk_idx) blk_idx = 0;
        int32_t blk_off = kv_pos & (block_size - 1);
        int32_t blk_num = block_table_row[blk_idx];

        ST_tile = 0;

        payloadK = __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1>(
            (fp16*)(kv_cache_ptr + (int64_t)blk_num * kv_stride_block),
            kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, blk_off);
        simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS; d++) {
            simd<bf16, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<bf16>().data());
            simd<bf16, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<bf16>().data());

            uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
            uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

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

            if (d < (int)PF_HD_BLKS - 1) {
                payloadK.set_x(kv_x_k + (d + 1) * 16);
                K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
            }
        }
    }

    // ============================================================
    // OUTER KV LOOP — ST_tile has QK[outerIter] scores on entry
    // ============================================================
    for (int32_t outerIter = 0; outerIter < kvOuterLoops; outerIter++) {
        uint32_t kv_start = outerIter * PF_KV_CHUNK;

        // ========================================
        // SOFTMAX FIRST HALF: scale scores, apply masks
        // ========================================
        ST_tile *= attnScoreMul;

        if (outerIter == kvOuterLoops - 1) {
            uint32_t kv_base_sg = kv_start + sg_i * PF_KV_PER_SG;
            #pragma unroll
            for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
                #pragma unroll
                for (int kv = 0; kv < 8; kv++) {
                    if (kv_base_sg + kv >= (uint32_t)max_kv_end)
                        ST_tile.select<16, 1>(qp * 256 + kv * 16) = FP32_MIN;
                    if (kv_base_sg + 8 + kv >= (uint32_t)max_kv_end)
                        ST_tile.select<16, 1>(qp * 256 + 128 + kv * 16) = FP32_MIN;
                }
            }
        }

        if (causal) {
            #pragma unroll
            for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
                #pragma unroll
                for (int qr = 0; qr < 16; qr++) {
                    int q_local = sg_j * 32 + qp * 16 + qr;
                    int q_abs = q_abs_base + q_local;
                    int kv_end_for_q = q_abs + 1;

                    uint32_t kv_base_sg = kv_start + sg_i * PF_KV_PER_SG;
                    #pragma unroll
                    for (int kv = 0; kv < 8; kv++) {
                        if ((int)(kv_base_sg + kv) >= kv_end_for_q)
                            ST_tile[qp * 256 + kv * 16 + qr] = FP32_MIN;
                        if ((int)(kv_base_sg + 8 + kv) >= kv_end_for_q)
                            ST_tile[qp * 256 + 128 + kv * 16 + qr] = FP32_MIN;
                    }
                }
            }
        }

        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            #pragma unroll
            for (int qr = 0; qr < 16; qr++) {
                int q_local = sg_j * 32 + qp * 16 + qr;
                if (q_local >= actual_q_rows) {
                    #pragma unroll
                    for (int kv = 0; kv < 16; kv++) {
                        ST_tile[qp * 256 + kv * 16 + qr] = FP32_MIN;
                    }
                }
            }
        }

        simd<float, 32> local_max;
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
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

        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            uint32_t q_base = sg_j * 32 + qp * 16;
            slm_block_store<float, 16>(PF_MAX_SLM_BASE + (sg_i * 128 + q_base) * 4,
                local_max.select<16, 1>(qp * 16));
        }

        // ========================================
        // BARRIER A: arrive, QK[k+1] overlap, wait
        // ========================================
        __esimd_nbarrier_arrive(0, 0, 32, 32);

        if (outerIter < kvOuterLoops - 1) {
            uint32_t next_kv_start = (outerIter + 1) * PF_KV_CHUNK;
            int32_t next_kv_pos = next_kv_start + sg_i * PF_KV_PER_SG;
            int32_t next_blk_idx = next_kv_pos / block_size;
            if (next_blk_idx > max_valid_blk_idx) next_blk_idx = 0;
            int32_t next_blk_off = next_kv_pos & (block_size - 1);
            int32_t next_blk_num = block_table_row[next_blk_idx];

            ST_next = 0;

            payloadK = __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1>(
                (fp16*)(kv_cache_ptr + (int64_t)next_blk_num * kv_stride_block),
                kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, next_blk_off);
            simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

            #pragma unroll
            for (int d = 0; d < (int)PF_HD_BLKS; d++) {
                simd<bf16, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<bf16>().data());
                simd<bf16, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<bf16>().data());

                uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
                uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

                simd<bf16, 256> Q_vnni0, Q_vnni1;
                Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(0) =
                    slm_block_load<uint32_t, 64>(q_slm_off0);
                Q_vnni0.template bit_cast_view<uint32_t>().select<64, 1>(64) =
                    slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
                Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(0) =
                    slm_block_load<uint32_t, 64>(q_slm_off1);
                Q_vnni1.template bit_cast_view<uint32_t>().select<64, 1>(64) =
                    slm_block_load<uint32_t, 64>(q_slm_off1 + 256);

                { auto acc = ST_next.select<128, 1>(0);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni0.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(128);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni0.data()), K_sb1); }
                { auto acc = ST_next.select<128, 1>(256);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni1.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(384);
                  acc = dpas<8, 8, float, float, bf16, bf16>(simd<float, 128>(acc.data()), simd<bf16, 256>(Q_vnni1.data()), K_sb1); }

                if (d < (int)PF_HD_BLKS - 1) {
                    payloadK.set_x(kv_x_k + (d + 1) * 16);
                    K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
                }
            }
        }

        __esimd_nbarrier(0, 0, 32);

        // ========================================
        // SOFTMAX SECOND HALF
        // ========================================
        simd<float, 32> global_max = FP32_MIN;
        #pragma unroll
        for (int si = 0; si < 8; si++) {
            #pragma unroll
            for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
                uint32_t q_base = sg_j * 32 + qp * 16;
                simd<float, 16> m = slm_block_load<float, 16>(PF_MAX_SLM_BASE + (si * 128 + q_base) * 4);
                global_max.select<16, 1>(qp * 16) = __ESIMD_NS::max<float, 16, float>(
                    global_max.select<16, 1>(qp * 16), m);
            }
        }
        global_max = __ESIMD_NS::max<float, 32, float>(global_max, fp32_max);

        delta = __ESIMD_NS::exp2<float, 32, float>(fp32_max - global_max);
        fp32_max = global_max;

        simd<float, 32> local_sum = 0;
        simd<bf16, 256> ST_bf16_0;
        simd<bf16, 256> ST_bf16_1;

        {
            simd<float, 16> gm = global_max.select<16, 1>(0);
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(kv * 16) - gm);
                ST_bf16_0.select<16, 1>(kv * 16) = s;
                local_sum.select<16, 1>(0) += s;
            }
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(128 + kv * 16) - gm);
                ST_bf16_0.select<16, 1>(128 + kv * 16) = s;
                local_sum.select<16, 1>(0) += s;
            }
        }
        {
            simd<float, 16> gm = global_max.select<16, 1>(16);
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + kv * 16) - gm);
                ST_bf16_1.select<16, 1>(kv * 16) = s;
                local_sum.select<16, 1>(16) += s;
            }
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + 128 + kv * 16) - gm);
                ST_bf16_1.select<16, 1>(128 + kv * 16) = s;
                local_sum.select<16, 1>(16) += s;
            }
        }

        // ========================================
        // S^T SCATTER TO SLM
        // ========================================
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            auto& ST_bf16 = (qp == 0) ? ST_bf16_0 : ST_bf16_1;
            simd<uint16_t, 256> ST_fp16_u16 = ST_bf16.template bit_cast_view<uint16_t>();

            uint32_t tile_addr_qh0 = PF_S_SLM_BASE + (sg_i * 16 + sg_j * PF_Q_GRPS + qp * 2 + 0) * 256;
            uint32_t tile_addr_qh1 = PF_S_SLM_BASE + (sg_i * 16 + sg_j * PF_Q_GRPS + qp * 2 + 1) * 256;

            simd<uint32_t, 16> q_offsets;
            #pragma unroll
            for (int q = 0; q < 8; q++)
                q_offsets[q] = tile_addr_qh0 + q * 32;
            #pragma unroll
            for (int q = 0; q < 8; q++)
                q_offsets[8 + q] = tile_addr_qh1 + q * 32;

            #pragma unroll
            for (int kg = 0; kg < 4; kg++) {
                int kv_base = kg * 4;
                simd<uint32_t, 16> packed0 =
                    simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 0) * 16)) |
                    (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 1) * 16)) << 16);
                simd<uint32_t, 16> packed1 =
                    simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 2) * 16)) |
                    (simd<uint32_t, 16>(ST_fp16_u16.select<16, 1>((kv_base + 3) * 16)) << 16);
                simd<uint32_t, 32> data;
                data.select<16, 1>(0) = packed0;
                data.select<16, 1>(16) = packed1;
                __ESIMD_ENS::lsc_slm_scatter<uint32_t, 2, __ESIMD_ENS::lsc_data_size::u32, 16>(
                    q_offsets + kv_base * 2, data);
            }
        }

        // ========================================
        // BARRIER B: arrive, V loads + compensation, wait
        // ========================================
        __esimd_nbarrier_arrive(0, 0, 32, 32);

        fp32_sum = fp32_sum * delta + local_sum;

        // Load V for kv_blk=0 via per-block 2D surface
        int32_t v_kv_pos_0 = kv_start;
        int32_t v_blk_idx_0 = v_kv_pos_0 / block_size;
        if (v_blk_idx_0 > max_valid_blk_idx) v_blk_idx_0 = 0;
        int32_t v_blk_off_0 = v_kv_pos_0 & (block_size - 1);
        int32_t v_blk_num_0 = block_table_row[v_blk_idx_0];

        // V surface: offset by kv_stride_split for value cache
        // Use VNNI transform (last template arg = true) to get bf16 VNNI-packed directly
        __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadV(
            (fp16*)(kv_cache_ptr + (int64_t)v_blk_num_0 * kv_stride_block + kv_stride_split),
            kv_surf_w, kv_surf_h, kv_surf_w,
            (uint32_t)(kv_head_offset + sg_i * 32), v_blk_off_0);

        simd<fp16, 256> V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
        payloadV.set_x((uint32_t)(kv_head_offset + sg_i * 32 + 16));
        simd<fp16, 256> V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

        // Compensation (fp32 multiply — A_tile is fp32)
        #pragma unroll
        for (int qg = 0; qg < (int)PF_Q_GRPS; qg++) {
            #pragma unroll
            for (int db = 0; db < (int)PF_D_BLKS_PER_SG; db++) {
                #pragma unroll
                for (int q = 0; q < (int)PF_Q_ROWS; q++) {
                    float d_val = delta[qg * PF_Q_ROWS + q];
                    A_tile.select<16, 1>((qg * PF_D_BLKS_PER_SG + db) * 128 + q * 16) *= d_val;
                }
            }
        }

        __esimd_nbarrier(0, 0, 32);

        // ========================================
        // VS PHASE — reuse V surface across kv_blks in same page
        // Note: no #pragma unroll — reduces register pressure
        // ========================================
        for (int kv_blk = 0; kv_blk < (int)PF_KV_BLKS; kv_blk++) {
            // Load V for this kv_blk
            if (kv_blk > 0) {
                int32_t v_kv_pos = kv_start + kv_blk * 16;
                int32_t v_bo = v_kv_pos & (block_size - 1);

                // Reuse V surface from kv_blk=0, just update Y
                payloadV.set_x((uint32_t)(kv_head_offset + sg_i * 32));
                payloadV.set_y(v_bo);
                V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
                    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
                payloadV.set_x((uint32_t)(kv_head_offset + sg_i * 32 + 16));
                V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
                    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
            }

            uint32_t s_base = PF_S_SLM_BASE + (kv_blk * 16 + sg_j * PF_Q_GRPS) * 256;
            simd<bf16, 128> S0, S1, S2, S3;
            S0.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base);
            S1.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 256);
            S2.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 512);
            S3.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(s_base + 768);

            auto V_bf16_0 = V_vnni0.template bit_cast_view<bf16>();
            auto V_bf16_1 = V_vnni1.template bit_cast_view<bf16>();

            { auto acc = A_tile.select<128, 1>(0 * 128);
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

        ST_tile = ST_next;
    }

    // ============================================================
    // FINAL OUTPUT: normalize and store as bf16
    // ============================================================
    #pragma unroll
    for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
        uint32_t q_base = sg_j * 32 + qp * 16;
        slm_block_store<float, 16>(PF_SUM_SLM_BASE + (sg_i * 128 + q_base) * 4,
            fp32_sum.select<16, 1>(qp * 16));
    }

    barrier();

    simd<float, 32> total_sum = 0;
    #pragma unroll
    for (int si = 0; si < 8; si++) {
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            uint32_t q_base = sg_j * 32 + qp * 16;
            total_sum.select<16, 1>(qp * 16) += slm_block_load<float, 16>(
                PF_SUM_SLM_BASE + (si * 128 + q_base) * 4);
        }
    }

    simd<float, 32> inv_sum;
    inv_sum.select<16, 1>(0) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(0));
    inv_sum.select<16, 1>(16) = __ESIMD_NS::inv<float, 16>(total_sum.select<16, 1>(16));

    uint32_t d_start = sg_i * PF_D_BLKS_PER_SG * 16;

    uint32_t outW = num_heads * PF_HD * sizeof(bf16) - 1;
    uint32_t outH = num_tokens - 1;
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 8, 1> payloadO(
        (fp16*)output_ptr, outW, outH, outW, 0, 0);

    #pragma unroll
    for (int qg = 0; qg < (int)PF_Q_GRPS; qg++) {
        #pragma unroll
        for (int db = 0; db < (int)PF_D_BLKS_PER_SG; db++) {
            // A_tile is fp32 — normalize and convert to bf16 for output
            simd<fp16, 128> fOut_fp16;

            #pragma unroll
            for (int q = 0; q < (int)PF_Q_ROWS; q++) {
                simd<float, 16> f32_out = A_tile.select<16, 1>((qg * PF_D_BLKS_PER_SG + db) * 128 + q * 16);
                float inv = inv_sum[qg * PF_Q_ROWS + q];
                f32_out *= inv;
                simd<bf16, 16> bf16_out = f32_out;
                fOut_fp16.select<16, 1>(q * 16) = bf16_out.template bit_cast_view<fp16>();
            }

            int q_row_in_tile = sg_j * 32 + qg * PF_Q_ROWS;
            int q_global_row = q_global_start + q_row_in_tile;

            if (q_row_in_tile + PF_Q_ROWS <= (uint32_t)actual_q_rows) {
                payloadO.set_x(head_idx * PF_HD + d_start + db * 16);
                payloadO.set_y(q_global_row);
                __ESIMD_ENS::lsc_store_2d<fp16, 16, 8, 1,
                    __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back>(payloadO, fOut_fp16);
            } else if (q_row_in_tile < actual_q_rows) {
                int valid_rows = actual_q_rows - q_row_in_tile;
                unsigned short* out_base = output_ptr +
                    (int64_t)q_global_row * num_heads * PF_HD +
                    (int64_t)head_idx * PF_HD + d_start + db * 16;
                int64_t row_stride = (int64_t)num_heads * PF_HD;
                for (int r = 0; r < valid_rows; r++) {
                    block_store<unsigned short, 16>(
                        out_base + (int64_t)r * row_stride,
                        fOut_fp16.select<16, 1>(r * 16).template bit_cast_view<unsigned short>());
                }
            }
        }
    }
}
