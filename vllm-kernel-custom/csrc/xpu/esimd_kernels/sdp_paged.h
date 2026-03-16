/* sdp_paged.h — Optimized paged SDP kernels (HD=256, bf16io).
 *
 * Five kernels:
 *   1. sdp_paged_kernel_scalar      — Scalar fallback for decode/prefill (all HD)
 *   2. sdp_paged_decode_phase1      — Per-chunk partial softmax for decode (legacy)
 *   3. sdp_paged_decode_phase2      — Cross-chunk log-sum-exp reduction
 *   4. sdp_paged_decode_gqa_phase1  — GQA-optimized decode: Q_HEAD_PER_T=4, K/V shared
 *   5. sdp_paged_prefill_dpas       — DPAS-based prefill (HD=256, 32-thread WG)
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
static constexpr int DEC_GQA_GROUP_SIZE = 4;   // Q heads per KV head (Qwen3.5-4B: 16Q/4KV)
static constexpr int DEC_GQA_CHUNK_SIZE = 64;  // KV tokens per chunk (tunable: 32/64/128)

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

    // Use num_chunks_per_seq from launcher (matches GQA chunk size)
    int actual_chunks = num_chunks_per_seq;
    // But clamp to actual sequence length
    {
        int seq_chunks = (seq_len + DEC_GQA_CHUNK_SIZE - 1) / DEC_GQA_CHUNK_SIZE;
        if (seq_chunks < actual_chunks) actual_chunks = seq_chunks;
    }

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
 * GQA-OPTIMIZED DECODE PHASE 1 — Single-thread, online softmax
 *
 * One thread per (request, kv_head, chunk).
 * Each thread processes DEC_GQA_CHUNK_SIZE KV tokens for 4 Q heads.
 * K/V loaded once, dot product computed for all 4 Q heads → 4× BW savings.
 * Online softmax: single pass loads K+V together.
 * ============================================================ */
ESIMD_INLINE void sdp_paged_decode_gqa_phase1(
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
    constexpr int QHT = DEC_GQA_GROUP_SIZE;    // 4
    constexpr int HD  = 256;

    int tid = ndi.get_global_id(0);

    int chunk_id = tid % num_chunks_per_seq;
    int temp = tid / num_chunks_per_seq;
    int kv_head_idx = temp % num_kv_heads;
    int req_idx = temp / num_kv_heads;

    int group_size = num_heads / num_kv_heads;
    int q_head_start = kv_head_idx * group_size;
    int seq_len = seq_lens_ptr[req_idx];

    // This thread's KV range
    int kv_start = chunk_id * DEC_GQA_CHUNK_SIZE;
    int kv_end = kv_start + DEC_GQA_CHUNK_SIZE;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_start >= seq_len) kv_end = kv_start;

    int64_t kv_head_offset = (int64_t)kv_head_idx * kv_stride_head;
    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;

    // Load Q for all 4 heads (scaled)
    const unsigned short* q_base = query_ptr +
        (int64_t)req_idx * num_heads * head_dim;

    simd<float, 64> q0_a = sdp_load_bf16_64(q_base + (q_head_start + 0) * head_dim) * attn_scale;
    simd<float, 64> q0_b = sdp_load_bf16_64(q_base + (q_head_start + 0) * head_dim + 64) * attn_scale;
    simd<float, 64> q0_c = sdp_load_bf16_64(q_base + (q_head_start + 0) * head_dim + 128) * attn_scale;
    simd<float, 64> q0_d = sdp_load_bf16_64(q_base + (q_head_start + 0) * head_dim + 192) * attn_scale;

    simd<float, 64> q1_a = sdp_load_bf16_64(q_base + (q_head_start + 1) * head_dim) * attn_scale;
    simd<float, 64> q1_b = sdp_load_bf16_64(q_base + (q_head_start + 1) * head_dim + 64) * attn_scale;
    simd<float, 64> q1_c = sdp_load_bf16_64(q_base + (q_head_start + 1) * head_dim + 128) * attn_scale;
    simd<float, 64> q1_d = sdp_load_bf16_64(q_base + (q_head_start + 1) * head_dim + 192) * attn_scale;

    simd<float, 64> q2_a = sdp_load_bf16_64(q_base + (q_head_start + 2) * head_dim) * attn_scale;
    simd<float, 64> q2_b = sdp_load_bf16_64(q_base + (q_head_start + 2) * head_dim + 64) * attn_scale;
    simd<float, 64> q2_c = sdp_load_bf16_64(q_base + (q_head_start + 2) * head_dim + 128) * attn_scale;
    simd<float, 64> q2_d = sdp_load_bf16_64(q_base + (q_head_start + 2) * head_dim + 192) * attn_scale;

    simd<float, 64> q3_a = sdp_load_bf16_64(q_base + (q_head_start + 3) * head_dim) * attn_scale;
    simd<float, 64> q3_b = sdp_load_bf16_64(q_base + (q_head_start + 3) * head_dim + 64) * attn_scale;
    simd<float, 64> q3_c = sdp_load_bf16_64(q_base + (q_head_start + 3) * head_dim + 128) * attn_scale;
    simd<float, 64> q3_d = sdp_load_bf16_64(q_base + (q_head_start + 3) * head_dim + 192) * attn_scale;

    // Online softmax state for 4 heads
    float mx0 = FP32_MIN, mx1 = FP32_MIN, mx2 = FP32_MIN, mx3 = FP32_MIN;
    float lse0 = 0, lse1 = 0, lse2 = 0, lse3 = 0;
    simd<float, 64> a0_a(0), a0_b(0), a0_c(0), a0_d(0);
    simd<float, 64> a1_a(0), a1_b(0), a1_c(0), a1_d(0);
    simd<float, 64> a2_a(0), a2_b(0), a2_c(0), a2_d(0);
    simd<float, 64> a3_a(0), a3_b(0), a3_c(0), a3_d(0);

    for (int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        int blk_idx = kv_pos / block_size;
        int blk_off = kv_pos & (block_size - 1);
        int blk_num = block_table_row[blk_idx];

        int64_t kv_base_off = (int64_t)blk_num * kv_stride_block +
                              (int64_t)blk_off * kv_stride_pos + kv_head_offset;

        // Load K
        const unsigned short* k_ptr = kv_cache_ptr + kv_base_off;
        simd<float, 64> k_a = sdp_load_bf16_64(k_ptr);
        simd<float, 64> k_b = sdp_load_bf16_64(k_ptr + 64);
        simd<float, 64> k_c = sdp_load_bf16_64(k_ptr + 128);
        simd<float, 64> k_d = sdp_load_bf16_64(k_ptr + 192);

        // QK dot products (scale already in Q)
        float s0 = sdp_dot256(q0_a, q0_b, q0_c, q0_d, k_a, k_b, k_c, k_d);
        float s1 = sdp_dot256(q1_a, q1_b, q1_c, q1_d, k_a, k_b, k_c, k_d);
        float s2 = sdp_dot256(q2_a, q2_b, q2_c, q2_d, k_a, k_b, k_c, k_d);
        float s3 = sdp_dot256(q3_a, q3_b, q3_c, q3_d, k_a, k_b, k_c, k_d);

        // Load V
        const unsigned short* v_ptr = kv_cache_ptr + kv_base_off + kv_stride_split;
        simd<float, 64> v_a = sdp_load_bf16_64(v_ptr);
        simd<float, 64> v_b = sdp_load_bf16_64(v_ptr + 64);
        simd<float, 64> v_c = sdp_load_bf16_64(v_ptr + 128);
        simd<float, 64> v_d = sdp_load_bf16_64(v_ptr + 192);

        // Online softmax + V accumulation for each head
        // Head 0
        float old_mx0 = mx0;
        if (s0 > mx0) mx0 = s0;
        float corr0 = sdp_esimd_expf(old_mx0 - mx0);
        float w0 = sdp_esimd_expf(s0 - mx0);
        a0_a = a0_a * corr0 + v_a * w0; a0_b = a0_b * corr0 + v_b * w0;
        a0_c = a0_c * corr0 + v_c * w0; a0_d = a0_d * corr0 + v_d * w0;
        lse0 = lse0 * corr0 + w0;

        // Head 1
        float old_mx1 = mx1;
        if (s1 > mx1) mx1 = s1;
        float corr1 = sdp_esimd_expf(old_mx1 - mx1);
        float w1 = sdp_esimd_expf(s1 - mx1);
        a1_a = a1_a * corr1 + v_a * w1; a1_b = a1_b * corr1 + v_b * w1;
        a1_c = a1_c * corr1 + v_c * w1; a1_d = a1_d * corr1 + v_d * w1;
        lse1 = lse1 * corr1 + w1;

        // Head 2
        float old_mx2 = mx2;
        if (s2 > mx2) mx2 = s2;
        float corr2 = sdp_esimd_expf(old_mx2 - mx2);
        float w2 = sdp_esimd_expf(s2 - mx2);
        a2_a = a2_a * corr2 + v_a * w2; a2_b = a2_b * corr2 + v_b * w2;
        a2_c = a2_c * corr2 + v_c * w2; a2_d = a2_d * corr2 + v_d * w2;
        lse2 = lse2 * corr2 + w2;

        // Head 3
        float old_mx3 = mx3;
        if (s3 > mx3) mx3 = s3;
        float corr3 = sdp_esimd_expf(old_mx3 - mx3);
        float w3 = sdp_esimd_expf(s3 - mx3);
        a3_a = a3_a * corr3 + v_a * w3; a3_b = a3_b * corr3 + v_b * w3;
        a3_c = a3_c * corr3 + v_c * w3; a3_d = a3_d * corr3 + v_d * w3;
        lse3 = lse3 * corr3 + w3;
    }

    // Write to global scratch for all 4 Q heads
    float maxes[4] = {mx0, mx1, mx2, mx3};
    float lses[4] = {lse0, lse1, lse2, lse3};

    #pragma unroll
    for (int h = 0; h < QHT; h++) {
        int scratch_idx = (req_idx * num_heads + q_head_start + h) * num_chunks_per_seq + chunk_id;
        float* scratch_base = scratch_ptr + scratch_idx * (DEC_SCRATCH_PER_CHUNK / 4);
        scratch_base[0] = maxes[h];
        scratch_base[1] = lses[h];

        unsigned short* out_bf16 = reinterpret_cast<unsigned short*>(scratch_base + 2);
        if (h == 0) { sdp_store_bf16_64(out_bf16, a0_a); sdp_store_bf16_64(out_bf16 + 64, a0_b);
                      sdp_store_bf16_64(out_bf16 + 128, a0_c); sdp_store_bf16_64(out_bf16 + 192, a0_d); }
        if (h == 1) { sdp_store_bf16_64(out_bf16, a1_a); sdp_store_bf16_64(out_bf16 + 64, a1_b);
                      sdp_store_bf16_64(out_bf16 + 128, a1_c); sdp_store_bf16_64(out_bf16 + 192, a1_d); }
        if (h == 2) { sdp_store_bf16_64(out_bf16, a2_a); sdp_store_bf16_64(out_bf16 + 64, a2_b);
                      sdp_store_bf16_64(out_bf16 + 128, a2_c); sdp_store_bf16_64(out_bf16 + 192, a2_d); }
        if (h == 3) { sdp_store_bf16_64(out_bf16, a3_a); sdp_store_bf16_64(out_bf16 + 64, a3_b);
                      sdp_store_bf16_64(out_bf16 + 128, a3_c); sdp_store_bf16_64(out_bf16 + 192, a3_d); }
    }
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


template<bool CAUSAL>
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

    // Convert int64 strides to uint32 derived values immediately — free int64 regs
    uint32_t kv_head_off_u32 = (uint32_t)((int64_t)kv_head_idx * kv_stride_head);
    uint32_t kv_row_bytes = (uint32_t)(kv_stride_pos * 2);
    int32_t num_blocks_total = (int32_t)(kv_stride_split / kv_stride_block);
    const unsigned short* kv_v_base = kv_cache_ptr + kv_stride_split;

    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    // block_size is always power of 2 — use shift for division
    int32_t block_size_shift = __builtin_ctz(block_size);
    int32_t block_size_mask = block_size - 1;
    int32_t max_valid_blk_idx = (seq_len - 1) >> block_size_shift;

// Experiment: bypass block table to measure overhead
#ifdef PAGED_BYPASS_BLOCK_TABLE
#define BLK_TABLE_LOAD(idx) (idx)
#else
#define BLK_TABLE_LOAD(idx) block_table_row[(idx)]
#endif

    int32_t q_global_start = req_q_start + q_offset;
    int32_t q_abs_base = (seq_len - req_query_len) + q_offset;

    // 2D surface parameters — FIXED BASE for entire KV cache
    uint32_t kv_surf_w = kv_row_bytes - 1;
    uint32_t kv_surf_h = (uint32_t)((num_blocks_total << block_size_shift) - 1);
    uint32_t kv_x_k = kv_head_off_u32;

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
    // REGISTER DECLARATIONS
    // ============================================================
    simd<float, 1024> A_tile = 0;
    simd<float, 512> ST_tile;
    simd<float, 512> ST_next;
    simd<float, 32> fp32_max = FP32_MIN;
    simd<float, 32> fp32_sum = 0;
    simd<float, 32> delta;

    int32_t max_kv_end;
    if constexpr (CAUSAL) {
        max_kv_end = q_abs_base + actual_q_rows;
        if (max_kv_end > seq_len) max_kv_end = seq_len;
    } else {
        max_kv_end = seq_len;
    }
    int32_t kvOuterLoops = (max_kv_end + PF_KV_CHUNK - 1) / PF_KV_CHUNK;
    if (kvOuterLoops <= 0) kvOuterLoops = 1;

    // Pre-compute causal boundaries (CAUSAL only — noncausal skips entirely, saves 32 regs)
    simd<int32_t, 16> causal_bound_0, causal_bound_1;
    if constexpr (CAUSAL) {
        simd<int32_t, 16> q_idx_vec;
        #pragma unroll
        for (int qr = 0; qr < 16; qr++) q_idx_vec[qr] = qr;

        simd<int32_t, 16> q_local_0 = sg_j * 32 + q_idx_vec;
        simd<int32_t, 16> q_local_1 = sg_j * 32 + 16 + q_idx_vec;

        causal_bound_0 = q_abs_base + q_local_0;
        causal_bound_1 = q_abs_base + q_local_1;
        causal_bound_0.merge(-1, q_local_0 >= actual_q_rows);
        causal_bound_1.merge(-1, q_local_1 >= actual_q_rows);
    }

    // Fixed-base payloads — configured ONCE, only set_x/set_y per iteration
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
        (fp16*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, 0);

    __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadKpf(
        (uint32_t*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, (uint32_t)(kv_x_k / 2), 0);

    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadV(
        (fp16*)(kv_v_base), kv_surf_w, kv_surf_h, kv_surf_w,
        (uint32_t)(kv_head_off_u32 + sg_i * 32), 0);

    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadVpf(
        (uint32_t*)(kv_v_base), kv_surf_w, kv_surf_h, kv_surf_w,
        (uint32_t)(kv_head_off_u32 / 2 + sg_i * 16), 0);

    // ============================================================
    // PROLOGUE K PREFETCH: prefetch K[0] + K[1]
    // Per-iteration block lookup: one block_table_row[] load per chunk.
    // ============================================================
    {
        int32_t pf0_phys = BLK_TABLE_LOAD(0);
        payloadKpf.set_y((uint32_t)((pf0_phys << block_size_shift) + sg_i * (int32_t)PF_KV_PER_SG));
        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS; d++) {
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + d * 8));
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
        }
    }
    if (kvOuterLoops > 1) {
        int32_t pf1_logical = (int32_t)PF_KV_CHUNK >> block_size_shift;
        int32_t pf1_off = (int32_t)PF_KV_CHUNK & block_size_mask;
        int32_t pf1_phys = BLK_TABLE_LOAD(pf1_logical);
        payloadKpf.set_y((uint32_t)((pf1_phys << block_size_shift) + pf1_off + sg_i * (int32_t)PF_KV_PER_SG));
        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS; d++) {
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + d * 8));
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadKpf);
        }
    }

    // ============================================================
    // PROLOGUE: QK[0] -> ST_tile, with V[0] prefetch interleaved
    // D-loop peeled: main loop d=0..PF_HD_BLKS-2, then last D-block
    // ============================================================
    {
        int32_t phys0 = BLK_TABLE_LOAD(0);
        uint32_t Y_base_K = (uint32_t)((phys0 << block_size_shift) + sg_i * PF_KV_PER_SG);
        uint32_t Y_base_V = (uint32_t)(phys0 << block_size_shift);

        ST_tile = 0;

        payloadK.set_y(Y_base_K);
        simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

        #pragma unroll
        for (int d = 0; d < (int)PF_HD_BLKS - 1; d++) {
            // V prefetch: half rate (every other D-block) — reduces memory pressure
            if ((d & 1) == 0) {
            payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
            payloadVpf.set_y(Y_base_V + (d >> 1) * 16);
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>(payloadVpf);
            }

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

            payloadK.set_x(kv_x_k + (d + 1) * 16);
            K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }

        // Last D-block (d = PF_HD_BLKS - 1): no next-K load
        {
            constexpr int d = (int)PF_HD_BLKS - 1;
            // V prefetch: half rate (every other D-block) — reduces memory pressure
            if ((d & 1) == 0) {
            payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
            payloadVpf.set_y(Y_base_V + (d >> 1) * 16);
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>(payloadVpf);
            }

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
        }
    }

    // ============================================================
    // OUTER KV LOOP — single loop, ST_tile has QK[outerIter] scores on entry
    // ============================================================
    for (int32_t outerIter = 0; outerIter < kvOuterLoops; outerIter++) {
        uint32_t kv_start = outerIter * PF_KV_CHUNK;

        // ========================================
        // SOFTMAX FIRST HALF: scale scores, apply masks
        // ========================================
        ST_tile *= attnScoreMul;

        // Masking: CAUSAL applies causal_bound every iteration;
        // noncausal only masks the last iteration for kv_pos >= seq_len and invalid Q rows.
        if constexpr (CAUSAL) {
            int32_t kv_base_sg = kv_start + sg_i * PF_KV_PER_SG;
            #pragma unroll
            for (int kv = 0; kv < 16; kv++) {
                int32_t kv_pos = kv_base_sg + kv;
                simd<int32_t, 16> v_kv_pos(kv_pos);
                ST_tile.select<16, 1>(0 * 256 + kv * 16).merge(FP32_MIN, v_kv_pos > causal_bound_0);
                ST_tile.select<16, 1>(1 * 256 + kv * 16).merge(FP32_MIN, v_kv_pos > causal_bound_1);
            }
        } else {
            // Noncausal: only need masking in the last iteration
            if (outerIter == kvOuterLoops - 1) {
                int32_t kv_base_sg = kv_start + sg_i * PF_KV_PER_SG;
                // Mask positions >= seq_len (partial last chunk)
                #pragma unroll
                for (int kv = 0; kv < 16; kv++) {
                    int32_t kv_pos = kv_base_sg + kv;
                    if (kv_pos >= seq_len) {
                        ST_tile.select<16, 1>(0 * 256 + kv * 16) = FP32_MIN;
                        ST_tile.select<16, 1>(1 * 256 + kv * 16) = FP32_MIN;
                    }
                }
                // Mask invalid Q rows (actual_q_rows < 128)
                if (actual_q_rows < (int)PF_WG_Q_ROWS) {
                    simd<int32_t, 16> q_idx_vec;
                    #pragma unroll
                    for (int qr = 0; qr < 16; qr++) q_idx_vec[qr] = qr;
                    simd<int32_t, 16> q_local_0 = sg_j * 32 + q_idx_vec;
                    simd<int32_t, 16> q_local_1 = sg_j * 32 + 16 + q_idx_vec;
                    #pragma unroll
                    for (int kv = 0; kv < 16; kv++) {
                        ST_tile.select<16, 1>(0 * 256 + kv * 16).merge(FP32_MIN, q_local_0 >= actual_q_rows);
                        ST_tile.select<16, 1>(1 * 256 + kv * 16).merge(FP32_MIN, q_local_1 >= actual_q_rows);
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
            int32_t next_logical = next_kv_start >> block_size_shift;
            int32_t next_off = next_kv_start & block_size_mask;
            int32_t next_phys = BLK_TABLE_LOAD(next_logical);
            uint32_t next_Y_base_K = (uint32_t)((next_phys << block_size_shift) + next_off + sg_i * PF_KV_PER_SG);
            uint32_t next_Y_base_V = (uint32_t)((next_phys << block_size_shift) + next_off);

            ST_next = 0;

            payloadK.set_y(next_Y_base_K);
            simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

            #pragma unroll
            for (int d = 0; d < (int)PF_HD_BLKS - 1; d++) {
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

                // V prefetch for outerIter+1
                // V prefetch: half rate (every other D-block) — reduces memory pressure
                if ((d & 1) == 0) {
                payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
                payloadVpf.set_y(next_Y_base_V + (d >> 1) * 16);
                __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>(payloadVpf);
                }

                payloadK.set_x(kv_x_k + (d + 1) * 16);
                K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
            }

            // Last D-block: no next-K load
            {
                constexpr int d = (int)PF_HD_BLKS - 1;
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

                // V prefetch last D-block
                // V prefetch: half rate (every other D-block) — reduces memory pressure
                if ((d & 1) == 0) {
                payloadVpf.set_x(kv_head_off_u32 / 2 + sg_i * 16 + (d & 1) * 8);
                payloadVpf.set_y(next_Y_base_V + (d >> 1) * 16);
                __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 8, 16, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>(payloadVpf);
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

        // V phase: per-iteration block lookup (no blk_table_cache)
        int32_t v_logical = kv_start >> block_size_shift;
        int32_t v_off = kv_start & block_size_mask;
        int32_t v_phys = BLK_TABLE_LOAD(v_logical);
        uint32_t v_Y_base = (uint32_t)((v_phys << block_size_shift) + v_off);

        // V load kv_blk=0
        payloadV.set_x(kv_head_off_u32 + sg_i * 32);
        payloadV.set_y(v_Y_base);
        simd<fp16, 256> V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
        payloadV.set_x(kv_head_off_u32 + sg_i * 32 + 16);
        simd<fp16, 256> V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

        // Compensation
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
        // VS PHASE + K PREFETCH (remaining tiles)
        // ========================================

// Helper macro: load S from SLM, load V (skip for kv_blk 0), run 8 DPAS
#define VS_LOAD_AND_DPAS(KV_BLK)                                                     \
        do {                                                                          \
            if ((KV_BLK) > 0) {                                                       \
                payloadV.set_x(kv_head_off_u32 + sg_i * 32);               \
                payloadV.set_y(v_Y_base + (KV_BLK) * 16);                             \
                V_vnni0 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,     \
                    __ESIMD_ENS::cache_hint::cached,                                  \
                    __ESIMD_ENS::cache_hint::cached>(payloadV);                       \
                payloadV.set_x(kv_head_off_u32 + sg_i * 32 + 16);          \
                V_vnni1 = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, true,     \
                    __ESIMD_ENS::cache_hint::cached,                                  \
                    __ESIMD_ENS::cache_hint::cached>(payloadV);                       \
            }                                                                         \
            uint32_t sb = PF_S_SLM_BASE + ((KV_BLK) * 16 + sg_j * PF_Q_GRPS) * 256; \
            simd<bf16, 128> sA, sB, sC, sD;                                          \
            sA.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb);           \
            sB.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 256);     \
            sC.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 512);     \
            sD.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 768);     \
            auto Vb0 = V_vnni0.template bit_cast_view<bf16>();                        \
            auto Vb1 = V_vnni1.template bit_cast_view<bf16>();                        \
            { auto acc = A_tile.select<128, 1>(0*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb0.data()), sA); } \
            { auto acc = A_tile.select<128, 1>(1*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb1.data()), sA); } \
            { auto acc = A_tile.select<128, 1>(2*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb0.data()), sB); } \
            { auto acc = A_tile.select<128, 1>(3*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb1.data()), sB); } \
            { auto acc = A_tile.select<128, 1>(4*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb0.data()), sC); } \
            { auto acc = A_tile.select<128, 1>(5*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb1.data()), sC); } \
            { auto acc = A_tile.select<128, 1>(6*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb0.data()), sD); } \
            { auto acc = A_tile.select<128, 1>(7*128); acc = dpas<8,8,float,float,bf16,bf16>(simd<float,128>(acc.data()), simd<bf16,256>(Vb1.data()), sD); } \
        } while(0)

#define K_PREFETCH_2(N)                                                                \
        do {                                                                           \
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + (2*(N)) * 8));                   \
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,            \
                __ESIMD_ENS::cache_hint::cached,                                       \
                __ESIMD_ENS::cache_hint::cached>(payloadKpf);                          \
            payloadKpf.set_x((uint32_t)(kv_x_k / 2 + (2*(N)+1) * 8));                 \
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,            \
                __ESIMD_ENS::cache_hint::cached,                                       \
                __ESIMD_ENS::cache_hint::cached>(payloadKpf);                          \
        } while(0)

        // K prefetch for outerIter+2: compute address
        {
            int32_t pf_kv_start = (outerIter + 2) * (int32_t)PF_KV_CHUNK;
            int32_t pf_logical = pf_kv_start >> block_size_shift;
            pf_logical = (pf_logical <= max_valid_blk_idx) ? pf_logical : max_valid_blk_idx;
            int32_t pf_off = pf_kv_start & block_size_mask;
            int32_t pf_phys = BLK_TABLE_LOAD(pf_logical);
            payloadKpf.set_y((uint32_t)((pf_phys << block_size_shift) + pf_off + sg_i * (int32_t)PF_KV_PER_SG));
        }

        VS_LOAD_AND_DPAS(0);  K_PREFETCH_2(0);
        VS_LOAD_AND_DPAS(1);  K_PREFETCH_2(1);
        VS_LOAD_AND_DPAS(2);  K_PREFETCH_2(2);
        VS_LOAD_AND_DPAS(3);  K_PREFETCH_2(3);
        VS_LOAD_AND_DPAS(4);  K_PREFETCH_2(4);
        VS_LOAD_AND_DPAS(5);  K_PREFETCH_2(5);
        VS_LOAD_AND_DPAS(6);  K_PREFETCH_2(6);
        VS_LOAD_AND_DPAS(7);  K_PREFETCH_2(7);

#undef K_PREFETCH_2
#undef VS_LOAD_AND_DPAS

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
#undef BLK_TABLE_LOAD
}
