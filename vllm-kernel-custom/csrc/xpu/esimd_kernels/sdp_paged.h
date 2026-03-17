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
#include <type_traits>

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

/* Load 64 fp16 values as f32. */
ESIMD_INLINE simd<float, 64> sdp_load_fp16_64(const unsigned short* ptr) {
    simd<unsigned short, 64> raw = block_load<unsigned short, 64>(ptr);
    simd<fp16, 64> as_fp16 = raw.template bit_cast_view<fp16>();
    return simd<float, 64>(as_fp16);
}

/* f32 -> fp16 store: 64 elements. */
ESIMD_INLINE void sdp_store_fp16_64(unsigned short* ptr, simd<float, 64> val) {
    simd<fp16, 64> narrow = val;
    block_store<unsigned short, 64>(ptr, narrow.template bit_cast_view<unsigned short>());
}

/* Dispatch load: bf16 or fp16. */
template<bool IS_BF16>
ESIMD_INLINE simd<float, 64> sdp_load_64(const unsigned short* ptr) {
    if constexpr (IS_BF16) return sdp_load_bf16_64(ptr);
    else return sdp_load_fp16_64(ptr);
}

/* Dispatch store: bf16 or fp16. */
template<bool IS_BF16>
ESIMD_INLINE void sdp_store_64(unsigned short* ptr, simd<float, 64> val) {
    if constexpr (IS_BF16) sdp_store_bf16_64(ptr, val);
    else sdp_store_fp16_64(ptr, val);
}

/* Half-precision type alias for template dispatch. */
template<bool IS_BF16>
using half_t = std::conditional_t<IS_BF16, bf16, fp16>;



/* ============================================================
 * SCALAR FALLBACK KERNEL — decode + prefill
 * One ESIMD thread per (request, head) or per (token, head).
 * Supports HD=128 and HD=256.
 * ============================================================ */
template<bool IS_BF16>
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

    simd<float, 64> q0 = sdp_load_64<IS_BF16>(q_row);
    simd<float, 64> q1 = sdp_load_64<IS_BF16>(q_row + 64);
    simd<float, 64> q2, q3;
    const bool hd256 = (head_dim == 256);
    if (hd256) {
        q2 = sdp_load_64<IS_BF16>(q_row + 128);
        q3 = sdp_load_64<IS_BF16>(q_row + 192);
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
        simd<float, 64> k0 = sdp_load_64<IS_BF16>(k_ptr);
        simd<float, 64> k1 = sdp_load_64<IS_BF16>(k_ptr + 64);

        float score;
        if (hd256) {
            simd<float, 64> k2 = sdp_load_64<IS_BF16>(k_ptr + 128);
            simd<float, 64> k3 = sdp_load_64<IS_BF16>(k_ptr + 192);
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
        acc0 += w * sdp_load_64<IS_BF16>(v_ptr);
        acc1 += w * sdp_load_64<IS_BF16>(v_ptr + 64);
        if (hd256) {
            acc2 += w * sdp_load_64<IS_BF16>(v_ptr + 128);
            acc3 += w * sdp_load_64<IS_BF16>(v_ptr + 192);
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

    sdp_store_64<IS_BF16>(out_row, acc0);
    sdp_store_64<IS_BF16>(out_row + 64, acc1);
    if (hd256) {
        sdp_store_64<IS_BF16>(out_row + 128, acc2);
        sdp_store_64<IS_BF16>(out_row + 192, acc3);
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
template<bool IS_BF16>
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

    simd<float, 64> q0 = sdp_load_64<IS_BF16>(q_row);
    simd<float, 64> q1 = sdp_load_64<IS_BF16>(q_row + 64);
    simd<float, 64> q2 = sdp_load_64<IS_BF16>(q_row + 128);
    simd<float, 64> q3 = sdp_load_64<IS_BF16>(q_row + 192);

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
        simd<float, 64> k0 = sdp_load_64<IS_BF16>(k_ptr);
        simd<float, 64> k1 = sdp_load_64<IS_BF16>(k_ptr + 64);
        simd<float, 64> k2 = sdp_load_64<IS_BF16>(k_ptr + 128);
        simd<float, 64> k3 = sdp_load_64<IS_BF16>(k_ptr + 192);

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
        acc0 += w * sdp_load_64<IS_BF16>(v_ptr);
        acc1 += w * sdp_load_64<IS_BF16>(v_ptr + 64);
        acc2 += w * sdp_load_64<IS_BF16>(v_ptr + 128);
        acc3 += w * sdp_load_64<IS_BF16>(v_ptr + 192);
    }

    // Store partials to scratch
    int scratch_idx = (req_idx * num_heads + head_idx) * num_chunks_per_seq + chunk_id;
    float* scratch_base = scratch_ptr + scratch_idx * (DEC_SCRATCH_PER_CHUNK / 4);

    // Store max and sum as scalar writes
    scratch_base[0] = max_score;
    scratch_base[1] = sum_exp;

    // Store output as bf16 (at offset 2 floats = 8 bytes)
    unsigned short* out_bf16 = reinterpret_cast<unsigned short*>(scratch_base + 2);
    sdp_store_64<IS_BF16>(out_bf16, acc0);
    sdp_store_64<IS_BF16>(out_bf16 + 64, acc1);
    sdp_store_64<IS_BF16>(out_bf16 + 128, acc2);
    sdp_store_64<IS_BF16>(out_bf16 + 192, acc3);
}


/* ============================================================
 * DECODE PHASE 2 — Cross-chunk reduction
 *
 * One ESIMD thread per (request, head).
 * Reads all chunk partials, does log-sum-exp correction, stores final output.
 * ============================================================ */
template<bool IS_BF16>
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
    simd<float, 64> acc0 = sdp_load_64<IS_BF16>(o0_bf16);
    simd<float, 64> acc1 = sdp_load_64<IS_BF16>(o0_bf16 + 64);
    simd<float, 64> acc2 = sdp_load_64<IS_BF16>(o0_bf16 + 128);
    simd<float, 64> acc3 = sdp_load_64<IS_BF16>(o0_bf16 + 192);

    // Merge remaining chunks
    for (int c = 1; c < actual_chunks; c++) {
        float* sc = scratch_ptr + (base_scratch_idx + c) * (DEC_SCRATCH_PER_CHUNK / 4);
        float chunk_max = sc[0];
        float chunk_sum = sc[1];

        const unsigned short* oc_bf16 = reinterpret_cast<const unsigned short*>(sc + 2);
        simd<float, 64> c0 = sdp_load_64<IS_BF16>(oc_bf16);
        simd<float, 64> c1 = sdp_load_64<IS_BF16>(oc_bf16 + 64);
        simd<float, 64> c2 = sdp_load_64<IS_BF16>(oc_bf16 + 128);
        simd<float, 64> c3 = sdp_load_64<IS_BF16>(oc_bf16 + 192);

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

    sdp_store_64<IS_BF16>(out_row, acc0);
    sdp_store_64<IS_BF16>(out_row + 64, acc1);
    sdp_store_64<IS_BF16>(out_row + 128, acc2);
    sdp_store_64<IS_BF16>(out_row + 192, acc3);
}


/* ============================================================
 * GQA-OPTIMIZED DECODE PHASE 1 — Single-thread, online softmax
 *
 * One thread per (request, kv_head, chunk).
 * Each thread processes DEC_GQA_CHUNK_SIZE KV tokens for 4 Q heads.
 * K/V loaded once, dot product computed for all 4 Q heads → 4× BW savings.
 * Online softmax: single pass loads K+V together.
 * ============================================================ */
template<bool IS_BF16>
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

    simd<float, 64> q0_a = sdp_load_64<IS_BF16>(q_base + (q_head_start + 0) * head_dim) * attn_scale;
    simd<float, 64> q0_b = sdp_load_64<IS_BF16>(q_base + (q_head_start + 0) * head_dim + 64) * attn_scale;
    simd<float, 64> q0_c = sdp_load_64<IS_BF16>(q_base + (q_head_start + 0) * head_dim + 128) * attn_scale;
    simd<float, 64> q0_d = sdp_load_64<IS_BF16>(q_base + (q_head_start + 0) * head_dim + 192) * attn_scale;

    simd<float, 64> q1_a = sdp_load_64<IS_BF16>(q_base + (q_head_start + 1) * head_dim) * attn_scale;
    simd<float, 64> q1_b = sdp_load_64<IS_BF16>(q_base + (q_head_start + 1) * head_dim + 64) * attn_scale;
    simd<float, 64> q1_c = sdp_load_64<IS_BF16>(q_base + (q_head_start + 1) * head_dim + 128) * attn_scale;
    simd<float, 64> q1_d = sdp_load_64<IS_BF16>(q_base + (q_head_start + 1) * head_dim + 192) * attn_scale;

    simd<float, 64> q2_a = sdp_load_64<IS_BF16>(q_base + (q_head_start + 2) * head_dim) * attn_scale;
    simd<float, 64> q2_b = sdp_load_64<IS_BF16>(q_base + (q_head_start + 2) * head_dim + 64) * attn_scale;
    simd<float, 64> q2_c = sdp_load_64<IS_BF16>(q_base + (q_head_start + 2) * head_dim + 128) * attn_scale;
    simd<float, 64> q2_d = sdp_load_64<IS_BF16>(q_base + (q_head_start + 2) * head_dim + 192) * attn_scale;

    simd<float, 64> q3_a = sdp_load_64<IS_BF16>(q_base + (q_head_start + 3) * head_dim) * attn_scale;
    simd<float, 64> q3_b = sdp_load_64<IS_BF16>(q_base + (q_head_start + 3) * head_dim + 64) * attn_scale;
    simd<float, 64> q3_c = sdp_load_64<IS_BF16>(q_base + (q_head_start + 3) * head_dim + 128) * attn_scale;
    simd<float, 64> q3_d = sdp_load_64<IS_BF16>(q_base + (q_head_start + 3) * head_dim + 192) * attn_scale;

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
        simd<float, 64> k_a = sdp_load_64<IS_BF16>(k_ptr);
        simd<float, 64> k_b = sdp_load_64<IS_BF16>(k_ptr + 64);
        simd<float, 64> k_c = sdp_load_64<IS_BF16>(k_ptr + 128);
        simd<float, 64> k_d = sdp_load_64<IS_BF16>(k_ptr + 192);

        // QK dot products (scale already in Q)
        float s0 = sdp_dot256(q0_a, q0_b, q0_c, q0_d, k_a, k_b, k_c, k_d);
        float s1 = sdp_dot256(q1_a, q1_b, q1_c, q1_d, k_a, k_b, k_c, k_d);
        float s2 = sdp_dot256(q2_a, q2_b, q2_c, q2_d, k_a, k_b, k_c, k_d);
        float s3 = sdp_dot256(q3_a, q3_b, q3_c, q3_d, k_a, k_b, k_c, k_d);

        // Load V
        const unsigned short* v_ptr = kv_cache_ptr + kv_base_off + kv_stride_split;
        simd<float, 64> v_a = sdp_load_64<IS_BF16>(v_ptr);
        simd<float, 64> v_b = sdp_load_64<IS_BF16>(v_ptr + 64);
        simd<float, 64> v_c = sdp_load_64<IS_BF16>(v_ptr + 128);
        simd<float, 64> v_d = sdp_load_64<IS_BF16>(v_ptr + 192);

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
        if (h == 0) { sdp_store_64<IS_BF16>(out_bf16, a0_a); sdp_store_64<IS_BF16>(out_bf16 + 64, a0_b);
                      sdp_store_64<IS_BF16>(out_bf16 + 128, a0_c); sdp_store_64<IS_BF16>(out_bf16 + 192, a0_d); }
        if (h == 1) { sdp_store_64<IS_BF16>(out_bf16, a1_a); sdp_store_64<IS_BF16>(out_bf16 + 64, a1_b);
                      sdp_store_64<IS_BF16>(out_bf16 + 128, a1_c); sdp_store_64<IS_BF16>(out_bf16 + 192, a1_d); }
        if (h == 2) { sdp_store_64<IS_BF16>(out_bf16, a2_a); sdp_store_64<IS_BF16>(out_bf16 + 64, a2_b);
                      sdp_store_64<IS_BF16>(out_bf16 + 128, a2_c); sdp_store_64<IS_BF16>(out_bf16 + 192, a2_d); }
        if (h == 3) { sdp_store_64<IS_BF16>(out_bf16, a3_a); sdp_store_64<IS_BF16>(out_bf16 + 64, a3_b);
                      sdp_store_64<IS_BF16>(out_bf16 + 128, a3_c); sdp_store_64<IS_BF16>(out_bf16 + 192, a3_d); }
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


template<bool CAUSAL, bool IS_BF16>
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
    const unsigned short* kv_v_base = kv_cache_ptr + kv_stride_split;

    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    // block_size is always power of 2 — use shift for division/modulo
    int32_t block_size_shift = __builtin_ctz(block_size);
    int32_t block_size_mask = block_size - 1;
    int32_t max_valid_blk_idx = (seq_len - 1) >> block_size_shift;

    // Physical rows per block in the 2D surface.
    // For contiguous [2, num_blocks, bs, nkvh, hd]: phys_rows_per_block = bs
    // For interleaved [num_blocks, 2, bs, nkvh, hd] presented as [2, ...]:
    //   phys_rows_per_block = stride(1)/stride(2) = 2*bs (K+V interleaved per block)
    int32_t phys_rows_per_block = (int32_t)(kv_stride_block / kv_stride_pos);
    int32_t phys_block_shift = __builtin_ctz(phys_rows_per_block);

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
    // Safe upper bound for surface height — covers any practical allocation
    uint32_t kv_surf_h = 0x3FFFFFU;
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
        payloadKpf.set_y((uint32_t)((pf0_phys << phys_block_shift) + sg_i * (int32_t)PF_KV_PER_SG));
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
        payloadKpf.set_y((uint32_t)((pf1_phys << phys_block_shift) + pf1_off + sg_i * (int32_t)PF_KV_PER_SG));
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
        uint32_t Y_base_K = (uint32_t)((phys0 << phys_block_shift) + sg_i * PF_KV_PER_SG);
        uint32_t Y_base_V = (uint32_t)(phys0 << phys_block_shift);

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

            simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
            simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

            uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
            uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

            simd<uint32_t, 128> Q_raw0, Q_raw1;
            Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
            Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
            Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
            Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
            simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
            simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

            { auto acc = ST_tile.select<128, 1>(0);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(128);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
            { auto acc = ST_tile.select<128, 1>(256);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(384);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }

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

            simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
            simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

            uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
            uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

            simd<uint32_t, 128> Q_raw0, Q_raw1;
            Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
            Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
            Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
            Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
            simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
            simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

            { auto acc = ST_tile.select<128, 1>(0);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(128);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
            { auto acc = ST_tile.select<128, 1>(256);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
            { auto acc = ST_tile.select<128, 1>(384);
              acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }
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
            uint32_t next_Y_base_K = (uint32_t)((next_phys << phys_block_shift) + next_off + sg_i * PF_KV_PER_SG);
            uint32_t next_Y_base_V = (uint32_t)((next_phys << phys_block_shift) + next_off);

            ST_next = 0;

            payloadK.set_y(next_Y_base_K);
            simd<fp16, 256> K_both = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);

            #pragma unroll
            for (int d = 0; d < (int)PF_HD_BLKS - 1; d++) {
                simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
                simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

                uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
                uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

                simd<uint32_t, 128> Q_raw0, Q_raw1;
                Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
                Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
                Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
                Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
                simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
                simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

                { auto acc = ST_next.select<128, 1>(0);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(128);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
                { auto acc = ST_next.select<128, 1>(256);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(384);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }

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
                simd<half_t<IS_BF16>, 128> K_sb0(K_both.select<128, 1>(0).template bit_cast_view<half_t<IS_BF16>>().data());
                simd<half_t<IS_BF16>, 128> K_sb1(K_both.select<128, 1>(128).template bit_cast_view<half_t<IS_BF16>>().data());

                uint32_t q_slm_off0 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 0) * 512;
                uint32_t q_slm_off1 = PF_Q_SLM_BASE + (d * PF_Q_TILES + sg_j * PF_Q_PAIRS + 1) * 512;

                simd<uint32_t, 128> Q_raw0, Q_raw1;
                Q_raw0.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off0);
                Q_raw0.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off0 + 256);
                Q_raw1.select<64, 1>(0) = slm_block_load<uint32_t, 64>(q_slm_off1);
                Q_raw1.select<64, 1>(64) = slm_block_load<uint32_t, 64>(q_slm_off1 + 256);
                simd<half_t<IS_BF16>, 256> Q_vnni0 = Q_raw0.template bit_cast_view<half_t<IS_BF16>>();
                simd<half_t<IS_BF16>, 256> Q_vnni1 = Q_raw1.template bit_cast_view<half_t<IS_BF16>>();

                { auto acc = ST_next.select<128, 1>(0);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(128);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni0.data()), K_sb1); }
                { auto acc = ST_next.select<128, 1>(256);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb0); }
                { auto acc = ST_next.select<128, 1>(384);
                  acc = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(simd<float, 128>(acc.data()), simd<half_t<IS_BF16>, 256>(Q_vnni1.data()), K_sb1); }

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
        simd<unsigned short, 256> ST_bf16_0;
        simd<unsigned short, 256> ST_bf16_1;

        {
            simd<float, 16> gm = global_max.select<16, 1>(0);
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_0.select<16, 1>(kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(0) += s;
            }
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(128 + kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_0.select<16, 1>(128 + kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(0) += s;
            }
        }
        {
            simd<float, 16> gm = global_max.select<16, 1>(16);
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_1.select<16, 1>(kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(16) += s;
            }
            #pragma unroll
            for (int kv = 0; kv < 8; kv++) {
                simd<float, 16> s = __ESIMD_NS::exp2<float, 16, float>(
                    ST_tile.select<16, 1>(256 + 128 + kv * 16) - gm);
                { simd<half_t<IS_BF16>, 16> _h(s); ST_bf16_1.select<16, 1>(128 + kv * 16) = _h.template bit_cast_view<unsigned short>(); }
                local_sum.select<16, 1>(16) += s;
            }
        }

        // ========================================
        // S^T SCATTER TO SLM
        // ========================================
        #pragma unroll
        for (int qp = 0; qp < (int)PF_Q_PAIRS; qp++) {
            auto& ST_bf16 = (qp == 0) ? ST_bf16_0 : ST_bf16_1;
            simd<uint16_t, 256> ST_fp16_u16(ST_bf16);

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
        uint32_t v_Y_base = (uint32_t)((v_phys << phys_block_shift) + v_off);

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
            simd<half_t<IS_BF16>, 128> sA, sB, sC, sD;                                          \
            sA.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb);           \
            sB.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 256);     \
            sC.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 512);     \
            sD.template bit_cast_view<uint32_t>() = slm_block_load<uint32_t, 64>(sb + 768);     \
            auto Vb0 = V_vnni0.template bit_cast_view<half_t<IS_BF16>>();                        \
            auto Vb1 = V_vnni1.template bit_cast_view<half_t<IS_BF16>>();                        \
            { auto acc = A_tile.select<128, 1>(0*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sA); } \
            { auto acc = A_tile.select<128, 1>(1*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sA); } \
            { auto acc = A_tile.select<128, 1>(2*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sB); } \
            { auto acc = A_tile.select<128, 1>(3*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sB); } \
            { auto acc = A_tile.select<128, 1>(4*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sC); } \
            { auto acc = A_tile.select<128, 1>(5*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sC); } \
            { auto acc = A_tile.select<128, 1>(6*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb0.data()), sD); } \
            { auto acc = A_tile.select<128, 1>(7*128); acc = dpas<8,8,float,float,half_t<IS_BF16>,half_t<IS_BF16>>(simd<float,128>(acc.data()), simd<half_t<IS_BF16>, 256>(Vb1.data()), sD); } \
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
            payloadKpf.set_y((uint32_t)((pf_phys << phys_block_shift) + pf_off + sg_i * (int32_t)PF_KV_PER_SG));
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
                simd<half_t<IS_BF16>, 16> bf16_out = f32_out;
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


/* ============================================================
 * DPAS-BASED PREFILL KERNEL — HD=128, bf16io, 16-thread WG
 *
 * Adapted from flash.attn.b.mha128.bf16io (non-paged) and
 * flash.attn.b.mha128.gqa.precomputed (causal reference).
 *
 * Architecture:
 *   16-thread WG, each thread owns 16 Q rows → 256 Q rows per WG.
 *   KV chunk = 64 tokens per outer iteration.
 *   No D-loop: HD=128 = 8 × 16 D-blocks, all inlined.
 *
 * bf16io strategy:
 *   QK: dpas<8,8,float,float,bf16,bf16> — bf16 Q/K, fp32 accum
 *   VS: dpas<8,8,fp16,fp16,fp16,fp16> — fp16 softmax × fp16 V, fp16 accum
 *   V conversion: bf16→fp16 interleaved with VS DPAS
 *   Compensation: fp16 multiply (native on Xe2)
 *
 * SLM: 32KB for V ping-pong (2 × 64 × 128 × sizeof(fp16))
 *   Each of 16 threads scatters its 32-element V tile to SLM.
 *   Standard barrier() between V scatter and VS load.
 *
 * Paged K/V: per-iteration block_table lookup, stride-aware.
 *   Uses phys_block_shift for interleaved KV layout.
 * ============================================================ */

template<bool CAUSAL, bool IS_BF16>
ESIMD_INLINE void sdp_paged_prefill_dpas_128(
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

    constexpr uint32_t HD = 128;
    constexpr uint32_t WG_Q_ROWS = 256;    // 16 threads × 16 Q rows
    constexpr uint32_t KV_CHUNK = 64;
    constexpr uint32_t slmSizeV = 2 * 64 * 128 * sizeof(fp16);  // 32KB V ping-pong
    constexpr uint32_t slmOffsetBaseV = 0;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    __ESIMD_NS::slm_init(slmSizeV);

    int32_t localLinearId = ndi.get_local_id(0);
    int32_t hhq = localLinearId & 0xf;   // 0..15, Q row group index
    // V cooperative load indices: each thread loads 4×16 = 64 bf16 → covers 64×128 tile
    int32_t hhv = localLinearId & 0x3;   // V column quarter (0..3)
    int32_t vvv = localLinearId >> 2;    // V row group (0..3)

    int32_t wg_id = ndi.get_group(0);

    int32_t head_idx = wg_id % num_heads;
    int32_t temp = wg_id / num_heads;
    int32_t req_idx = temp / max_q_tiles_per_req;
    int32_t q_tile_idx = temp % max_q_tiles_per_req;

    if (req_idx >= batch) return;

    int32_t req_q_start = query_start_loc_ptr[req_idx];
    int32_t req_q_end = query_start_loc_ptr[req_idx + 1];
    int32_t req_query_len = req_q_end - req_q_start;
    int32_t q_offset = q_tile_idx * (int32_t)WG_Q_ROWS;

    if (q_offset >= req_query_len) return;

    int32_t seq_len = seq_lens_ptr[req_idx];
    if (seq_len <= 0) return;

    int32_t group_size = num_heads / num_kv_heads;
    int32_t kv_head_idx = head_idx / group_size;

    // Paged KV cache access
    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    int32_t block_size_shift = __builtin_ctz(block_size);
    int32_t block_size_mask = block_size - 1;
    int32_t phys_rows_per_block = (int32_t)(kv_stride_block / kv_stride_pos);
    int32_t phys_block_shift = __builtin_ctz(phys_rows_per_block);

    // KV surface parameters
    uint32_t kv_head_off_u32 = (uint32_t)((int64_t)kv_head_idx * kv_stride_head);
    uint32_t kv_row_bytes = (uint32_t)(kv_stride_pos * 2);  // bf16 = 2 bytes
    uint32_t kv_surf_w = kv_row_bytes - 1;
    uint32_t kv_surf_h = 0x3FFFFFU;  // safe upper bound

    // Q positions
    int32_t q_global_start = req_q_start + q_offset;
    int32_t history_len = seq_len - req_query_len;
    int32_t q_abs_base = history_len + q_offset;
    int32_t actual_q_rows = req_query_len - q_offset;
    if (actual_q_rows > (int32_t)WG_Q_ROWS) actual_q_rows = (int32_t)WG_Q_ROWS;

    // Compute Q row range for this thread
    int32_t my_q_start = hhq * 16;

    // ============================================================
    // REGISTER DECLARATIONS
    // ============================================================
    // Q stored as bf16 (no conversion — bf16 QK DPAS)
    simd<half_t<IS_BF16>, 16 * 128> bf16QState;
    simd<float, 16 * 32> tempBuffer;
    simd<float, 16 * 64> tempOutput;     // QK scores: 16 Q × 64 KV
    auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
    auto tempBufferAsHalf = tempBuffer.template bit_cast_view<half_t<IS_BF16>>();
    auto ui32Temp = tempBuffer.template bit_cast_view<uint32_t>();

    // SxV accumulator: fp16 (native fp16 compensation)
    simd<fp16, 16 * 128> finalOutput = 0;
    simd<float, 16> fp32SoftMaxTemp = 0;
    simd<float, 16> fp32HistoricMaxTemp = FP32_MIN;
    simd<uint32_t, 16> baseOffsetInc16AsVector(baseOffsetInc16);

    // ============================================================
    // CAUSAL BOUNDARIES (pre-compute)
    // ============================================================
    simd<int32_t, 16> causal_boundaries;
    int32_t kvSeqOutLoopCount;

    if constexpr (CAUSAL) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int32_t q_pos = q_offset + my_q_start + i;
            if (q_pos < req_query_len) {
                causal_boundaries[i] = history_len + q_pos;
            } else {
                causal_boundaries[i] = -1;
            }
        }

        // Early termination: max KV position this WG can attend to
        int32_t max_q_in_wg = q_offset + actual_q_rows - 1;
        int32_t max_kv_pos = history_len + max_q_in_wg;
        kvSeqOutLoopCount = (max_kv_pos + (int32_t)KV_CHUNK) / (int32_t)KV_CHUNK;
        int32_t total_kv_blocks = (seq_len + (int32_t)KV_CHUNK - 1) / (int32_t)KV_CHUNK;
        if (kvSeqOutLoopCount > total_kv_blocks) kvSeqOutLoopCount = total_kv_blocks;
    } else {
        kvSeqOutLoopCount = (seq_len + (int32_t)KV_CHUNK - 1) / (int32_t)KV_CHUNK;
    }
    if (kvSeqOutLoopCount <= 0) kvSeqOutLoopCount = 1;

    // ============================================================
    // Q LOAD — bf16, 2D surface, VNNI transpose
    // ============================================================
    {
        uint32_t widthInByteQ = num_heads * HD * sizeof(bf16) - 1;
        uint32_t heightQ = num_tokens - 1;
        uint32_t qCoordX = head_idx * (HD >> 1);  // in uint32 units (VNNI pair)
        uint32_t qCoordY = q_global_start + my_q_start;

        __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
            (uint32_t*)query_ptr, widthInByteQ, heightQ, widthInByteQ, qCoordX, qCoordY);

        #pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
            bf16QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * kk) =
                __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);
            qCoordX += 8;
            payloadQ.set_x(qCoordX);
        }
    }

    // ============================================================
    // K/V 2D SURFACE DESCRIPTORS (fixed base, per-iteration set_y)
    // ============================================================
    uint32_t kv_x_k = kv_head_off_u32;  // K X coordinate

    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
        (fp16*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, 0);

    // V: each thread loads a 16×16 tile with transpose+VNNI
    const unsigned short* kv_v_base = kv_cache_ptr + kv_stride_split;
    uint32_t vCoordX = kv_head_off_u32 + hhv * 32;

    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadV(
        (fp16*)kv_v_base, kv_surf_w, kv_surf_h, kv_surf_w, vCoordX, 0);

    // K prefetch descriptor
    uint32_t prefCoordX = (kv_head_off_u32 >> 1) + (localLinearId & 0x1) * 32;
    __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefK(
        (uint32_t*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, prefCoordX, 0);

    // SLM offset for this thread's V scatter
    unsigned int slmOffsetV = slmOffsetBaseV + localLinearId * 512 * sizeof(fp16);

    // ============================================================
    // PROLOGUE: Prefetch K[0], load V[0], convert V[0], scatter V[0]
    // ============================================================
    {
        int32_t phys0 = block_table_row[0];
        uint32_t Y_base_K_pref = (uint32_t)(phys0 << phys_block_shift);

        // Prefetch K[0]
        payloadPrefK.set_y(Y_base_K_pref + (localLinearId >> 1) * 8);
        #pragma unroll
        for (int32_t kk = 0; kk < 2; kk++) {
            payloadPrefK.set_x(prefCoordX + 16 * kk);
            __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
        }

        // Load V[0] (bf16 bits loaded as fp16)
        uint32_t v0_Y = (uint32_t)(phys0 << phys_block_shift) + vvv * 16;
        payloadV.set_y(v0_Y);
        tempBufferAsFp16.select<512, 1>(0) =
            __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

                // Convert V[0] bf16->fp16 (only needed when data is bf16)
        if constexpr (IS_BF16) {
            #pragma unroll
            for (int ci = 0; ci < 32; ci++) {
                simd<float, 16> cvt = tempBufferAsHalf.template select<16, 1>(16 * ci);
                tempBufferAsFp16.select<16, 1>(16 * ci) = cvt;
            }
        }

        // Scatter V[0] to SLM
        simd<uint32_t, 32> simdSlmOffsetsV;
        simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
        simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
        simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV;
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
                simdSlmOffsetsV.select<16, 1>(16 * kk),
                tempBufferAsFp16.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
        }
    }

    int loopIdx;

    // ============================================================
    // MAIN LOOP: outerIter = 0 .. kvSeqOutLoopCount - 2
    // ============================================================
    for (loopIdx = 0; loopIdx < kvSeqOutLoopCount - 1; loopIdx++) {
        uint32_t slmPingpongLoad = loopIdx & 0x1;
        uint32_t slmPingpongStore = (loopIdx + 1) & 0x1;
        slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
        slmPingpongStore = slmPingpongStore * 64 * 128 * sizeof(fp16);
        auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
        auto tempQkAsHalf = tempOutput.template bit_cast_view<half_t<IS_BF16>>();
        simd<fp16, 512> fp16VState;
        tempOutput = 0;

        // Paged K access: compute physical block for this KV chunk
        int32_t kv_start = loopIdx * (int32_t)KV_CHUNK;
        int32_t logical_blk = kv_start >> block_size_shift;
        int32_t blk_offset = kv_start & block_size_mask;
        int32_t phys_blk = block_table_row[logical_blk];
        uint32_t Y_base_K = (uint32_t)((phys_blk << phys_block_shift) + blk_offset);

        // ===== Q @ K^T (bf16 DPAS, 8 D-blocks, no D-loop) =====
        {
            #pragma unroll
            for (int32_t nn = 0; nn < 8; nn++) {
                payloadK.set_x(kv_x_k + 16 * nn);
                #pragma unroll
                for (int32_t l = 0; l < 4; l++) {
                    payloadK.set_y(Y_base_K + 16 * l);
                    tempBufferAsFp16.select<256, 1>(256 * l) =
                        __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
                }
                #pragma unroll
                for (int32_t kk = 0; kk < 8; kk++) {
                    auto ccTile = tempOutput.select<128, 1>(128 * kk);
                    auto aaTile = bf16QState.template select<256, 1>(256 * nn);
                    auto bbTile = tempBufferAsHalf.template select<128, 1>(128 * kk);
                    ccTile = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(
                        simd<float, 128>(ccTile.data()),
                        simd<half_t<IS_BF16>, 256>(aaTile.data()),
                        simd<half_t<IS_BF16>, 128>(bbTile.data()));
                }
            }
        }

        // ===== CAUSAL MASK =====
        if constexpr (CAUSAL) {
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                #pragma unroll
                for (int m = 0; m < 8; m++) {
                    int32_t kv_pos = kv_start + kk * 8 + m;
                    simd<int32_t, 16> v_kv_pos(kv_pos);
                    tempOutput.select<16, 1>(kk * 128 + m * 16).merge(FP32_MIN, v_kv_pos > causal_boundaries);
                }
            }
        }

        // ===== V LOAD (bf16 bits) for NEXT iteration's SLM scatter =====
        {
            int32_t next_kv_start = (loopIdx + 1) * (int32_t)KV_CHUNK;
            int32_t next_logical = next_kv_start >> block_size_shift;
            int32_t next_blk_off = next_kv_start & block_size_mask;
            int32_t next_phys = block_table_row[next_logical];
            uint32_t next_v_Y = (uint32_t)((next_phys << phys_block_shift) + next_blk_off) + vvv * 16;
            payloadV.set_y(next_v_Y);
        }
        fp16VState =
            __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

        // ===== SOFTMAX =====
        {
            auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
            auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
            auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
            simd<float, 8 * 16> ttemp;
            fp32CurrentMaxTemp = fp32HistoricMaxTemp;

            // Row-wise max
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                ttemp.select<32, 1>(32 * kk) = __ESIMD_NS::max<float, 32, float>(
                    tempOutput.select<32, 1>(64 * kk),
                    tempOutput.select<32, 1>(64 * kk + 32));
            }
            #pragma unroll
            for (int kkk = 0; kkk < 6; ++kkk) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    ttemp.select<32, 1>(32 * kk) =
                        __ESIMD_NS::max<float, 32, float>(
                            ttemp.select<32, 1>(32 * kk),
                            tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16));
                }
            }
            ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(64));
            ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(32));
            ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
            fp32CurrentMaxTemp.merge(
                ttemp.select<16, 1>(0),
                ttemp.select<16, 1>(0) > fp32CurrentMaxTemp);

            fp32Exp2Temp.select<16, 1>(0) = fp32CurrentMaxTemp.select<16, 1>(0) * attnScoreMul;

            // exp2(score * attnScoreMul - max * attnScoreMul)
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(128 * k + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                    ttemp.select<16, 1>(16 * kk + 32) = tempOutput.select<16, 1>(128 * k + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                }
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    ttemp.select<16, 1>(16 * kk + 64) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                    ttemp.select<16, 1>(16 * kk + 64 + 32) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                }
                #pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    tempOutput.select<16, 1>(128 * k + 16 * kk) = __ESIMD_NS::exp2<float, 16, float>(ttemp.select<16, 1>(16 * kk));
                }
            }

            // Compensation factor
            fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
            fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);

            // Row-wise sum
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                ttemp.select<32, 1>(32 * kk) = tempOutput.select<32, 1>(64 * kk) + tempOutput.select<32, 1>(64 * kk + 32);
            }
            #pragma unroll
            for (int kkk = 0; kkk < 6; ++kkk) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    ttemp.select<32, 1>(32 * kk) = ttemp.select<32, 1>(32 * kk) + tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16);
                }
            }
            ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(64);
            ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(32);
            ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);
            fp32HistoricMaxTemp = fp32CurrentMaxTemp;

            // Compensation — fp16 multiply (native on Xe2)
            simd<fp16, 32> compensationTemp;
            compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
            compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;
            #pragma unroll
            for (int kk = 0; kk < 64; kk++) {
                finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
            }

            // Pack softmax weights fp32 → fp16 VNNI
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 64 * kk);
                }
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 64 * kk + 32);
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempQkAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk);
                }
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempQkAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk + 32);
                }
            }
        }

        barrier();

        // ===== S×V — fp16 DPAS with V bf16→fp16 conversion interleaved =====
        {
            auto vAsBf16 = fp16VState.template bit_cast_view<bf16>();
            // First half: 32 DPAS + 32 V conversion chunks interleaved
            #pragma unroll
            for (int nn = 0; nn < 2; nn++) {
                #pragma unroll
                for (int l = 0; l < 2; l++) {
                    #pragma unroll
                    for (int ll = 0; ll < 2; ll++) {
                        tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
                            slm_block_load<fp16, 512>(slmOffsetBaseV +
                                slmPingpongLoad +
                                16 * 128 * nn * sizeof(fp16) +
                                16 * 64 * l * sizeof(fp16) +
                                512 * ll * sizeof(fp16));
                    }
                    #pragma unroll
                    for (int ll = 0; ll < 8; ll++) {
                        auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
                        auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
                        auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
                        ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                            simd<fp16, 128>(ccTile.data()),
                            simd<fp16, 256>(aaTile.data()),
                            simd<fp16, 128>(bbTile.data()));
                        // V conversion: XVE works while XMX processes DPAS
                        if constexpr (IS_BF16) {
                            int32_t ci = nn * 16 + l * 8 + ll;
                            simd<float, 16> cvt = vAsBf16.select<16, 1>(16 * ci);
                            fp16VState.select<16, 1>(16 * ci) = cvt;
                        }
                    }
                }
            }
            // Second half: 32 DPAS, V conversion already done
            #pragma unroll
            for (int nn = 0; nn < 2; nn++) {
                #pragma unroll
                for (int l = 0; l < 2; l++) {
                    #pragma unroll
                    for (int ll = 0; ll < 2; ll++) {
                        tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
                            slm_block_load<fp16, 512>(slmOffsetBaseV +
                                slmPingpongLoad +
                                16 * 128 * 2 * sizeof(fp16) +
                                16 * 128 * nn * sizeof(fp16) +
                                16 * 64 * l * sizeof(fp16) +
                                512 * ll * sizeof(fp16));
                    }
                    #pragma unroll
                    for (int ll = 0; ll < 8; ll++) {
                        auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
                        auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
                        auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
                        ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                            simd<fp16, 128>(ccTile.data()),
                            simd<fp16, 256>(aaTile.data()),
                            simd<fp16, 128>(bbTile.data()));
                    }
                }
            }

            // SLM scatter next V (fp16VState now converted to fp16)
            simd<uint32_t, 32> simdSlmOffsetsV;
            simdSlmOffsetsV.select<16, 1>(0) = baseOffsetInc16AsVector;
            simdSlmOffsetsV.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
            simdSlmOffsetsV.select<32, 1>(0) = simdSlmOffsetsV.select<32, 1>(0) * 16 * sizeof(fp16) + slmOffsetV + slmPingpongStore;
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
                    simdSlmOffsetsV.select<16, 1>(16 * kk),
                    fp16VState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk));
            }
        }

        // K prefetch for next+1 iteration
        {
            int32_t pf_kv_start = (loopIdx + 2) * (int32_t)KV_CHUNK;
            int32_t pf_logical = pf_kv_start >> block_size_shift;
            int32_t max_valid = (seq_len - 1) >> block_size_shift;
            if (pf_logical > max_valid) pf_logical = max_valid;
            int32_t pf_off = pf_kv_start & block_size_mask;
            int32_t pf_phys = block_table_row[pf_logical];
            payloadPrefK.set_y((uint32_t)((pf_phys << phys_block_shift) + pf_off + (localLinearId >> 1) * 8));
            #pragma unroll
            for (int32_t kk = 0; kk < 2; kk++) {
                payloadPrefK.set_x(prefCoordX + 16 * kk);
                __ESIMD_ENS::lsc_prefetch_2d<uint32_t, 16, 8, 1, false, false,
                    __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadPrefK);
            }
        }
    }

    // ============================================================
    // LAST LOOP — with boundary + causal masks
    // ============================================================
    {
        uint32_t slmPingpongLoad = (loopIdx) & 0x1;
        slmPingpongLoad = slmPingpongLoad * 64 * 128 * sizeof(fp16);
        auto tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
        auto tempQkAsHalf = tempOutput.template bit_cast_view<half_t<IS_BF16>>();
        tempOutput = 0;

        int32_t kv_start = loopIdx * (int32_t)KV_CHUNK;
        int32_t logical_blk = kv_start >> block_size_shift;
        int32_t blk_offset = kv_start & block_size_mask;
        int32_t phys_blk = block_table_row[logical_blk];
        uint32_t Y_base_K = (uint32_t)((phys_blk << phys_block_shift) + blk_offset);

        // Q @ K^T (bf16 DPAS)
        {
            #pragma unroll
            for (int32_t nn = 0; nn < 8; nn++) {
                payloadK.set_x(kv_x_k + 16 * nn);
                #pragma unroll
                for (int32_t l = 0; l < 4; l++) {
                    payloadK.set_y(Y_base_K + 16 * l);
                    tempBufferAsFp16.select<256, 1>(256 * l) =
                        __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
                }
                #pragma unroll
                for (int32_t kk = 0; kk < 8; kk++) {
                    auto ccTile = tempOutput.select<128, 1>(128 * kk);
                    auto aaTile = bf16QState.template select<256, 1>(256 * nn);
                    auto bbTile = tempBufferAsHalf.template select<128, 1>(128 * kk);
                    ccTile = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(
                        simd<float, 128>(ccTile.data()),
                        simd<half_t<IS_BF16>, 256>(aaTile.data()),
                        simd<half_t<IS_BF16>, 128>(bbTile.data()));
                }
            }
        }

        // Apply boundary mask + causal mask, then softmax
        {
            auto fp32CurrentMaxTemp = tempBuffer.select<16, 1>(0);
            auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
            auto fp32Exp2Temp = tempBuffer.select<16, 1>(32);
            auto softmaxPositions = ui32Temp.select<16, 1>(48);
            simd<float, 8 * 16> ttemp;

            // Boundary mask (kv_pos >= seq_len)
            softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + loopIdx * (int32_t)KV_CHUNK;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN, softmaxPositions.select<16, 0>(kk) >= (uint32_t)seq_len);
                }
                softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
            }

            // Causal mask
            if constexpr (CAUSAL) {
                #pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    #pragma unroll
                    for (int m = 0; m < 8; m++) {
                        int32_t kv_pos = kv_start + kk * 8 + m;
                        simd<int32_t, 16> v_kv_pos(kv_pos);
                        tempOutput.select<16, 1>(kk * 128 + m * 16).merge(FP32_MIN, v_kv_pos > causal_boundaries);
                    }
                }
            }

            // Invalid Q row mask
            if (actual_q_rows < (int32_t)WG_Q_ROWS) {
                if (my_q_start >= actual_q_rows) {
                    tempOutput = FP32_MIN;
                } else if (my_q_start + 16 > actual_q_rows) {
                    int valid = actual_q_rows - my_q_start;
                    #pragma unroll
                    for (int kk = 0; kk < 64; kk++) {
                        #pragma unroll
                        for (int qi = 0; qi < 16; qi++) {
                            if (qi >= valid) tempOutput[kk * 16 + qi] = FP32_MIN;
                        }
                    }
                }
            }

            fp32CurrentMaxTemp = fp32HistoricMaxTemp;
            // Row-wise max
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                ttemp.select<32, 1>(32 * kk) = __ESIMD_NS::max<float, 32, float>(
                    tempOutput.select<32, 1>(64 * kk),
                    tempOutput.select<32, 1>(64 * kk + 32));
            }
            #pragma unroll
            for (int kkk = 0; kkk < 6; ++kkk) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    ttemp.select<32, 1>(32 * kk) =
                        __ESIMD_NS::max<float, 32, float>(
                            ttemp.select<32, 1>(32 * kk),
                            tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16));
                }
            }
            ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(ttemp.select<64, 1>(0), ttemp.select<64, 1>(64));
            ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(ttemp.select<32, 1>(0), ttemp.select<32, 1>(32));
            ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(ttemp.select<16, 1>(0), ttemp.select<16, 1>(16));
            fp32CurrentMaxTemp.merge(
                ttemp.select<16, 1>(0),
                ttemp.select<16, 1>(0) > fp32CurrentMaxTemp);

            fp32Exp2Temp.select<16, 1>(0) = fp32CurrentMaxTemp.select<16, 1>(0) * attnScoreMul;

            #pragma unroll
            for (int k = 0; k < 8; k++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    ttemp.select<16, 1>(16 * kk) = tempOutput.select<16, 1>(128 * k + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                    ttemp.select<16, 1>(16 * kk + 32) = tempOutput.select<16, 1>(128 * k + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                }
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    ttemp.select<16, 1>(16 * kk + 64) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                    ttemp.select<16, 1>(16 * kk + 64 + 32) = tempOutput.select<16, 1>(128 * k + 64 + 32 * kk + 16) * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
                }
                #pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    tempOutput.select<16, 1>(128 * k + 16 * kk) = __ESIMD_NS::exp2<float, 16, float>(ttemp.select<16, 1>(16 * kk));
                }
            }

            fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<16, 1>(0);
            fp32SoftMaxCompensation = __ESIMD_NS::exp2<float, 16, float>(fp32SoftMaxCompensation);

            if (loopIdx != 0) {
                fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) * fp32SoftMaxCompensation.select<16, 1>(0);
            }

            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                ttemp.select<32, 1>(32 * kk) = tempOutput.select<32, 1>(64 * kk) + tempOutput.select<32, 1>(64 * kk + 32);
            }
            #pragma unroll
            for (int kkk = 0; kkk < 6; ++kkk) {
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    ttemp.select<32, 1>(32 * kk) = ttemp.select<32, 1>(32 * kk) + tempOutput.select<32, 1>((4 * kkk + kk) * 32 + 16 * 16);
                }
            }
            ttemp.select<64, 1>(0) = ttemp.select<64, 1>(0) + ttemp.select<64, 1>(64);
            ttemp.select<32, 1>(0) = ttemp.select<32, 1>(0) + ttemp.select<32, 1>(32);
            ttemp.select<16, 1>(0) = ttemp.select<16, 1>(0) + ttemp.select<16, 1>(16);
            fp32SoftMaxTemp.select<16, 1>(0) = fp32SoftMaxTemp.select<16, 1>(0) + ttemp.select<16, 1>(0);
            fp32HistoricMaxTemp = fp32CurrentMaxTemp;

            if (loopIdx != 0) {
                simd<fp16, 32> compensationTemp;
                compensationTemp.select<16, 1>(0) = fp32SoftMaxCompensation;
                compensationTemp.select<16, 1>(16) = fp32SoftMaxCompensation;
                #pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    finalOutput.select<32, 1>(32 * kk) = finalOutput.select<32, 1>(32 * kk) * compensationTemp.select<32, 1>(0);
                }
            }

            // Pack softmax weights fp32 → fp16 VNNI
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 64 * kk);
                }
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempBufferAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 64 * kk + 32);
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempQkAsFp16.select<32, 2>(128 * k + 64 * kk) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk);
                }
                #pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    tempQkAsFp16.select<32, 2>(128 * k + 64 * kk + 1) = tempOutput.select<32, 1>(128 * k + 512 + 64 * kk + 32);
                }
            }
        }

        barrier();

        // S×V — fp16 DPAS (last iteration, no SLM scatter needed)
        {
            #pragma unroll
            for (int nn = 0; nn < 2; nn++) {
                #pragma unroll
                for (int l = 0; l < 2; l++) {
                    #pragma unroll
                    for (int ll = 0; ll < 2; ll++) {
                        tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
                            slm_block_load<fp16, 512>(slmOffsetBaseV +
                                slmPingpongLoad +
                                16 * 128 * nn * sizeof(fp16) +
                                16 * 64 * l * sizeof(fp16) +
                                512 * ll * sizeof(fp16));
                    }
                    #pragma unroll
                    for (int ll = 0; ll < 8; ll++) {
                        auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
                        auto aaTile = tempBufferAsFp16.select<256, 1>(256 * nn);
                        auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
                        ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                            simd<fp16, 128>(ccTile.data()),
                            simd<fp16, 256>(aaTile.data()),
                            simd<fp16, 128>(bbTile.data()));
                    }
                }
            }
            #pragma unroll
            for (int nn = 0; nn < 2; nn++) {
                #pragma unroll
                for (int l = 0; l < 2; l++) {
                    #pragma unroll
                    for (int ll = 0; ll < 2; ll++) {
                        tempQkAsFp16.select<512, 1>(1024 + 512 * ll) =
                            slm_block_load<fp16, 512>(slmOffsetBaseV +
                                slmPingpongLoad +
                                16 * 128 * 2 * sizeof(fp16) +
                                16 * 128 * nn * sizeof(fp16) +
                                16 * 64 * l * sizeof(fp16) +
                                512 * ll * sizeof(fp16));
                    }
                    #pragma unroll
                    for (int ll = 0; ll < 8; ll++) {
                        auto ccTile = finalOutput.select<128, 1>(1024 * l + 128 * ll);
                        auto aaTile = tempQkAsFp16.select<256, 1>(256 * nn);
                        auto bbTile = tempQkAsFp16.select<128, 1>(1024 + 128 * ll);
                        ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                            simd<fp16, 128>(ccTile.data()),
                            simd<fp16, 256>(aaTile.data()),
                            simd<fp16, 128>(bbTile.data()));
                    }
                }
            }
        }
    }

    // ============================================================
    // OUTPUT: fp16 acc → /sum → bf16 → scatter to output
    // ============================================================
    simd<float, 16> softMaxDividor;
    softMaxDividor.select<16, 1>(0) = fp32SoftMaxTemp;
    softMaxDividor = 1.0f / softMaxDividor;

    // Reuse bf16QState for output staging
    #pragma unroll
    for (int kk = 0; kk < 64; kk++) {
        simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
        simd<float, 32> divMul;
        divMul.select<16, 1>(0) = softMaxDividor.select<16, 1>(0);
        divMul.select<16, 1>(16) = softMaxDividor.select<16, 1>(0);
        f16Temp = f16Temp * divMul;
        // Convert fp32 → bf16
        bf16QState.template select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
        bf16QState.template select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
    }

    // Scatter output rows — bf16, stride-aware
    simd<uint32_t, 16> simdOffsets = baseOffsetInc16AsVector;
    simdOffsets = simdOffsets + q_global_start + my_q_start;
    simd_mask<16> mask = simdOffsets < (uint32_t)num_tokens;
    // Also mask invalid Q rows within this WG
    {
        simd<int32_t, 16> local_q_idx;
        #pragma unroll
        for (int i = 0; i < 16; i++) local_q_idx[i] = my_q_start + i;
        mask &= (local_q_idx < actual_q_rows);
    }
    simdOffsets = simdOffsets * (uint32_t)(num_heads * HD) * sizeof(bf16) + (uint32_t)(head_idx * HD) * sizeof(bf16);
    #pragma unroll
    for (int kk = 0; kk < 16; kk++) {
        __ESIMD_ENS::lsc_scatter<uint32_t, 4, __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back, 16, uint32_t>(
            (uint32_t*)output_ptr, simdOffsets, bf16QState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk), mask);
        simdOffsets += 4 * sizeof(uint32_t);
    }
}
