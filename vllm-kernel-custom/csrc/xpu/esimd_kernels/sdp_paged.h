/* sdp_paged.h — ESIMD paged SDP kernel (decode + prefill) with causal masking.
 *
 * Supports arbitrary block_size (power-of-2) and head_dim=256 (Qwen3.5-4B).
 * KV cache layout: [2, num_blocks, block_size, num_kv_heads, head_dim] bf16 (NHD).
 *   kv_cache[0] = key cache, kv_cache[1] = value cache.
 *
 * Decode: One ESIMD thread per (request, query_head).
 *   Iterates over all KV positions with online softmax.
 *
 * Prefill: One ESIMD thread per (request, query_head, q_token).
 *   Each thread handles one query token, iterates over KV with causal mask.
 *
 * Grid dispatch:
 *   Decode (max_query_len == 1): nd_range<1>(batch * num_heads, 1)
 *   Prefill (max_query_len > 1):  nd_range<1>(total_q_tokens * num_heads, 1)
 */

#include "utils.h"

/* ---- ESIMD scalar math helpers ---- */
/* In ESIMD context, scalar C math (expf) is forbidden.
 * Use simd<float,8> ESIMD intrinsics and extract lane 0. */
ESIMD_INLINE float sdp_esimd_expf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::exp(v);
    return v[0];
}

/* ---- Helpers ---- */

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

/* Dot product of two split-128 f32 vectors. */
ESIMD_INLINE float sdp_dot128(simd<float, 64> a_lo, simd<float, 64> a_hi,
                               simd<float, 64> b_lo, simd<float, 64> b_hi) {
    return sdp_dot64(a_lo, b_lo) + sdp_dot64(a_hi, b_hi);
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

/* Load 64 bf16 values as f32. */
ESIMD_INLINE simd<float, 64> sdp_load_bf16_64(const unsigned short* ptr) {
    simd<unsigned short, 64> raw = block_load<unsigned short, 64>(ptr);
    simd<unsigned int, 64> bits = raw;
    bits <<= 16;
    return bits.template bit_cast_view<float>();
}

/* ---- Main kernel ---- */

/* Tensor layouts:
 *   query:           [num_tokens, num_heads, head_dim]       bf16
 *   kv_cache:        [2, num_blocks, block_size, num_kv_heads, head_dim] bf16
 *   output:          [num_tokens, num_heads, head_dim]       bf16
 *   block_table:     [batch, max_blocks_per_seq]             i32
 *   seq_lens:        [batch]                                 i32
 *   query_start_loc: [batch+1]                               i32
 *
 * For decode: num_tokens == batch (one token per request).
 * For prefill: num_tokens == sum of all query lengths.
 *
 * head_dim can be 128 or 256. The kernel handles both via the HD parameter.
 * GQA: num_heads > num_kv_heads, with group_size = num_heads / num_kv_heads.
 */
ESIMD_INLINE void sdp_paged_kernel(
    const unsigned short* __restrict__ query_ptr,     // bf16
    const unsigned short* __restrict__ kv_cache_ptr,  // bf16
    unsigned short* __restrict__ output_ptr,           // bf16
    const int* __restrict__ block_table_ptr,
    const int* __restrict__ seq_lens_ptr,
    const int* __restrict__ query_start_loc_ptr,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    // KV cache strides (in bf16 elements):
    //   kv_stride_split: stride of dim 0 (between K and V planes)
    //   kv_stride_block: stride of dim 1 (between blocks)
    //   kv_stride_pos:   stride of dim 2 (between positions within block)
    //   kv_stride_head:  stride of dim 3 (between KV heads)
    int64_t kv_stride_split, int64_t kv_stride_block,
    int64_t kv_stride_pos, int64_t kv_stride_head,
    float attn_scale,
    int causal,
    int max_query_len,     // max query length across batch
    nd_item<1>& ndi)
{
    const int tid = ndi.get_global_id(0);

    int req_idx, q_offset_in_req, head_idx;

    if (max_query_len == 1) {
        // Decode mode: tid = req_idx * num_heads + head_idx
        req_idx = tid / num_heads;
        head_idx = tid % num_heads;
        q_offset_in_req = 0;
    } else {
        // Prefill mode: tid = token_linear_idx * num_heads + head_idx
        int token_linear_idx = tid / num_heads;
        head_idx = tid % num_heads;
        // Binary search for req_idx from query_start_loc
        // (simplified: linear scan since batch is small)
        req_idx = 0;
        int batch_size = 0;
        // Find batch_size from the last non-zero entry
        // We compute it from query_start_loc: batch = number of requests
        // The caller passes num_tokens total, we scan query_start_loc
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

    // Absolute query position (for causal masking)
    // For decode: q_pos = seq_len - 1 (last position)
    // For prefill: q_pos = (seq_len - query_len) + q_offset_in_req
    int query_len;
    if (max_query_len == 1) {
        query_len = 1;
    } else {
        query_len = query_start_loc_ptr[req_idx + 1] -
                    query_start_loc_ptr[req_idx];
    }
    const int q_abs_pos = (seq_len - query_len) + q_offset_in_req;

    // Load query: query[token_idx, head_idx, 0:head_dim]
    int token_idx;
    if (max_query_len == 1) {
        token_idx = req_idx;
    } else {
        token_idx = query_start_loc_ptr[req_idx] + q_offset_in_req;
    }

    const unsigned short* q_row = query_ptr +
        (int64_t)token_idx * num_heads * head_dim +
        (int64_t)head_idx * head_dim;

    // Load query into registers (up to 256 dims, split into 64-element chunks)
    simd<float, 64> q0 = sdp_load_bf16_64(q_row);
    simd<float, 64> q1 = sdp_load_bf16_64(q_row + 64);
    simd<float, 64> q2, q3;
    const bool hd256 = (head_dim == 256);
    if (hd256) {
        q2 = sdp_load_bf16_64(q_row + 128);
        q3 = sdp_load_bf16_64(q_row + 192);
    }

    // Online softmax state
    float max_score = -3.402823466e+38f;  // -FLT_MAX
    float sum_exp = 0.0f;
    simd<float, 64> acc0(0.0f), acc1(0.0f), acc2(0.0f), acc3(0.0f);

    // KV cache addressing using passed-in strides (handles any physical layout).
    // Logical shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
    // Key at: kv_cache_ptr + 0 * kv_stride_split + block * kv_stride_block
    //         + pos * kv_stride_pos + kv_head * kv_stride_head + d
    // Val at: kv_cache_ptr + 1 * kv_stride_split + block * kv_stride_block
    //         + pos * kv_stride_pos + kv_head * kv_stride_head + d
    const int64_t kv_head_offset = (int64_t)kv_head_idx * kv_stride_head;

    // Block table for this request
    const int* block_table_row = block_table_ptr +
        (int64_t)req_idx * max_blocks_per_seq;

    // Determine how many KV positions to attend to
    int kv_end = causal ? (q_abs_pos + 1) : seq_len;

    // Iterate over all KV positions
    for (int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
        // Page translation: kv_pos -> physical block + offset
        int block_idx = kv_pos / block_size;
        int block_offset = kv_pos % block_size;
        int block_num = block_table_row[block_idx];

        // Compute base offset for this KV position (key plane, kv_split=0)
        int64_t kv_base = (int64_t)block_num * kv_stride_block +
                          (int64_t)block_offset * kv_stride_pos +
                          kv_head_offset;

        // Load K[kv_pos] from key plane (kv_split=0)
        const unsigned short* k_ptr = kv_cache_ptr + kv_base;
        simd<float, 64> k0 = sdp_load_bf16_64(k_ptr);
        simd<float, 64> k1 = sdp_load_bf16_64(k_ptr + 64);

        // Compute QK dot product
        float score;
        if (hd256) {
            simd<float, 64> k2 = sdp_load_bf16_64(k_ptr + 128);
            simd<float, 64> k3 = sdp_load_bf16_64(k_ptr + 192);
            score = sdp_dot256(q0, q1, q2, q3, k0, k1, k2, k3) * attn_scale;
        } else {
            score = sdp_dot128(q0, q1, k0, k1) * attn_scale;
        }

        // Online softmax update
        float new_max = (score > max_score) ? score : max_score;
        float correction = sdp_esimd_expf(max_score - new_max);
        acc0 *= correction;
        acc1 *= correction;
        if (hd256) { acc2 *= correction; acc3 *= correction; }
        sum_exp *= correction;

        float w = sdp_esimd_expf(score - new_max);
        sum_exp += w;
        max_score = new_max;

        // Load V[kv_pos] from value plane (kv_split=1)
        const unsigned short* v_ptr_pos = kv_cache_ptr + kv_base + kv_stride_split;
        acc0 += w * sdp_load_bf16_64(v_ptr_pos);
        acc1 += w * sdp_load_bf16_64(v_ptr_pos + 64);
        if (hd256) {
            acc2 += w * sdp_load_bf16_64(v_ptr_pos + 128);
            acc3 += w * sdp_load_bf16_64(v_ptr_pos + 192);
        }
    }

    // Finalize: output = acc / sum_exp
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        acc0 *= inv_sum;
        acc1 *= inv_sum;
        if (hd256) { acc2 *= inv_sum; acc3 *= inv_sum; }
    }

    // Store output as bf16
    unsigned short* out_row = output_ptr +
        (int64_t)token_idx * num_heads * head_dim +
        (int64_t)head_idx * head_dim;

    // f32 -> bf16 with rounding
    auto store_bf16 = [](unsigned short* ptr, simd<float, 64> val) {
        simd<unsigned int, 64> bits = val.template bit_cast_view<unsigned int>();
        simd<unsigned int, 64> rounding = ((bits >> 16) & 1u) + 0x7FFFu;
        bits += rounding;
        simd<unsigned short, 64> bf16_bits = (bits >> 16);
        block_store<unsigned short, 64>(ptr, bf16_bits);
    };

    store_bf16(out_row, acc0);
    store_bf16(out_row + 64, acc1);
    if (hd256) {
        store_bf16(out_row + 128, acc2);
        store_bf16(out_row + 192, acc3);
    }
}
