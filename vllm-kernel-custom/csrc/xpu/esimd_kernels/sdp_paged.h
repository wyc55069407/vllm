/* sdp_paged.h — Optimized paged SDP kernels for Intel BMG XPU.
 *
 * Kernels:
 *   1. sdp_paged_kernel_scalar          — Scalar fallback for decode/prefill (all HD)
 *   2. sdp_paged_decode_opt_phase1      — Optimized two-phase decode phase 1 (WG+SLM)
 *   3. sdp_paged_decode_opt_phase2      — Optimized two-phase decode phase 2 (reduction)
 *   4. sdp_paged_prefill_dpas           — DPAS-based prefill (HD=256, 32-thread WG)
 *   5. sdp_paged_prefill_dpas_128       — DPAS-based prefill (HD=128, 16-thread WG)
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
 * Scratch layout for optimized decode kernels (fp32, like reference).
 * Three separate buffers indexed by [chunk_idx * num_heads + head_idx]:
 *   scratch_out : float[num_chunks * num_heads * HD]  — partial output (fp32)
 *   scratch_max : float[num_chunks * num_heads]       — partial max
 *   scratch_lse : float[num_chunks * num_heads]       — partial lse
 *
 * Total per chunk per head: HD*4 + 4 + 4 bytes = 520B (HD=128), 1032B (HD=256)
 * ============================================================ */
template<uint32_t HD>
static constexpr int DEC_OPT_SCRATCH_PER_CHUNK = (8 + HD * 2 + 15) & ~15;


/* ============================================================
 * OPTIMIZED DECODE PHASE 1 — Workgroup-level, SLM-reduced
 *
 * Template parameters:
 *   HD            : head dimension (128 or 256)
 *   Q_HEAD_PER_T  : Q heads processed per ESIMD thread (4 or 8)
 *   sp_blk_size   : KV tokens per thread (e.g. 64)
 *   chunk_size    : total KV tokens per WG chunk (e.g. 256)
 *   IS_BF16       : true for bf16, false for fp16
 *
 * 3D nd_range:
 *   global(1, chunk_num * sp_blk_num_per_t,
 *          batch * headKv * head_groups_per_g)
 *   local (1, sp_blk_num_per_t, head_groups_per_g)
 *
 * Each thread: sp_blk_size KV tokens, Q_HEAD_PER_T Q heads.
 * K/V loaded once per token, dot product for all Q_HEAD_PER_T heads.
 * Online softmax within sp_blk, SLM reduction across sp_blks,
 * then write chunk partial to global scratch.
 * ============================================================ */
template<uint32_t HD, uint32_t NUM_KV_HEADS, uint32_t Q_HEAD_PER_T,
         uint32_t sp_blk_size, uint32_t chunk_size,
         uint32_t HEAD_GROUPS_PER_G, bool IS_BF16>
ESIMD_INLINE void sdp_paged_decode_opt_phase1(
    const unsigned short* __restrict__ query_ptr,
    const unsigned short* __restrict__ kv_cache_ptr,
    float* __restrict__ scratch_out,      // [num_chunks, num_heads, HD]
    float* __restrict__ scratch_max,      // [num_chunks, num_heads]
    float* __restrict__ scratch_lse,      // [num_chunks, num_heads]
    const int* __restrict__ block_table_ptr,
    const int* __restrict__ seq_lens_ptr,
    int num_heads,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_split, int64_t kv_stride_block,
    float attn_scale,
    int num_chunks_per_seq,
    int batch,
    nd_item<3>& ndi)
{
    using namespace sycl::ext::intel::esimd;

    constexpr int sp_blk_num_per_t = chunk_size / sp_blk_size;
    // Compile-time constants matching the reference kernel pattern
    constexpr int KV_STRIDE_POS = NUM_KV_HEADS * HD;  // stride between tokens (in elements)
    constexpr int q_head_num_per_kv_head = HEAD_GROUPS_PER_G * Q_HEAD_PER_T;

    // 3D nd_range (reference pattern):
    //   global: (1, chunk_num * sp_blk_num_per_t, batch * nkvh * HEAD_GROUPS_PER_G)
    //   local:  (1, sp_blk_num_per_t, HEAD_GROUPS_PER_G)
    // Head groups in same WG share L1 cache for K/V reads (critical for 16:1 GQA)
    int chunk_idx      = ndi.get_group(1);
    int sp_blk_idx     = ndi.get_local_id(1);
    int head_group_idx = ndi.get_local_id(2);   // 0 .. HEAD_GROUPS_PER_G-1 (within WG)
    int kv_wg_idx      = ndi.get_group(2);       // 0 .. batch * nkvh - 1
    int kv_head_idx    = kv_wg_idx % (int)NUM_KV_HEADS;
    int req_idx        = kv_wg_idx / (int)NUM_KV_HEADS;
    int q_head_idx     = kv_head_idx * q_head_num_per_kv_head
                        + head_group_idx * Q_HEAD_PER_T;

    if (req_idx >= batch) return;

    int seq_len = seq_lens_ptr[req_idx];

    // KV range for this thread
    int kv_logical_start = chunk_idx * chunk_size + sp_blk_idx * sp_blk_size;
    int kv_end = kv_logical_start + sp_blk_size;
    if (kv_end > seq_len) kv_end = seq_len;
    if (kv_logical_start >= seq_len) kv_end = kv_logical_start;

    // SLM layout (reference pattern — includes HEAD_GROUPS_PER_G):
    //   [0 .. slm_reduce_size)           : output float[hg, sp, Q_HEAD_PER_T, HD]
    //   [slm_reduce_size .. +slm_max)    : max    float[hg, sp, Q_HEAD_PER_T]
    //   [+slm_max .. +slm_lse)           : lse    float[hg, sp, Q_HEAD_PER_T]
    constexpr int slm_reduce_size =
        HEAD_GROUPS_PER_G * sp_blk_num_per_t * Q_HEAD_PER_T * HD * sizeof(float);
    constexpr int slm_max_size =
        HEAD_GROUPS_PER_G * sp_blk_num_per_t * Q_HEAD_PER_T * sizeof(float);
    constexpr int slm_lse_size =
        HEAD_GROUPS_PER_G * sp_blk_num_per_t * Q_HEAD_PER_T * sizeof(float);

    // Init SLM if we have multiple threads to reduce (sp_blks or head_groups)
    if constexpr (sp_blk_num_per_t > 1) {
        slm_init(slm_reduce_size + slm_max_size + slm_lse_size);
    }

    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;

    // Load Q — reference pattern: single load + scale
    const unsigned short* q_base = query_ptr +
        (int64_t)req_idx * num_heads * HD;

    simd<fp16, Q_HEAD_PER_T * HD> qIn;
    if constexpr (!IS_BF16) {
        // fp16: direct block_load (reference pattern — no conversion)
        const fp16* q_fp16 = reinterpret_cast<const fp16*>(
            q_base + (int64_t)q_head_idx * HD);
        qIn = block_load<fp16, Q_HEAD_PER_T * HD>(q_fp16);
        qIn = qIn * attn_scale;
    } else {
        // bf16: load → float → scale → fp16
        #pragma unroll
        for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
            if constexpr (HD == 128) {
                simd<float, 64> q0 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD);
                simd<float, 64> q1 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD + 64);
                q0 *= attn_scale; q1 *= attn_scale;
                qIn.template select<64, 1>(h * HD) = q0;
                qIn.template select<64, 1>(h * HD + 64) = q1;
            } else {
                simd<float, 64> q0 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD) * attn_scale;
                simd<float, 64> q1 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD + 64) * attn_scale;
                simd<float, 64> q2 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD + 128) * attn_scale;
                simd<float, 64> q3 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD + 192) * attn_scale;
                qIn.template select<64, 1>(h * HD) = q0;
                qIn.template select<64, 1>(h * HD + 64) = q1;
                qIn.template select<64, 1>(h * HD + 128) = q2;
                qIn.template select<64, 1>(h * HD + 192) = q3;
            }
        }
    }

    // Online softmax state
    simd<float, Q_HEAD_PER_T> maxKq          = FP32_MIN;
    simd<float, Q_HEAD_PER_T> old_maxKq      = FP32_MIN;
    simd<float, Q_HEAD_PER_T> max_correction = FP32_MIN;
    simd<float, Q_HEAD_PER_T> lse            = 0;
    simd<float, Q_HEAD_PER_T * HD> output    = 0;

    // ---- Paged inner loop (compile-time stride, no while loop) ----
    // block_size >= sp_blk_size guaranteed, so no block boundary within an sp_blk.
    // Single block lookup per sp_blk (reference pattern).
    int kv_start = kv_logical_start;
    int valid_t = kv_end - kv_start;

    int blk_idx = kv_start / block_size;
    int blk_off = kv_start & (block_size - 1);
    int blk_num = block_table_row[blk_idx];

    // Compute K and V base pointers (ONLY int64 ops, done once)
    int head_off = kv_head_idx * (int)HD;
    const unsigned short* k_base = kv_cache_ptr
        + (int64_t)blk_num * kv_stride_block + head_off;
    const unsigned short* v_base = k_base + (int)kv_stride_split;

    // Compile-time stride: KV_STRIDE_POS = NUM_KV_HEADS * HD (matches reference)
    // Starting offset within block, using compile-time stride
    int kv_off = blk_off * KV_STRIDE_POS;

    for (int t = 0; t < valid_t; t++, kv_off += KV_STRIDE_POS) {
        // Load K and V together (V load hides behind K dot product ALU)
        const unsigned short* k_ptr = k_base + kv_off;
        const unsigned short* v_ptr = v_base + kv_off;
        simd<fp16, HD> kIn;
        simd<fp16, HD> vIn;
        if constexpr (!IS_BF16) {
            kIn = block_load<fp16, HD>(reinterpret_cast<const fp16*>(k_ptr));
            vIn = block_load<fp16, HD>(reinterpret_cast<const fp16*>(v_ptr));
        } else if constexpr (HD == 128) {
            simd<float, 64> k0 = sdp_load_bf16_64(k_ptr);
            simd<float, 64> k1 = sdp_load_bf16_64(k_ptr + 64);
            simd<float, 64> v0 = sdp_load_bf16_64(v_ptr);
            simd<float, 64> v1 = sdp_load_bf16_64(v_ptr + 64);
            kIn.template select<64, 1>(0) = k0;
            kIn.template select<64, 1>(64) = k1;
            vIn.template select<64, 1>(0) = v0;
            vIn.template select<64, 1>(64) = v1;
        } else {
            simd<float, 64> k0 = sdp_load_bf16_64(k_ptr);
            simd<float, 64> k1 = sdp_load_bf16_64(k_ptr + 64);
            simd<float, 64> k2 = sdp_load_bf16_64(k_ptr + 128);
            simd<float, 64> k3 = sdp_load_bf16_64(k_ptr + 192);
            simd<float, 64> v0 = sdp_load_bf16_64(v_ptr);
            simd<float, 64> v1 = sdp_load_bf16_64(v_ptr + 64);
            simd<float, 64> v2 = sdp_load_bf16_64(v_ptr + 128);
            simd<float, 64> v3 = sdp_load_bf16_64(v_ptr + 192);
            kIn.template select<64, 1>(0) = k0;
            kIn.template select<64, 1>(64) = k1;
            kIn.template select<64, 1>(128) = k2;
            kIn.template select<64, 1>(192) = k3;
            vIn.template select<64, 1>(0) = v0;
            vIn.template select<64, 1>(64) = v1;
            vIn.template select<64, 1>(128) = v2;
            vIn.template select<64, 1>(192) = v3;
        }

        // QK dot products
        simd<float, Q_HEAD_PER_T> kq_out;
        #pragma unroll
        for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
            kq_out[h] = sycl::ext::intel::esimd::detail::sum<float, fp16, HD>(
                qIn.template select<HD, 1>(h * HD) * kIn);
        }

        // Online softmax
        old_maxKq = maxKq;
        maxKq = __ESIMD_NS::max<float, Q_HEAD_PER_T, float>(kq_out, old_maxKq);
        kq_out = kq_out - maxKq;
        kq_out = __ESIMD_NS::exp2<float, Q_HEAD_PER_T, float>(
            kq_out * sycl::ext::intel::esimd::detail::log2e);

        if (t >= 1) {
            max_correction = old_maxKq - maxKq;
            max_correction = __ESIMD_NS::exp2<float, Q_HEAD_PER_T, float>(
                max_correction * sycl::ext::intel::esimd::detail::log2e);
            #pragma unroll
            for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
                output.template select<HD, 1>(h * HD) =
                    output.template select<HD, 1>(h * HD) * max_correction[h];
            }
            lse = lse * max_correction;
        }
        lse = lse + kq_out;

        // V accumulation (V already loaded — no memory stall)
        #pragma unroll
        for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
            float w = kq_out[h];
            simd<float, HD> vf = vIn * w;
            output.template select<HD, 1>(h * HD) =
                output.template select<HD, 1>(h * HD) + vf;
        }
    }

    // =========================================================
    // SLM reduction (only when sp_blk_num_per_t > 1)
    // =========================================================
    if constexpr (sp_blk_num_per_t > 1) {
        // Store this sp_blk's partial result into SLM
        // SLM index: head_group_idx * sp_blk_num_per_t + sp_blk_idx (reference pattern)
        int idx_slm = head_group_idx * sp_blk_num_per_t + sp_blk_idx;
        slm_block_store<float, Q_HEAD_PER_T>(
            slm_reduce_size + idx_slm * Q_HEAD_PER_T * sizeof(float), maxKq);
        slm_block_store<float, Q_HEAD_PER_T>(
            slm_reduce_size + slm_max_size + idx_slm * Q_HEAD_PER_T * sizeof(float), lse);

        // Store output as single large SLM store (matching reference pattern)
        slm_block_store<float, Q_HEAD_PER_T * HD>(
            idx_slm * Q_HEAD_PER_T * HD * sizeof(float), output);

        output = 0;
        barrier();

        // Intra-chunk reduce: sp_blk_idx==0 thread reduces all sp_blks (within its head_group)
        if (sp_blk_idx == 0) {
            int slm_r_h_offset = (head_group_idx * sp_blk_num_per_t) * Q_HEAD_PER_T * sizeof(float);
            int slm_r_o_h_offset = (head_group_idx * sp_blk_num_per_t) * Q_HEAD_PER_T * HD * sizeof(float);

            simd<float, Q_HEAD_PER_T> max_final = FP32_MIN;
            simd<float, Q_HEAD_PER_T> lse_final = 0;

            // Pass 1: find global max
            for (int c_idx = 0; c_idx < sp_blk_num_per_t; c_idx++) {
                simd<float, Q_HEAD_PER_T> cur_max = slm_block_load<float, Q_HEAD_PER_T>(
                    slm_reduce_size + slm_r_h_offset + c_idx * Q_HEAD_PER_T * sizeof(float));
                max_final = __ESIMD_NS::max<float, Q_HEAD_PER_T, float>(cur_max, max_final);
            }

            // Pass 2: accumulate with correction
            for (int c_idx = 0; c_idx < sp_blk_num_per_t; c_idx++) {
                simd<float, Q_HEAD_PER_T> cur_max = slm_block_load<float, Q_HEAD_PER_T>(
                    slm_reduce_size + slm_r_h_offset + c_idx * Q_HEAD_PER_T * sizeof(float));
                simd<float, Q_HEAD_PER_T> cur_lse = slm_block_load<float, Q_HEAD_PER_T>(
                    slm_reduce_size + slm_max_size + slm_r_h_offset
                    + c_idx * Q_HEAD_PER_T * sizeof(float));
                simd<float, Q_HEAD_PER_T> correction =
                    __ESIMD_NS::exp2<float, Q_HEAD_PER_T, float>(
                        (cur_max - max_final) * sycl::ext::intel::esimd::detail::log2e);
                lse_final = lse_final + cur_lse * correction;

                #pragma unroll
                for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
                    float corr_h = correction[h];
                    // Single HD-sized SLM load per head (matching reference pattern)
                    simd<float, HD> cur_o = slm_block_load<float, HD>(
                        slm_r_o_h_offset + (c_idx * Q_HEAD_PER_T + h) * HD * sizeof(float));
                    output.template select<HD, 1>(h * HD) =
                        output.template select<HD, 1>(h * HD) + cur_o * corr_h;
                }
            }

            // Write chunk result to global scratch
            maxKq = max_final;
            lse = lse_final;
        } else {
            return;  // Only sp_blk_idx==0 writes global scratch
        }
    }

    // Write to global scratch as fp32
    // Layout: scratch_*[(req_idx * num_chunks + chunk_idx) * num_heads + head_idx]
    int scratch_req_base = req_idx * num_chunks_per_seq * num_heads;
    #pragma unroll
    for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
        int scratch_idx = scratch_req_base + chunk_idx * num_heads + (q_head_idx + h);
        scratch_max[scratch_idx] = maxKq[h];
        scratch_lse[scratch_idx] = lse[h];
        float* out_base = scratch_out + (int64_t)scratch_idx * HD;
        // Single HD-sized global store per head (matching reference pattern)
        block_store<float, HD>(out_base, output.template select<HD, 1>(h * HD));
    }
}


/* ============================================================
 * OPTIMIZED DECODE PHASE 2 — Cross-chunk reduction
 *
 * Template parameters:
 *   HD              : head dimension (128 or 256)
 *   HEADS_PER_THREAD: Q heads per ESIMD thread (16 for HD=128, 4 for HD=256)
 *   IS_BF16         : true for bf16, false for fp16
 *
 * Each workitem handles HEADS_PER_THREAD consecutive Q heads.
 * Global threads: batch * num_heads / HEADS_PER_THREAD.
 * Two-pass: global max across chunks, then accumulate with correction.
 * ============================================================ */
template<uint32_t HD, uint32_t HEADS_PER_THREAD, bool IS_BF16>
ESIMD_INLINE void sdp_paged_decode_opt_phase2(
    float* __restrict__ scratch_out,      // [num_chunks, num_heads, HD]
    float* __restrict__ scratch_max,      // [num_chunks, num_heads]
    float* __restrict__ scratch_lse,      // [num_chunks, num_heads]
    unsigned short* __restrict__ output_ptr,
    const int* __restrict__ seq_lens_ptr,
    int num_heads,
    int num_chunks_per_seq,
    int batch,
    nd_item<1>& ndi)
{
    using namespace sycl::ext::intel::esimd;

    int global_id = ndi.get_global_id(0);
    int q_head_start = global_id * HEADS_PER_THREAD;
    int req_idx = q_head_start / num_heads;
    q_head_start = q_head_start % num_heads;

    if (req_idx >= batch) return;

    int seq_len = seq_lens_ptr[req_idx];
    if (seq_len <= 0) return;

    int actual_chunks = num_chunks_per_seq;

    // Scratch layout: scratch_*[chunk * num_heads + head]
    // For batch>1: scratch is per-request, offset by req_idx * num_chunks_per_seq * num_heads
    int scratch_req_offset = req_idx * num_chunks_per_seq * num_heads;

    if constexpr (HD == 128 && HEADS_PER_THREAD == 16) {
        // Reference-matching pattern: process 16 heads simultaneously
        // GRF: reduce_final = simd<fp32, 128*16> = 8KB, fits in 16KB doubleGRF
        simd<float, HEADS_PER_THREAD> max_final = FP32_MIN;
        simd<float, HD * HEADS_PER_THREAD> reduce_final = 0;
        simd<float, HEADS_PER_THREAD> lse_final = 0;

        int base_out = q_head_start * HD;
        int base_lsmax = q_head_start;

        // Pass 1: global max across all chunks for these 16 heads
        for (int ck = 0; ck < actual_chunks; ck++) {
            simd<float, HEADS_PER_THREAD> cur_max =
                block_load<float, HEADS_PER_THREAD>(
                    scratch_max + scratch_req_offset + base_lsmax + ck * num_heads);
            max_final = __ESIMD_NS::max<float, HEADS_PER_THREAD, float>(cur_max, max_final);
        }

        // Pass 2: weighted accumulate
        for (int ck = 0; ck < actual_chunks; ck++) {
            int32_t out_ck = base_out + (scratch_req_offset + ck * num_heads) * HD;
            int32_t lsmax_ck = base_lsmax + scratch_req_offset + ck * num_heads;

            simd<float, HEADS_PER_THREAD> cur_max =
                block_load<float, HEADS_PER_THREAD>(scratch_max + lsmax_ck);
            simd<float, HEADS_PER_THREAD> cur_lse =
                block_load<float, HEADS_PER_THREAD>(scratch_lse + lsmax_ck);
            simd<float, HEADS_PER_THREAD> correction =
                __ESIMD_NS::pow<float, HEADS_PER_THREAD, float>(2.718f, cur_max - max_final);
            lse_final = lse_final + cur_lse * correction;

            #pragma unroll
            for (int j = 0; j < (int)HEADS_PER_THREAD; j++) {
                simd<float, HD> cur_o = block_load<float, HD>(scratch_out + out_ck + j * HD);
                float corr_j = correction[j];
                reduce_final.template select<HD, 1>(HD * j) =
                    cur_o * corr_j +
                    reduce_final.template select<HD, 1>(HD * j);
            }
        }

        // Normalize by lse and write fp16/bf16 final output
        unsigned short* out_base = output_ptr +
            (int64_t)req_idx * num_heads * HD + (int64_t)q_head_start * HD;
        #pragma unroll
        for (int i = 0; i < (int)HEADS_PER_THREAD; i++) {
            reduce_final.template select<HD, 1>(i * HD) =
                reduce_final.template select<HD, 1>(i * HD) / lse_final[i];
            sdp_store_64<IS_BF16>(out_base + i * HD,
                reduce_final.template select<64, 1>(i * HD));
            sdp_store_64<IS_BF16>(out_base + i * HD + 64,
                reduce_final.template select<64, 1>(i * HD + 64));
        }
    } else {
        // HD=256 or small HEADS_PER_THREAD: per-head loop with HD-sized loads
        #pragma unroll
        for (int j = 0; j < (int)HEADS_PER_THREAD; j++) {
            int head_idx = q_head_start + j;
            int base_idx = scratch_req_offset + head_idx;

            // Read first chunk
            float global_max = scratch_max[base_idx];
            float global_sum = scratch_lse[base_idx];
            float* out_base0 = scratch_out + (int64_t)base_idx * HD;

            simd<float, HD> acc;
            if constexpr (HD == 128) {
                acc = block_load<float, HD>(out_base0);
            } else {
                // HD=256: load in 2 halves
                acc.template select<128, 1>(0) = block_load<float, 128>(out_base0);
                acc.template select<128, 1>(128) = block_load<float, 128>(out_base0 + 128);
            }

            // Merge remaining chunks
            for (int c = 1; c < actual_chunks; c++) {
                int idx = scratch_req_offset + c * num_heads + head_idx;
                float chunk_max = scratch_max[idx];
                float chunk_sum = scratch_lse[idx];

                if (chunk_sum == 0.0f) continue;

                float* out_base_c = scratch_out + (int64_t)idx * HD;
                simd<float, HD> cur;
                if constexpr (HD == 128) {
                    cur = block_load<float, HD>(out_base_c);
                } else {
                    cur.template select<128, 1>(0) = block_load<float, 128>(out_base_c);
                    cur.template select<128, 1>(128) = block_load<float, 128>(out_base_c + 128);
                }

                float new_max = (chunk_max > global_max) ? chunk_max : global_max;
                float corr_old = sdp_esimd_expf(global_max - new_max);
                float corr_new = sdp_esimd_expf(chunk_max - new_max);

                acc = acc * corr_old + cur * corr_new;
                global_sum = global_sum * corr_old + chunk_sum * corr_new;
                global_max = new_max;
            }

            // Normalize
            if (global_sum > 0.0f) {
                acc *= (1.0f / global_sum);
            }

            // Store final output as fp16/bf16
            unsigned short* out_row = output_ptr +
                (int64_t)req_idx * num_heads * HD +
                (int64_t)head_idx * HD;

            sdp_store_64<IS_BF16>(out_row, acc.template select<64, 1>(0));
            sdp_store_64<IS_BF16>(out_row + 64, acc.template select<64, 1>(64));
            if constexpr (HD == 256) {
                sdp_store_64<IS_BF16>(out_row + 128, acc.template select<64, 1>(128));
                sdp_store_64<IS_BF16>(out_row + 192, acc.template select<64, 1>(192));
            }
        }
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
