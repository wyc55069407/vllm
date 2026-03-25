/* infllmv2_utils.h — Utility ESIMD kernels for InfLLMv2 pipeline.
 *
 * Kernels:
 *   1. infllmv2_k_pooling_paged — K pooling directly from paged KV cache
 *   2. infllmv2_force_last_block — GPU-side last-block insertion into topk mask
 *
 * These eliminate CPU-side waits in the sparse attention pipeline.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <algorithm>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

/* ============================================================
 * Kernel 1: Paged K Pooling
 *
 * Reads K directly from paged KV cache using block_table,
 * performs sliding-window mean pooling, outputs contiguous pooled K.
 *
 * KV cache layout: [2, num_pages, page_size, num_kv_heads, head_dim]
 *   - kv_cache[0] = key cache (at offset 0 from base pointer)
 *   - Stride-aware via kv_stride_block (elements per page)
 *
 * IS_BF16: true if kv_cache dtype is bfloat16, false for float16.
 *   KV data is loaded as unsigned short and converted to fp16 for accumulation.
 *
 * block_table: [batch, max_blocks_per_seq] i32
 * seq_lens:    [batch] i32
 * key_pooled:  [batch, num_kv_heads, num_pooled_blocks, head_dim] fp16
 *
 * Each thread handles one (batch, kv_head, pooled_block) triple.
 * idx: 3D id [batch_idx, kv_head_idx, pooled_block_idx]
 * ============================================================ */
template<int NUM_KV_HEADS, int HEAD_DIM, int KERNEL_SIZE, int KERNEL_STRIDE, bool IS_BF16>
ESIMD_INLINE void infllmv2_k_pooling_paged(
    const unsigned short* __restrict__ kv_cache_ptr,  // paged KV cache base
    const int* __restrict__ block_table_ptr,          // [batch, max_blocks_per_seq]
    const int* __restrict__ seq_lens_ptr,             // [batch]
    fp16* __restrict__ key_pooled,                    // [batch, nkvh, num_pooled, hd]
    int max_blocks_per_seq,
    int page_size,                                    // vLLM block_size (128)
    int num_pooled_blocks,
    int64_t kv_stride_block,                          // stride per page block (elements)
    sycl::id<3> idx
) {
    int batch_idx   = idx[0];
    int kv_head_idx = idx[1];
    int block_idx   = idx[2];

    if (block_idx >= num_pooled_blocks) return;

    int seq_len = seq_lens_ptr[batch_idx];
    const int* bt_row = block_table_ptr + batch_idx * max_blocks_per_seq;

    int window_start = block_idx * KERNEL_STRIDE;
    int window_end   = window_start + KERNEL_SIZE;
    if (window_end > seq_len) window_end = seq_len;
    int actual_window_size = window_end - window_start;
    if (actual_window_size <= 0) {
        size_t out_offset = ((batch_idx * NUM_KV_HEADS + kv_head_idx)
                            * num_pooled_blocks + block_idx)
                           * HEAD_DIM * sizeof(fp16);
        simd<fp16, HEAD_DIM> zeros(fp16(0.0f));
        block_store<fp16, HEAD_DIM>(key_pooled, out_offset, zeros);
        return;
    }

    // Accumulate in fp16
    simd<fp16, HEAD_DIM> k_sum(fp16(0.0f));

    for (int pos = window_start; pos < window_end; pos++) {
        int page_idx = pos / page_size;
        int token_in_page = pos % page_size;
        int phys_page = bt_row[page_idx];

        // Byte offset from kv_cache_ptr to this token's K vector:
        // phys_page * kv_stride_block + token_in_page * nkvh * hd + kv_head_idx * hd
        // All in elements (unsigned short), multiply by sizeof(us) for bytes
        size_t elem_offset = (size_t)phys_page * kv_stride_block
                           + (size_t)token_in_page * NUM_KV_HEADS * HEAD_DIM
                           + (size_t)kv_head_idx * HEAD_DIM;
        size_t byte_offset = elem_offset * sizeof(unsigned short);

        // Load raw unsigned short data
        simd<unsigned short, HEAD_DIM> raw = block_load<unsigned short, HEAD_DIM>(
            kv_cache_ptr, byte_offset);

        // Convert to fp16 for accumulation
        simd<fp16, HEAD_DIM> k_vec;
        if constexpr (IS_BF16) {
            // bf16 → fp16 conversion via simd<uint32_t>:
            // bf16 bits are in upper 16 bits of float32,
            // so shift left by 16 → reinterpret as float → convert to fp16
            simd<uint32_t, HEAD_DIM> raw32 = raw;  // zero-extend us → u32
            raw32 = raw32 << 16;                    // bf16 bits → float32 bits
            simd<float, HEAD_DIM> k_f32 = raw32.template bit_cast_view<float>();
            k_vec = k_f32;  // float → fp16 conversion
        } else {
            // fp16: direct reinterpret
            k_vec = raw.template bit_cast_view<fp16>();
        }

        k_sum += k_vec;
    }

    fp16 scale = fp16(1.0f / actual_window_size);
    simd<fp16, HEAD_DIM> k_mean = k_sum * scale;

    size_t out_offset = ((batch_idx * NUM_KV_HEADS + kv_head_idx)
                        * num_pooled_blocks + block_idx)
                       * HEAD_DIM * sizeof(fp16);
    block_store<fp16, HEAD_DIM>(key_pooled, out_offset, k_mean);
}


/* ============================================================
 * Kernel 2: Force Last Block Insertion
 *
 * For each (batch, kv_head) pair, ensures the last sparse block
 * (containing the current decode token) is in the topk mask.
 * If not present, replaces the last slot.
 *
 * sparse_mask: [batch, num_kv_heads, topk] i32  (topk block indices)
 * seq_lens:    [batch] i32
 * sparse_block_size: 64 (tokens per sparse block)
 *
 * Dispatch: 2D range [batch, num_kv_heads], 1 thread per pair.
 * ============================================================ */
template<int TOPK>
ESIMD_INLINE void infllmv2_force_last_block(
    int32_t* __restrict__ sparse_mask,   // [batch, nkvh, TOPK]
    const int* __restrict__ seq_lens,    // [batch]
    int num_kv_heads,
    int sparse_block_size,
    sycl::id<2> idx
) {
    int batch_idx = idx[0];
    int kv_head_idx = idx[1];

    int seq_len = seq_lens[batch_idx];
    int last_blk = (seq_len - 1) / sparse_block_size;

    int32_t* row = sparse_mask
        + (batch_idx * num_kv_heads + kv_head_idx) * TOPK;

    // Check if last_blk is already present
    // Load TOPK values (64 int32s = 256 bytes, 2 loads of 32)
    simd<int32_t, 32> v0 = block_load<int32_t, 32>(row, 0);
    simd<int32_t, 32> v1 = block_load<int32_t, 32>(row, 32 * sizeof(int32_t));

    simd_mask<32> found0 = (v0 == last_blk);
    simd_mask<32> found1 = (v1 == last_blk);

    if (found0.any() || found1.any()) return;

    // Not found — replace last slot
    row[TOPK - 1] = last_blk;
}
