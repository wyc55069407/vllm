#pragma once

/*
 * InfLLM v2 - Pattern Detection Kernels (Torch Extension Format)
 *
 * Ported from standalone reference implementations:
 *   infllmv2_state1.cpp       — k_pooling, qk_gemm, qk_max_pooling, topk_indices
 *   infllmv2_state1_decode.cpp — qk_gemm_decode, softmax_decode, pooling_decode
 *
 * Key differences from standalone versions:
 *   - No queue& parameter; the .sycl dispatch file handles queue submission
 *   - Each function takes sycl::nd_item or sycl::id as the last parameter
 *   - Batch-relative offsets; batch dimension handled by caller
 *
 * Based on: InfLLM-V2: Dense-Sparse Switchable Attention
 * https://arxiv.org/abs/2509.24663
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

using fp16 = sycl::half;

// ============================================================================
// Kernel 1: K Cache Sliding Window Pooling
// ============================================================================
//
// Each thread pools one (kv_head, block) pair.
// idx: 3D id with [batch, kv_head, block_idx] — provided by caller.
//
// key_cache:  [bsz, NUM_KV_HEADS, kv_len, HEAD_DIM]
// key_pooled: [bsz, NUM_KV_HEADS, num_blocks, HEAD_DIM]

template<int NUM_KV_HEADS, int HEAD_DIM, int KERNEL_SIZE, int KERNEL_STRIDE>
ESIMD_INLINE void infllmv2_k_pooling(
    fp16* key_cache,
    fp16* key_pooled,
    int kv_len,
    int num_blocks,
    sycl::id<3> idx
) {
    int batch_idx   = idx[0];
    int kv_head_idx = idx[1];
    int block_idx   = idx[2];

    int window_start = block_idx * KERNEL_STRIDE;
    int window_end   = window_start + KERNEL_SIZE;
    window_end = std::min(window_end, kv_len);
    int actual_window_size = window_end - window_start;

    simd<fp16, HEAD_DIM> k_sum(fp16(0.0f));

    size_t kv_head_base = ((batch_idx * NUM_KV_HEADS + kv_head_idx) * kv_len)
                          * HEAD_DIM * sizeof(fp16);

    for (int pos = window_start; pos < window_end; pos++) {
        size_t key_offset = kv_head_base + pos * HEAD_DIM * sizeof(fp16);
        simd<fp16, HEAD_DIM> k_vec = block_load<fp16, HEAD_DIM>(key_cache, key_offset);
        k_sum += k_vec;
    }

    fp16 scale = fp16(1.0f / actual_window_size);
    simd<fp16, HEAD_DIM> k_mean = k_sum * scale;

    size_t output_offset = ((batch_idx * NUM_KV_HEADS + kv_head_idx) * num_blocks + block_idx)
                           * HEAD_DIM * sizeof(fp16);
    block_store<fp16, HEAD_DIM>(key_pooled, output_offset, k_mean);
}

// ============================================================================
// Kernel 2: QK GEMM with Online Softmax — XMX/DPAS
// ============================================================================
//
// Each thread handles one (batch, kv_head, seq_idx) tuple.
// Uses 2D memory access (config_2d_mem_access, lsc_load_2d) and DPAS.
//
// query:        [bsz, NUM_HEADS, seq_len, HEAD_DIM]
// key_pooled:   [bsz, NUM_KV_HEADS, num_blocks, HEAD_DIM]
// block_scores: [bsz, NUM_KV_HEADS, seq_len, num_blocks]

template<int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int VS, int KERNEL_STRIDE>
ESIMD_INLINE void infllmv2_qk_gemm(
    fp16* query,
    fp16* key_pooled,
    fp16* block_scores,
    int seq_len,
    int num_blocks,
    int cache_len,
    bool causal,
    sycl::nd_item<3> item
) {
    constexpr int GQA_SIZE = NUM_HEADS / NUM_KV_HEADS;

    constexpr int M  = 8;
    constexpr int K  = 16;
    constexpr int N  = 16;
    constexpr int MS = GQA_SIZE / M;
    constexpr int KS = HEAD_DIM / K;
    constexpr int NS = VS       / N;
    static_assert(GQA_SIZE % M == 0, "GQA_SIZE must be divisible by M=8");
    static_assert(HEAD_DIM % K == 0, "HEAD_DIM must be divisible by K=16");
    static_assert(VS       % N == 0, "VS must be divisible by N=16");

    int batch_idx   = item.get_global_id(0);
    int kv_head_idx = item.get_global_id(1);
    int seq_idx     = item.get_global_id(2);

    int boundary = causal ? std::min((seq_idx + cache_len) / KERNEL_STRIDE, num_blocks - 1)
                          : num_blocks - 1;

    fp16 scale = fp16(1.0f / std::sqrt(static_cast<float>(HEAD_DIM)));

    config_2d_mem_access<fp16, K, M, 1> q_payload(
        query + (size_t)batch_idx * NUM_HEADS * seq_len * HEAD_DIM,
        (uint32_t)(seq_len * HEAD_DIM * sizeof(fp16)) - 1u,
        (uint32_t)NUM_HEADS - 1u,
        (uint32_t)(seq_len * HEAD_DIM * sizeof(fp16)) - 1u,
        0, 0);
    config_2d_mem_access<uint32_t, K / 2, N, 1> k_payload(
        reinterpret_cast<const uint32_t*>(key_pooled)
            + (batch_idx * NUM_KV_HEADS + kv_head_idx) * num_blocks * (HEAD_DIM / 2),
        (uint32_t)(HEAD_DIM * sizeof(fp16)) - 1u,
        (uint32_t)num_blocks - 1u,
        (uint32_t)(HEAD_DIM * sizeof(fp16)) - 1u,
        0, 0);

    simd<fp16, M * K> q_tiles[MS][KS];
    #pragma unroll
    for (int mt = 0; mt < MS; mt++) {
        q_payload.set_y(kv_head_idx * GQA_SIZE + mt * M);
        #pragma unroll
        for (int kt = 0; kt < KS; kt++) {
            q_payload.set_x(seq_idx * HEAD_DIM + kt * K);
            q_tiles[mt][kt] = lsc_load_2d<fp16, K, M, 1,
                /*Transposed=*/false, /*Transformed=*/false,
                cache_hint::streaming, cache_hint::uncached>(q_payload) * scale;
        }
    }

    simd<fp16, GQA_SIZE> max_f(-65504.0f);
    simd<float, GQA_SIZE> sum_f(0.0f);
    simd<fp16, M * N> o_tiles[MS];

    auto compute_acc = [&](int bs) [[intel::sycl_explicit_simd]] {
        #pragma unroll
        for (int mt = 0; mt < MS; mt++) o_tiles[mt] = simd<fp16, M * N>(0.0f);
        k_payload.set_y(bs);
        #pragma unroll
        for (int kt = 0; kt < KS; kt++) {
            k_payload.set_x(kt * (K / 2));
            auto k_u32 = lsc_load_2d<uint32_t, K / 2, N, 1,
                /*Transposed=*/true, /*Transformed=*/false,
                cache_hint::cached, cache_hint::cached>(k_payload);
            auto k_tile = k_u32.template bit_cast_view<fp16>().read();
            #pragma unroll
            for (int mt = 0; mt < MS; mt++) {
                o_tiles[mt] = xmx::dpas<8, 8, fp16, fp16, fp16, fp16>(
                    o_tiles[mt], k_tile, q_tiles[mt][kt]);
            }
        }
    };

    // ---- Pass 1: compute running max and sum_exp ----
    for (int bs = 0; bs < num_blocks; bs += N) {
        if (bs > boundary) continue;

        compute_acc(bs);

        simd_mask<N> causal_mask = (bs + simd<int, N>(0, 1)) <= boundary;
        #pragma unroll
        for (int mt = 0; mt < MS; mt++)
            #pragma unroll
            for (int m = 0; m < M; m++)
                o_tiles[mt].template select<N, 1>(m * N) =
                    merge(o_tiles[mt].template select<N, 1>(m * N).read(),
                          simd<fp16, N>(-65504.0f), causal_mask);

        simd<fp16, GQA_SIZE> blk_max;
        #pragma unroll
        for (int mt = 0; mt < MS; mt++)
            #pragma unroll
            for (int m = 0; m < M; m++)
                blk_max[mt * M + m] = hmax<fp16>(o_tiles[mt].template select<N, 1>(m * N).read());

        simd<fp16, GQA_SIZE> new_max = max(max_f, blk_max);
        simd<fp16, GQA_SIZE> blk_sum;
        #pragma unroll
        for (int mt = 0; mt < MS; mt++)
            #pragma unroll
            for (int m = 0; m < M; m++)
                blk_sum[mt * M + m] = sycl::ext::intel::esimd::detail::sum<fp16, fp16, N>(
                    exp(o_tiles[mt].template select<N, 1>(m * N).read()
                        - static_cast<fp16>(new_max[mt * M + m])));

        sum_f = sum_f * exp(max_f - new_max) + blk_sum;
        max_f = new_max;
    }

    // ---- Pass 2: recompute scores, apply softmax, write output ----
    for (int bs = 0; bs < num_blocks; bs += N) {
        simd<fp16, N> out(0.0f);

        if (bs <= boundary) {
            compute_acc(bs);

            simd_mask<N> causal_mask = (bs + simd<int, N>(0, 1)) <= boundary;
            #pragma unroll
            for (int mt = 0; mt < MS; mt++)
                #pragma unroll
                for (int m = 0; m < M; m++)
                    o_tiles[mt].template select<N, 1>(m * N) =
                        merge(o_tiles[mt].template select<N, 1>(m * N).read(),
                              simd<fp16, N>(-65504.0f), causal_mask);

            #pragma unroll
            for (int mt = 0; mt < MS; mt++)
                #pragma unroll
                for (int m = 0; m < M; m++)
                    out += exp(o_tiles[mt].template select<N, 1>(m * N).read()
                               - static_cast<fp16>(max_f[mt * M + m]))
                           / static_cast<fp16>(sum_f[mt * M + m]);
            out /= (fp16)GQA_SIZE;
        }

        size_t out_off = ((batch_idx * NUM_KV_HEADS + kv_head_idx) * seq_len
                          + seq_idx) * num_blocks + bs;
        int actual_n = std::min(N, num_blocks - bs);
        if (actual_n == N) {
            block_store<fp16, N>(block_scores, out_off * sizeof(fp16),
                                 out, properties{alignment<2>});
        } else {
            for (int c = 0; c < actual_n; c++)
                block_scores[out_off + c] = out[c];
        }
    }
}

// ============================================================================
// Kernel 3: QK Block Scores Max Pooling
// ============================================================================
//
// Each thread processes one (batch, kv_head, seq_idx) row.
// idx: 3D id with [batch, kv_head, seq_idx] — provided by caller.
//
// block_scores:  [bsz, NUM_KV_HEADS, seq_len, num_blocks]
// pooled_scores: [bsz, NUM_KV_HEADS, seq_len, num_pooled]

template<int NUM_KV_HEADS, int BLOCK_SIZE, int KERNEL_STRIDE>
ESIMD_INLINE void infllmv2_qk_max_pooling(
    fp16* block_scores,
    fp16* pooled_scores,
    int seq_len,
    int num_blocks,
    int num_pooled,
    int init_block,
    int local_block,
    int cache_len,
    sycl::id<3> idx
) {
    constexpr int POOLING_STRIDE = BLOCK_SIZE / KERNEL_STRIDE;
    constexpr int WINDOW_SIZE = POOLING_STRIDE + 2;
    constexpr int BS = 32;

    size_t input_base  = ((idx[0] * NUM_KV_HEADS + idx[1]) * seq_len + idx[2]) * num_blocks;
    size_t output_base = ((idx[0] * NUM_KV_HEADS + idx[1]) * seq_len + idx[2]) * num_pooled;

    // Initialize base offsets: [0, POOLING_STRIDE, 2*POOLING_STRIDE, ...]
    simd<int, BS> base_offsets = simd<int, BS>(0, 1) * POOLING_STRIDE;

    // Compute q_idx once per thread
    int q_idx = (int(idx[2]) + cache_len) / BLOCK_SIZE;

    // Process outputs in batches using gather operations
    for (int pool_idx = 0; pool_idx < num_pooled; pool_idx += BS) {
        simd<fp16, BS> max_elems(-65504.0f);  // -inf for FP16

        // Compute and clamp base window offsets
        simd<int, BS> base_win_offsets =
            max(pool_idx * POOLING_STRIDE + base_offsets - 1, 0);

        // Iterate through window positions
        #pragma unroll
        for (int w = 0; w < WINDOW_SIZE; w++) {
            simd<int, BS> offsets = min(base_win_offsets + w, num_blocks - 1);

            // Gather and update max
            max_elems = max(max_elems,
                gather<fp16, BS>(block_scores + input_base,
                    simd<uint32_t, BS>(offsets) * sizeof(fp16)));
        }

        // Apply init/local attention mask
        simd<int, BS> kv_idxs = pool_idx + simd<int, BS>(0, 1);
        simd_mask<BS> mask = (kv_idxs < init_block) | ((kv_idxs <= q_idx) & (kv_idxs + local_block >= q_idx));
        max_elems = merge(simd<fp16, BS>(fp16(65504.0f)), max_elems, mask);

        // Store results
        int actual_size = std::min(BS, num_pooled - pool_idx);
        if (actual_size == BS) {
            block_store<fp16, BS>(pooled_scores,
                (output_base + pool_idx) * sizeof(fp16),
                max_elems, properties{alignment<2>});
        } else {
            for (int i = 0; i < actual_size; i++) {
                pooled_scores[output_base + pool_idx + i] = max_elems[i];
            }
        }
    }
}

// ============================================================================
// Kernel 4: FP16 TopK Index Selection
// ============================================================================
//
// Each thread finds top-K indices for one (batch, kv_head, seq_idx) row.
// idx: 3D id with [batch, kv_head, seq_idx] — provided by caller.
//
// input:  [bsz, NUM_KV_HEADS, seq_len, num_pooled]
// output: [bsz, NUM_KV_HEADS, seq_len, TOPK]

template<int NUM_KV_HEADS, int TOPK>
ESIMD_INLINE void infllmv2_topk_indices(
    const fp16* input,
    int32_t* output,
    int seq_len,
    int num_pooled,
    sycl::id<3> idx
) {
    constexpr int BS = 64;

    const int b = idx[0];
    const int h = idx[1];
    const int s = idx[2];

    const size_t in_base =
        ((size_t)(b * NUM_KV_HEADS + h) * seq_len + s) * num_pooled;
    const size_t out_base =
        ((size_t)(b * NUM_KV_HEADS + h) * seq_len + s) * TOPK;

    // Load the first TOPK elements as initial candidates.
    simd<fp16, TOPK> topk_vals = block_load<fp16, TOPK>(
        input, in_base * sizeof(fp16), properties{alignment<2>});
    simd<int32_t, TOPK> topk_idxs(0, 1);
    fp16 min_val = hmin<fp16>(topk_vals);

    constexpr int MASK_N = TOPK / 32;
    auto replace_min = [&](fp16 val, int new_idx) [[intel::sycl_explicit_simd]] {
        simd<uint32_t, MASK_N> masks;
        #pragma unroll
        for (int mi = 0; mi < MASK_N; mi++) {
            simd<fp16, 32> c = topk_vals.template select<32, 1>(mi * 32);
            masks[mi] = pack_mask(c == min_val);
        }
        simd<uint32_t, MASK_N> fbl_result = fbl(masks);
        simd<int32_t, MASK_N> bit_indices = fbl_result.template bit_cast_view<int32_t>();
        simd<int32_t, MASK_N> global_indices =
            bit_indices + simd<int32_t, MASK_N>(0, 32);
        global_indices = merge(bit_indices, global_indices, bit_indices == -1);
        int pos = (int)hmax<int32_t>(global_indices);
        topk_vals[pos] = val;
        topk_idxs[pos] = new_idx;
        min_val = hmin<fp16>(topk_vals);
    };

    const int n_full = num_pooled / BS;
    for (int ci = TOPK / BS; ci < n_full; ci++) {
        simd<fp16, BS> chunk = block_load<fp16, BS>(
            input,
            (in_base + ci * BS) * sizeof(fp16),
            properties{alignment<2>});

        #pragma unroll
        for (int j = 0; j < BS; j++) {
            const fp16 val = chunk[j];
            if (val > min_val)
                replace_min(val, ci * BS + j);
        }
    }

    const int tail_start = n_full * BS;
    for (int i = tail_start; i < num_pooled; i++) {
        const fp16 val = input[in_base + i];
        if (val > min_val)
            replace_min(val, i);
    }

    block_store<int32_t, TOPK>(output, out_base * sizeof(int32_t), topk_idxs);
}

// ============================================================================
// Kernel 5: QK GEMM Decode
// ============================================================================
//
// Each thread handles multiple K blocks for one (batch, kv_head, seq_idx).
// Thread-ID based striping over num_blocks; no SLM or barriers.
//
// query:        [bsz, NUM_HEADS, seq_len, HEAD_DIM]
// key_pooled:   [bsz, NUM_KV_HEADS, num_blocks, HEAD_DIM]
// block_scores: [bsz, NUM_HEADS, seq_len, num_blocks]

template<int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int KERNEL_STRIDE>
ESIMD_INLINE void infllmv2_qk_gemm_decode(
    fp16* query,
    fp16* key_pooled,
    fp16* block_scores,
    int seq_len,
    int num_blocks,
    int cache_len,
    bool causal,
    int ts,
    sycl::nd_item<3> item
) {
    static_assert(NUM_HEADS % NUM_KV_HEADS == 0, "NUM_HEADS must be divisible by NUM_KV_HEADS");
    constexpr int GQA_SIZE = NUM_HEADS / NUM_KV_HEADS;
    const fp16 SCALE = fp16(1.0f / std::sqrt(float(HEAD_DIM)));

    const int batch_idx   = item.get_global_id(0);
    const int kv_head_idx = item.get_global_id(1);
    const int z           = item.get_global_id(2);
    const int seq_idx     = z / ts;
    const int tid         = z % ts;

    // Shared base: (batch * NUM_HEADS + kv_head * GQA_SIZE) * seq_len + seq_idx
    const size_t base_hs =
        (size_t)(batch_idx * NUM_HEADS + kv_head_idx * GQA_SIZE) * seq_len + seq_idx;
    const size_t head_stride_bytes = (size_t)seq_len * HEAD_DIM * sizeof(fp16);

    // Load all Q vectors once, reused across all K blocks this thread handles
    simd<fp16, HEAD_DIM> q_vecs[GQA_SIZE];
    #pragma unroll
    for (int i = 0; i < GQA_SIZE; i++)
        q_vecs[i] = block_load<fp16, HEAD_DIM>(query,
            base_hs * HEAD_DIM * sizeof(fp16) + i * head_stride_bytes);

    const int boundary = causal ? (cache_len + seq_idx) / KERNEL_STRIDE : num_blocks - 1;

    const simd<uint32_t, GQA_SIZE> offsets(0, (uint32_t)(seq_len * num_blocks * sizeof(fp16)));

    for (int blk_idx = tid; blk_idx < num_blocks; blk_idx += ts) {
        // Load K block [1, HEAD_DIM], pre-multiply scale
        const simd<fp16, HEAD_DIM> k_scaled =
            block_load<fp16, HEAD_DIM>(key_pooled,
                ((size_t)(batch_idx * NUM_KV_HEADS + kv_head_idx) * num_blocks + blk_idx)
                * HEAD_DIM * sizeof(fp16)) * SCALE;

        simd<fp16, GQA_SIZE> scores_out;
        if (blk_idx > boundary) {
            scores_out = fp16(-65504.0f);
        } else {
            #pragma unroll
            for (int i = 0; i < GQA_SIZE; i++)
                scores_out[i] =
                    sycl::ext::intel::esimd::detail::sum<fp16, fp16, HEAD_DIM>(
                        q_vecs[i] * k_scaled);
        }

        scatter<fp16, GQA_SIZE>(block_scores + base_hs * num_blocks + blk_idx, offsets, scores_out);
    }
}

// ============================================================================
// Kernel 6: Softmax Decode (SLM-based cooperative reduction)
// ============================================================================
//
// Each work-group handles one (batch, head, seq_idx) row of length num_blocks.
// WG_SIZE threads stride over num_blocks cooperatively and reduce via SLM.
// NOTE: This kernel uses reqd_sub_group_size(32), NOT sycl_explicit_simd.
//
// block_scores: [bsz, NUM_HEADS, seq_len, num_blocks] (modified in-place)

template<int NUM_HEADS, int WG_SIZE>
void infllmv2_softmax_decode_kernel(
    fp16* block_scores,
    int seq_len,
    int num_blocks,
    sycl::nd_item<3> item,
    sycl::local_accessor<float, 1> slm
) {
    const int batch_idx = item.get_global_id(0);
    const int head_idx  = item.get_global_id(1);
    const int seq_idx   = item.get_global_id(2) / WG_SIZE;
    const int tid       = item.get_local_id(2);

    fp16 *row = block_scores +
        ((size_t)(batch_idx * NUM_HEADS + head_idx) * seq_len + seq_idx) * num_blocks;

    // Pass 1: find max
    float xmax = -std::numeric_limits<float>::max();
    for (int i = tid; i < num_blocks; i += WG_SIZE)
        xmax = std::max(xmax, static_cast<float>(row[i]));
    slm[tid] = xmax;

    item.barrier();

    #pragma unroll
    for (int i = 0; i < WG_SIZE; i++)
        xmax = std::max(xmax, slm[i]);

    // Pass 2: sum exp(x - xmax)
    float exp_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += WG_SIZE)
        exp_sum += sycl::exp(static_cast<float>(row[i]) - xmax);
    slm[tid + WG_SIZE] = exp_sum;

    item.barrier();

    exp_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < WG_SIZE; i++)
        exp_sum += slm[i + WG_SIZE];

    // Pass 3: normalize
    for (int i = tid; i < num_blocks; i += WG_SIZE)
        row[i] = fp16(sycl::exp(static_cast<float>(row[i]) - xmax) / exp_sum);
}

// ============================================================================
// Kernel 7: Pooling Decode (GQA mean over query heads)
// ============================================================================
//
// For each output element (b, kv_h, s, blk), averages the GQA_SIZE input
// values from the corresponding Q heads.
//
// input:  [bsz, NUM_HEADS,    seq_len, num_blocks]
// output: [bsz, NUM_KV_HEADS, seq_len, num_blocks]

template<int NUM_HEADS, int NUM_KV_HEADS>
ESIMD_INLINE void infllmv2_pooling_decode(
    fp16* input,
    fp16* output,
    int seq_len,
    int num_blocks,
    sycl::nd_item<3> item
) {
    static_assert(NUM_HEADS % NUM_KV_HEADS == 0, "NUM_HEADS must be divisible by NUM_KV_HEADS");
    constexpr int GQA_SIZE = NUM_HEADS / NUM_KV_HEADS;

    const int batch_idx   = item.get_global_id(0);
    const int kv_head_idx = item.get_global_id(1);
    const int z           = item.get_global_id(2);
    const int seq_idx     = z / num_blocks;
    const int block_idx   = z % num_blocks;

    // Base: input[b, kv_h*GQA_SIZE, s, blk] -- first Q head of this KV group
    const size_t base_off =
        ((size_t)(batch_idx * NUM_HEADS + kv_head_idx * GQA_SIZE) * seq_len + seq_idx)
        * num_blocks + block_idx;

    // Gather GQA_SIZE values; consecutive Q heads are stride bytes apart
    const uint32_t stride_bytes = (uint32_t)(seq_len * num_blocks * sizeof(fp16));
    simd<uint32_t, GQA_SIZE> offsets(0, stride_bytes);
    simd<fp16, GQA_SIZE> vals = gather<fp16, GQA_SIZE>(input + base_off, offsets);

    // Mean in float to avoid half-precision accumulation error
    const float sum_val = sycl::ext::intel::esimd::detail::sum<float, float, GQA_SIZE>(
        simd<float, GQA_SIZE>(vals));

    // Write mean
    const size_t out_idx =
        ((size_t)(batch_idx * NUM_KV_HEADS + kv_head_idx) * seq_len + seq_idx)
        * num_blocks + block_idx;
    output[out_idx] = fp16(sum_val / GQA_SIZE);
}
