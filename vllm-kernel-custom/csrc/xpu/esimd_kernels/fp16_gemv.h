/* FP16/BF16 ESIMD GEMV kernel for unquantized linear layers.
 *
 * Computes: output[M, N] = weight[N, K] @ x[M, K]^T
 *
 * K-split with SLM reduction: GROUP_SIZE threads share one output row,
 * each computing K/GROUP_SIZE elements then reducing via SLM.
 *
 * DT  = sycl::half or sycl::ext::oneapi::bfloat16
 * SIMD_W = 128 (256 bytes per load, max efficient on Xe2)
 * GROUP_SIZE = 4 (threads per output row)
 *
 * Dispatch: nd_range<2>(N * GROUP_SIZE, GROUP_SIZE)  per token m.
 *           Outer loop over M inside kernel.
 *
 * BMG target: 450 GB/s peak, memory-bound.
 */

#pragma once

#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl;

template<typename DT, int SIMD_W = 128, int GROUP_SIZE = 4>
ESIMD_INLINE void fp16_gemv_kernel(
    const DT* __restrict__ x,       // [M, K]
    const DT* __restrict__ weight,  // [N, K] row-major
    DT* __restrict__ output,        // [M, N]
    const int M, const int N, const int K,
    nd_item<2> item) {

    const int row      = item.get_group(0);   // output row [0, N)
    const int local_id = item.get_local_id(1); // [0, GROUP_SIZE)

    if (row >= N) return;

    constexpr int SLM_BYTES = GROUP_SIZE * 8 * sizeof(float); // max M=8
    slm_init<SLM_BYTES>();

    const int k_per_thread = K / GROUP_SIZE;
    const int k_start = local_id * k_per_thread;
    const int iters = k_per_thread / SIMD_W;

    const DT* wptr = weight + (size_t)row * K;

    for (int m = 0; m < M; m++) {
        const DT* xptr = x + (size_t)m * K;

        simd<float, SIMD_W> acc(0.0f);

        #pragma unroll
        for (int i = 0; i < iters; i++) {
            int k = k_start + i * SIMD_W;
            simd<DT, SIMD_W> xv = block_load<DT, SIMD_W>(xptr + k);
            simd<DT, SIMD_W> wv = block_load<DT, SIMD_W>(wptr + k);
            acc += convert<float>(xv) * convert<float>(wv);
        }

        float partial = sycl::ext::intel::esimd::detail::sum<float, float, SIMD_W>(acc);

        // SLM reduction
        int slm_off = (m * GROUP_SIZE + local_id) * sizeof(float);
        slm_block_store<float, 1>(slm_off, simd<float, 1>(partial));
        barrier();

        if (local_id == 0) {
            simd<float, GROUP_SIZE> parts =
                slm_block_load<float, GROUP_SIZE>(m * GROUP_SIZE * sizeof(float));
            float result = sycl::ext::intel::esimd::detail::sum<float, float, GROUP_SIZE>(parts);
            *(output + (size_t)m * N + row) = DT(result);
        }
        barrier();
    }
}
