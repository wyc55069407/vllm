/* General-purpose W4A16 GPTQ ESIMD GEMV kernel for non-MoE linear layers.
 *
 * Computes: output[M, N] = dequant(W[N, K/2]) @ x[M, K]
 *
 * Reuses dequant_dot from moe_decode.h (proven 370 GB/s on BMG).
 * Includes fused gate_up_silu variant for shared expert.
 *
 * Weight layout (oneDNN u4):
 *   weight: [N, K/2] uint8  — 2 u4 values per byte (lo=even_k, hi=odd_k)
 *   scales: [N, K/GS] IT    — contiguous per-row (transposed from oneDNN at load)
 *
 * INT4 dequant (symmetric GPTQ): (u4_value - 8) * scale
 *
 * IT = input/output type (bf16 or fp16)
 * GS = group size (32, 64, 128)
 * VL = vector length per K iteration (must be multiple of GS)
 * ROWS = output rows per thread (input reuse factor)
 *
 * Dispatch: nd_range<3>(M, 1, ceil(N/ROWS)) with range<3>(1,1,1) local
 *
 * BMG target: 450 GB/s peak, >80% = 360+ GB/s
 */

#pragma once

#include "moe_decode.h"  // reuse dequant_dot


/* ─── General W4A16 GEMV ─────────────────────────────────────────────────────
 * y[M, N] = dequant(W[N, K/2]) @ x[M, K]
 *
 * Each thread computes ROWS output elements for 1 token.
 * Input loaded once per K chunk, reused across ROWS output rows.
 */
template<typename IT, int GS, int VL, int ROWS>
void w4a16_gemv_kernel(
    const IT* __restrict__ x,          // [M, K]
    const uint8_t* __restrict__ w,     // [N, K/2]  packed u4
    const IT* __restrict__ scales,     // [N, K/GS] contiguous per-row
    IT* __restrict__ output,           // [M, N]
    const int M, const int N, const int K,
    sycl::nd_item<3> item) {

    const int m  = item.get_global_id(0);   // token index
    const int rg = item.get_global_id(2);   // row group index
    const int base_row = rg * ROWS;

    if (m >= M || base_row >= N) return;

    const int half_K = K / 2;
    const int nsg = K / GS;       // number of scale groups per row
    const IT* xptr = x + (size_t)m * K;

    float sums[ROWS] = {};

    for (int k = 0; k < K; k += VL) {
        simd<float, VL> input_f = convert<float>(block_load<IT, VL>(xptr + k));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            int row = base_row + r;
            if (row >= N) break;

            sums[r] += dequant_dot<IT, GS, VL>(
                w + (size_t)row * half_K + k / 2,
                scales + (size_t)row * nsg + k / GS,
                input_f);
        }
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        int row = base_row + r;
        if (row >= N) break;
        *(output + (size_t)m * N + row) = IT(sums[r]);
    }
}


/* ─── Fused Gate+Up+SiLU for shared expert ────────────────────────────────────
 * Computes: output[M, N] = SiLU(gate) * up
 *   where gate[i] = dot(W_gate[i], x),  up[i] = dot(W_up[i], x)
 *
 * Weight layout: w[2*N, K/2] — first N rows = gate, next N rows = up
 * Scales: scales[2*N, K/GS]
 *
 * Each thread: ROWS output elements, computing both gate and up dot products
 * per row, then applying SiLU(gate) * up in registers.
 */
template<typename IT, int GS, int VL, int ROWS>
void w4a16_gate_up_silu_kernel(
    const IT* __restrict__ x,          // [M, K]
    const uint8_t* __restrict__ w,     // [2*N, K/2]  packed u4
    const IT* __restrict__ scales,     // [2*N, K/GS]
    IT* __restrict__ output,           // [M, N]
    const int M, const int N, const int K,
    sycl::nd_item<3> item) {

    const int m  = item.get_global_id(0);
    const int rg = item.get_global_id(2);
    const int base_row = rg * ROWS;

    if (m >= M || base_row >= N) return;

    const int half_K = K / 2;
    const int nsg = K / GS;
    const int two_N = 2 * N;
    const IT* xptr = x + (size_t)m * K;

    float gate_sums[ROWS] = {};
    float up_sums[ROWS] = {};

    for (int k = 0; k < K; k += VL) {
        simd<float, VL> input_f = convert<float>(block_load<IT, VL>(xptr + k));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            int row = base_row + r;
            if (row >= N) break;

            // Gate (first N rows)
            gate_sums[r] += dequant_dot<IT, GS, VL>(
                w + (size_t)row * half_K + k / 2,
                scales + (size_t)row * nsg + k / GS,
                input_f);
            // Up (next N rows, offset by N)
            up_sums[r] += dequant_dot<IT, GS, VL>(
                w + (size_t)(N + row) * half_K + k / 2,
                scales + (size_t)(N + row) * nsg + k / GS,
                input_f);
        }
    }

    // SiLU(gate) * up
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        int row = base_row + r;
        if (row >= N) break;
        float g = gate_sums[r];
        float u = up_sums[r];
        float silu_g = g / (1.f + sycl::exp(-g));
        *(output + (size_t)m * N + row) = IT(silu_g * u);
    }
}
