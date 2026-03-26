/* MoE decode fused ESIMD kernel for W4A16 GPTQ (INT4 symmetric)
 *
 * Pipeline: topk → fused up_gate_silu_mul → down → gather
 *
 * Designed for decode (M=1..8 tokens). Each token selects topk experts.
 * Simple per-element parallelism: 1 thread = 1 output element of 1 expert.
 * Memory-bound — assumes different tokens pick different experts (no reuse).
 *
 * vLLM GPTQ INT4 layout (symmetric, no zero points):
 *   w13_qweight: [E, 2*N, K/2]  uint8  (2 u4 values per byte)
 *   w13_scales:  [E, 2*N, K/GS] bf16
 *   w2_qweight:  [E, K, N/2]    uint8
 *   w2_scales:   [E, K, N/GS]   bf16
 *
 * INT4 dequant (symmetric GPTQ):  (u4_value - 8) * scale
 *
 * IT = input/output type (bf16 or fp16)
 * GS = group size for quantization (32, 64, 128)
 */

#pragma once

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

/* ─── Up+Gate+SiLU fused kernel ──────────────────────────────────────────────
 * For each (token, expert_slot, row) compute:
 *   gate = dot(x, w13_gate[eid, row, :])
 *   up   = dot(x, w13_up[eid, row, :])
 *   out  = silu(gate) * up
 *
 * Parallelism: range<3>(M, topk, N)  — 1 thread = 1 output element
 * w13_qweight layout: [E, 2*N, K/2] — gate at rows [0..N), up at rows [N..2N)
 * w13_scales layout:  [E, 2*N, K/GS]
 */
template<typename IT, int GS>
void moe_up_gate_silu_kernel(
    const IT* __restrict__ x,          // [M, K]
    const uint8_t* __restrict__ w13,   // [E, 2*N, K/2]
    const IT* __restrict__ w13_scales, // [E, 2*N, K/GS]
    const int* __restrict__ topk_ids,  // [M, topk]
    IT* __restrict__ intermediates,    // [M, topk, N]
    const int M, const int N, const int K,
    const int topk, const int num_experts,
    sycl::nd_item<3> item) {

    const int token = item.get_global_id(0);
    const int slot  = item.get_global_id(1);
    const int row   = item.get_global_id(2);

    if (token >= M || slot >= topk || row >= N) return;

    const int eid = topk_ids[token * topk + slot];
    if (eid < 0 || eid >= num_experts) {
        intermediates[(token * topk + slot) * N + row] = IT(0);
        return;
    }

    const int half_K = K / 2;
    const int num_groups = K / GS;
    const int two_N = 2 * N;

    // Pointers to gate row and up row for this expert
    const uint8_t* gate_w = w13 + (size_t)eid * two_N * half_K + (size_t)row * half_K;
    const uint8_t* up_w   = w13 + (size_t)eid * two_N * half_K + (size_t)(N + row) * half_K;
    const IT* gate_s = w13_scales + (size_t)eid * two_N * num_groups + (size_t)row * num_groups;
    const IT* up_s   = w13_scales + (size_t)eid * two_N * num_groups + (size_t)(N + row) * num_groups;

    const IT* xptr = x + (size_t)token * K;

    simd<float, GS> gate_acc(0.f), up_acc(0.f);

    for (int k = 0; k < K; k += GS) {
        // Load input
        simd<float, GS> xv = convert<float>(block_load<IT, GS>(xptr + k));

        // Load and convert scales (bf16/fp16 → float)
        float gate_scale = (float)gate_s[k / GS];
        float up_scale   = (float)up_s[k / GS];

        // Gate: load GS/2 packed bytes → GS u4 values, dequantize
        simd<uint8_t, GS / 2> gp = block_load<uint8_t, GS / 2>(gate_w + k / 2);
        simd<uint8_t, GS> gu;
        gu.template select<GS / 2, 2>(0) = gp & 0x0F;
        gu.template select<GS / 2, 2>(1) = (gp >> 4) & 0x0F;
        simd<float, GS> gw_dq = (convert<float>(gu) - 8.0f) * gate_scale;
        gate_acc += xv * gw_dq;

        // Up: same pattern
        simd<uint8_t, GS / 2> up = block_load<uint8_t, GS / 2>(up_w + k / 2);
        simd<uint8_t, GS> uu;
        uu.template select<GS / 2, 2>(0) = up & 0x0F;
        uu.template select<GS / 2, 2>(1) = (up >> 4) & 0x0F;
        simd<float, GS> uw_dq = (convert<float>(uu) - 8.0f) * up_scale;
        up_acc += xv * uw_dq;
    }

    float g = sycl::ext::intel::esimd::detail::sum<float, float, GS>(gate_acc);
    float u = sycl::ext::intel::esimd::detail::sum<float, float, GS>(up_acc);

    // SiLU(gate) * up
    float silu_g = g / (1.f + sycl::exp(-g));
    intermediates[(token * topk + slot) * N + row] = IT(silu_g * u);
}


/* ─── Down projection kernel ─────────────────────────────────────────────────
 * For each (token, expert_slot, row) compute:
 *   result = dot(intermediates[token, slot, :], w2[eid, row, :])
 *
 * Parallelism: range<3>(M, topk, K)  — 1 thread = 1 output row
 * w2_qweight layout: [E, K, N/2]
 * w2_scales layout:  [E, K, N/GS]
 */
template<typename IT, int GS>
void moe_down_kernel(
    const IT* __restrict__ intermediates, // [M, topk, N]
    const uint8_t* __restrict__ w2,       // [E, K, N/2]
    const IT* __restrict__ w2_scales,     // [E, K, N/GS]
    const int* __restrict__ topk_ids,     // [M, topk]
    IT* __restrict__ down_out,            // [M, topk, K]
    const int M, const int N, const int K,
    const int topk, const int num_experts,
    sycl::nd_item<3> item) {

    const int token = item.get_global_id(0);
    const int slot  = item.get_global_id(1);
    const int row   = item.get_global_id(2);

    if (token >= M || slot >= topk || row >= K) return;

    const int eid = topk_ids[token * topk + slot];
    if (eid < 0 || eid >= num_experts) {
        down_out[(token * topk + slot) * K + row] = IT(0);
        return;
    }

    const int half_N = N / 2;
    const int num_groups = N / GS;

    const uint8_t* dw = w2 + (size_t)eid * K * half_N + (size_t)row * half_N;
    const IT* ds = w2_scales + (size_t)eid * K * num_groups + (size_t)row * num_groups;
    const IT* hi = intermediates + (size_t)(token * topk + slot) * N;

    simd<float, GS> acc(0.f);

    for (int n = 0; n < N; n += GS) {
        simd<float, GS> hv = convert<float>(block_load<IT, GS>(hi + n));

        float scale = (float)ds[n / GS];

        simd<uint8_t, GS / 2> dp = block_load<uint8_t, GS / 2>(dw + n / 2);
        simd<uint8_t, GS> du;
        du.template select<GS / 2, 2>(0) = dp & 0x0F;
        du.template select<GS / 2, 2>(1) = (dp >> 4) & 0x0F;
        simd<float, GS> dw_dq = (convert<float>(du) - 8.0f) * scale;
        acc += hv * dw_dq;
    }

    float result = sycl::ext::intel::esimd::detail::sum<float, float, GS>(acc);
    down_out[(token * topk + slot) * K + row] = IT(result);
}


/* ─── Gather (weighted accumulation) kernel ──────────────────────────────────
 * For each (token, dim) accumulate:
 *   output[token, dim] = sum_over_slot(topk_weights[token, slot] * down_out[token, slot, dim])
 *
 * Parallelism: range<2>(M, K)  — 1 thread = 1 output element
 */
template<typename IT>
void moe_gather_kernel(
    const IT* __restrict__ down_out,          // [M, topk, K]
    const float* __restrict__ topk_weights,   // [M, topk]
    IT* __restrict__ output,                  // [M, K]
    const int M, const int K, const int topk,
    sycl::nd_item<2> item) {

    const int token = item.get_global_id(0);
    const int dim   = item.get_global_id(1);

    if (token >= M || dim >= K) return;

    float acc = 0.f;
    for (int s = 0; s < topk; s++) {
        float w = topk_weights[token * topk + s];
        float v = (float)down_out[(token * topk + s) * K + dim];
        acc += w * v;
    }
    output[token * K + dim] = IT(acc);
}
