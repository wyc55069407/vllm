/* MoE decode fused ESIMD kernel for W4A16 GPTQ (INT4 symmetric)
 *
 * Optimized v3: dequant_dot fusion + multi-row (ROWS) batching
 *   - dequant_dot: fuses dequant + dot product WITHOUT materializing full VL
 *     weight vector. Processes GS-sized blocks, uses strided select on input.
 *     Saves ~2KB register pressure vs separate dequant+dot.
 *   - ROWS: each thread computes ROWS output rows, sharing a single input
 *     vector load. Reduces L2 traffic by ROWS x, reduces thread count.
 *   - SIMD select for lo/hi nibble interleaving (proven 2x on W4A16 GEMV)
 *
 * Pipeline: topk -> fused up_gate_silu_mul -> down -> gather
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
 * VL = vector length per iteration (must be multiple of GS)
 * ROWS = output rows per thread (input reuse factor)
 *
 * BMG target: 20 XE cores, 160 EUs, 1280 HW threads
 * Peak BW = 450 GB/s. Target >80% = 360+ GB/s.
 */

#pragma once

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;


/* ─── Helper: Fused dequant + dot product ──────────────────────────────────
 * Loads VL/2 packed bytes, unpacks nibbles per GS block, and computes dot
 * product with input_f using strided select — never materializes a full
 * VL-sized weight vector.
 *
 * Register pressure: ~3KB peak (input_f passed by ref, lo/hi are GS/2-sized)
 * vs ~6KB for separate dequant_block + reduce(input * weight).
 */
template<typename IT, int GS, int VL>
SYCL_ESIMD_FUNCTION inline float dequant_dot(
    const uint8_t* packed_ptr,
    const IT* scale_ptr,
    simd<float, VL>& input_f) {

    constexpr int NUM_BLOCKS = VL / GS;

    simd<uint8_t, VL / 2> packed = block_load<uint8_t, VL / 2>(packed_ptr);
    simd<float, NUM_BLOCKS> scales = convert<float>(
        block_load<IT, NUM_BLOCKS>(scale_ptr));

    float acc = 0.f;

    #pragma unroll
    for (int blk = 0; blk < NUM_BLOCKS; blk++) {
        float sc = scales[blk];
        int poff = blk * (GS / 2);

        auto p = packed.template select<GS / 2, 1>(poff);
        simd<float, GS / 2> lo = p & 0x0F;
        simd<float, GS / 2> hi = (p >> 4) & 0x0F;
        lo = (lo - 8.0f) * sc;
        hi = (hi - 8.0f) * sc;

        int base = blk * GS;
        auto in_lo = input_f.template select<GS / 2, 2>(base + 0);
        auto in_hi = input_f.template select<GS / 2, 2>(base + 1);
        acc += reduce<float>(in_lo * lo + in_hi * hi, std::plus<>());
    }
    return acc;
}


/* ─── Helper: Fused dequant + dot product (transposed scales) ─────────────
 * Same as dequant_dot but loads scales with stride (for transposed layout).
 * Transposed scales: [E, K/GS, 2*N] or [E, N/GS, K]
 * Scale for group g at row r: scale_ptr[g * scale_stride + r]
 * Here scale_ptr points to (first_group * scale_stride + row), stride between
 * consecutive groups = scale_stride.
 */
template<typename IT, int GS, int VL>
SYCL_ESIMD_FUNCTION inline float dequant_dot_ts(
    const uint8_t* packed_ptr,
    const IT* scale_ptr,
    int scale_stride,
    simd<float, VL>& input_f) {

    constexpr int NUM_BLOCKS = VL / GS;

    simd<uint8_t, VL / 2> packed = block_load<uint8_t, VL / 2>(packed_ptr);

    // Load scales with stride (non-contiguous)
    simd<float, NUM_BLOCKS> scales;
    #pragma unroll
    for (int i = 0; i < NUM_BLOCKS; i++) {
        scales[i] = (float)scale_ptr[i * scale_stride];
    }

    float acc = 0.f;

    #pragma unroll
    for (int blk = 0; blk < NUM_BLOCKS; blk++) {
        float sc = scales[blk];
        int poff = blk * (GS / 2);

        auto p = packed.template select<GS / 2, 1>(poff);
        simd<float, GS / 2> lo = p & 0x0F;
        simd<float, GS / 2> hi = (p >> 4) & 0x0F;
        lo = (lo - 8.0f) * sc;
        hi = (hi - 8.0f) * sc;

        int base = blk * GS;
        auto in_lo = input_f.template select<GS / 2, 2>(base + 0);
        auto in_hi = input_f.template select<GS / 2, 2>(base + 1);
        acc += reduce<float>(in_lo * lo + in_hi * hi, std::plus<>());
    }
    return acc;
}


/* ─── Up+Gate+SiLU fused kernel (multi-row, transposed scales) ────────────
 * Same as moe_up_gate_silu_rows but reads scales from transposed layout:
 *   w13_scales_t: [E, K/GS, 2*N]
 */
template<typename IT, int GS, int VL, int ROWS>
void moe_up_gate_silu_rows_ts(
    const IT* __restrict__ x,              // [M, K]
    const uint8_t* __restrict__ w13,       // [E, 2*N, K/2]
    const IT* __restrict__ w13_scales_t,   // [E, K/GS, 2*N]  (transposed)
    const int* __restrict__ topk_ids,      // [M, topk]
    IT* __restrict__ intermediates,        // [M*topk, N]
    const int M, const int N, const int K,
    const int topk, const int num_experts,
    sycl::nd_item<3> item) {

    const int token = item.get_global_id(0);
    const int slot  = item.get_global_id(1);
    const int rg    = item.get_global_id(2);

    if (token >= M || slot >= topk) return;
    const int base_row = rg * ROWS;
    if (base_row >= N) return;

    const int eid = topk_ids[token * topk + slot];
    const int out_off = (token * topk + slot) * N + base_row;

    if (eid < 0 || eid >= num_experts) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            if (base_row + r < N)
                intermediates[out_off + r] = IT(0);
        return;
    }

    const int half_K = K / 2;
    const int nsg = K / GS;       // number of scale groups along K
    const int two_N = 2 * N;
    const IT* xptr = x + (size_t)token * K;

    const size_t expert_w_base = (size_t)eid * two_N * half_K;
    // Transposed: [E, K/GS, 2*N] — expert base is same total size
    const size_t expert_s_base = (size_t)eid * nsg * two_N;

    float gate_sums[ROWS] = {};
    float up_sums[ROWS] = {};

    for (int k = 0; k < K; k += VL) {
        simd<float, VL> input_f = convert<float>(block_load<IT, VL>(xptr + k));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            int row = base_row + r;
            if (row >= N) break;

            // Transposed scale: base + (k/GS) * two_N + row
            gate_sums[r] += dequant_dot_ts<IT, GS, VL>(
                w13 + expert_w_base + (size_t)row * half_K + k / 2,
                w13_scales_t + expert_s_base + (size_t)(k / GS) * two_N + row,
                two_N,
                input_f);
            up_sums[r] += dequant_dot_ts<IT, GS, VL>(
                w13 + expert_w_base + (size_t)(N + row) * half_K + k / 2,
                w13_scales_t + expert_s_base + (size_t)(k / GS) * two_N + (N + row),
                two_N,
                input_f);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (base_row + r >= N) break;
        float g = gate_sums[r];
        float u = up_sums[r];
        float silu_g = g / (1.f + sycl::exp(-g));
        intermediates[out_off + r] = IT(silu_g * u);
    }
}


/* ─── Down projection kernel (multi-row, transposed scales) ───────────────
 * Same as moe_down_rows but reads scales from transposed layout:
 *   w2_scales_t: [E, N/GS, K]
 */
template<typename IT, int GS, int VL_D, int ROWS>
void moe_down_rows_ts(
    const IT* __restrict__ intermediates, // [M*topk, N]
    const uint8_t* __restrict__ w2,       // [E, K, N/2]
    const IT* __restrict__ w2_scales_t,   // [E, N/GS, K]  (transposed)
    const int* __restrict__ topk_ids,     // [M, topk]
    IT* __restrict__ down_out,            // [M*topk, K]
    const int M, const int N, const int K,
    const int topk, const int num_experts,
    sycl::nd_item<3> item) {

    const int token = item.get_global_id(0);
    const int slot  = item.get_global_id(1);
    const int rg    = item.get_global_id(2);

    if (token >= M || slot >= topk) return;
    const int base_row = rg * ROWS;
    if (base_row >= K) return;

    const int eid = topk_ids[token * topk + slot];
    const int out_off = (token * topk + slot) * K + base_row;

    if (eid < 0 || eid >= num_experts) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            if (base_row + r < K)
                down_out[out_off + r] = IT(0);
        return;
    }

    const int half_N = N / 2;
    const int nsg = N / GS;       // number of scale groups along N
    const IT* hi = intermediates + (size_t)(token * topk + slot) * N;

    const size_t expert_w_base = (size_t)eid * K * half_N;
    // Transposed: [E, N/GS, K] — expert base is same total size
    const size_t expert_s_base = (size_t)eid * nsg * K;

    float row_sums[ROWS] = {};

    for (int n = 0; n < N; n += VL_D) {
        simd<float, VL_D> hi_f = convert<float>(block_load<IT, VL_D>(hi + n));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            int row = base_row + r;
            if (row >= K) break;

            // Transposed scale: base + (n/GS) * K + row
            row_sums[r] += dequant_dot_ts<IT, GS, VL_D>(
                w2 + expert_w_base + (size_t)row * half_N + n / 2,
                w2_scales_t + expert_s_base + (size_t)(n / GS) * K + row,
                K,
                hi_f);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (base_row + r >= K) break;
        down_out[out_off + r] = IT(row_sums[r]);
    }
}


/* ─── Up+Gate+SiLU fused kernel (multi-row) ───────────────────────────────
 *
 * 1 thread = ROWS output elements (gate + up for ROWS rows of one expert)
 * Parallelism: range<3>(M, topk, N/ROWS)
 *
 * Each thread loads input x[K] once per VL iteration and reuses it for all
 * ROWS rows' gate and up dequant_dot calls. This gives ROWS x L2 traffic
 * reduction for the input vector.
 */
template<typename IT, int GS, int VL, int ROWS>
void moe_up_gate_silu_rows(
    const IT* __restrict__ x,          // [M, K]
    const uint8_t* __restrict__ w13,   // [E, 2*N, K/2]
    const IT* __restrict__ w13_scales, // [E, 2*N, K/GS]
    const int* __restrict__ topk_ids,  // [M, topk]
    IT* __restrict__ intermediates,    // [M*topk, N]
    const int M, const int N, const int K,
    const int topk, const int num_experts,
    sycl::nd_item<3> item) {

    const int token = item.get_global_id(0);
    const int slot  = item.get_global_id(1);
    const int rg    = item.get_global_id(2);

    if (token >= M || slot >= topk) return;
    const int base_row = rg * ROWS;
    if (base_row >= N) return;

    const int eid = topk_ids[token * topk + slot];
    const int out_off = (token * topk + slot) * N + base_row;

    if (eid < 0 || eid >= num_experts) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            if (base_row + r < N)
                intermediates[out_off + r] = IT(0);
        return;
    }

    const int half_K = K / 2;
    const int nsg = K / GS;
    const int two_N = 2 * N;
    const IT* xptr = x + (size_t)token * K;

    const size_t expert_w_base = (size_t)eid * two_N * half_K;
    const size_t expert_s_base = (size_t)eid * two_N * nsg;

    float gate_sums[ROWS] = {};
    float up_sums[ROWS] = {};

    for (int k = 0; k < K; k += VL) {
        simd<float, VL> input_f = convert<float>(block_load<IT, VL>(xptr + k));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            int row = base_row + r;
            if (row >= N) break;

            gate_sums[r] += dequant_dot<IT, GS, VL>(
                w13 + expert_w_base + (size_t)row * half_K + k / 2,
                w13_scales + expert_s_base + (size_t)row * nsg + k / GS,
                input_f);
            up_sums[r] += dequant_dot<IT, GS, VL>(
                w13 + expert_w_base + (size_t)(N + row) * half_K + k / 2,
                w13_scales + expert_s_base + (size_t)(N + row) * nsg + k / GS,
                input_f);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (base_row + r >= N) break;
        float g = gate_sums[r];
        float u = up_sums[r];
        float silu_g = g / (1.f + sycl::exp(-g));
        intermediates[out_off + r] = IT(silu_g * u);
    }
}


/* ─── Down projection kernel (multi-row) ──────────────────────────────────
 *
 * 1 thread = ROWS output rows (K dim) of down projection.
 * Parallelism: range<3>(M, topk, K/ROWS)
 *
 * Each thread loads intermediate[N] once per VL iteration and reuses it
 * for all ROWS weight rows' dequant_dot calls.
 */
template<typename IT, int GS, int VL_D, int ROWS>
void moe_down_rows(
    const IT* __restrict__ intermediates, // [M*topk, N]
    const uint8_t* __restrict__ w2,       // [E, K, N/2]
    const IT* __restrict__ w2_scales,     // [E, K, N/GS]
    const int* __restrict__ topk_ids,     // [M, topk]
    IT* __restrict__ down_out,            // [M*topk, K]
    const int M, const int N, const int K,
    const int topk, const int num_experts,
    sycl::nd_item<3> item) {

    const int token = item.get_global_id(0);
    const int slot  = item.get_global_id(1);
    const int rg    = item.get_global_id(2);

    if (token >= M || slot >= topk) return;
    const int base_row = rg * ROWS;
    if (base_row >= K) return;

    const int eid = topk_ids[token * topk + slot];
    const int out_off = (token * topk + slot) * K + base_row;

    if (eid < 0 || eid >= num_experts) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++)
            if (base_row + r < K)
                down_out[out_off + r] = IT(0);
        return;
    }

    const int half_N = N / 2;
    const int nsg = N / GS;
    const IT* hi = intermediates + (size_t)(token * topk + slot) * N;

    const size_t expert_w_base = (size_t)eid * K * half_N;
    const size_t expert_s_base = (size_t)eid * K * nsg;

    float row_sums[ROWS] = {};

    for (int n = 0; n < N; n += VL_D) {
        simd<float, VL_D> hi_f = convert<float>(block_load<IT, VL_D>(hi + n));

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            int row = base_row + r;
            if (row >= K) break;

            row_sums[r] += dequant_dot<IT, GS, VL_D>(
                w2 + expert_w_base + (size_t)row * half_N + n / 2,
                w2_scales + expert_s_base + (size_t)row * nsg + n / GS,
                hi_f);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (base_row + r >= K) break;
        down_out[out_off + r] = IT(row_sums[r]);
    }
}


/* ─── Gather (weighted accumulation) kernel ──────────────────────────────────
 * For each (token, dim) accumulate:
 *   output[token, dim] = sum_over_slot(topk_weights[token, slot] * down_out[token, slot, dim])
 *
 * Parallelism: range<2>(M, K) — 1 thread = 1 output element
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
