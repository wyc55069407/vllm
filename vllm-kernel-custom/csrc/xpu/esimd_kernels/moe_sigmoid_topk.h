// moe_sigmoid_topk.h — Fused sigmoid + bias + topk + renorm for MoE routing
// Optimized for MiniCPM5: E=160 experts, topk=16
// Standard GRF (common_ops extension)

#pragma once
#include "utils.h"
#include <sycl/ext/intel/esimd/math.hpp>
#include <limits>

// Horizontal max of simd<float, 32> via tree reduction
inline float hmax32(simd<float, 32> v) {
    // 32 → 16
    simd<float, 16> lo16(v.select<16, 1>(0));
    simd<float, 16> hi16(v.select<16, 1>(16));
    simd<float, 16> mx16;
    mx16.merge(lo16, hi16, lo16 > hi16);
    // 16 → 8
    simd<float, 8> lo8(mx16.select<8, 1>(0));
    simd<float, 8> hi8(mx16.select<8, 1>(8));
    simd<float, 8> mx8;
    mx8.merge(lo8, hi8, lo8 > hi8);
    // 8 → 4
    simd<float, 4> lo4(mx8.select<4, 1>(0));
    simd<float, 4> hi4(mx8.select<4, 1>(4));
    simd<float, 4> mx4;
    mx4.merge(lo4, hi4, lo4 > hi4);
    // 4 → 2
    simd<float, 2> lo2(mx4.select<2, 1>(0));
    simd<float, 2> hi2(mx4.select<2, 1>(2));
    simd<float, 2> mx2;
    mx2.merge(lo2, hi2, lo2 > hi2);
    // 2 → 1
    return mx2[0] > mx2[1] ? mx2[0] : mx2[1];
}

// Find first lane matching val in a simd<float, 32>, mask it out, return lane index
inline int find_and_mask32(simd<float, 32>& s, float val) {
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if ((float)s[i] == val) {
            s[i] = NEG_INF;
            return i;
        }
    }
    return 0;  // should not reach
}

// Fused sigmoid + bias + topk + renormalize kernel
// E=160, TOPK=16 hardcoded for MiniCPM5
// Each work-item processes one token
template<typename IT, int E = 160, int TOPK = 16>
sycl::event moe_fused_sigmoid_topk(
    sycl::queue& q,
    const IT* logits,          // [M, E] fp16/bf16
    const float* bias,         // [E] float32 (or nullptr if no correction bias)
    float* topk_weights,       // [M, TOPK] float32 output (renormalized)
    int* topk_ids,             // [M, TOPK] int32 output
    int M)
{
    static_assert(E == 160, "Only E=160 supported");
    static_assert(TOPK == 16, "Only TOPK=16 supported");
    constexpr int SW = 32;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(M),
            [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                const int tok = (int)id[0];
                const IT* row = logits + (size_t)tok * E;

                // 1. Load E=160 logits, convert to float32
                simd<float, SW> s0 = convert<float>(block_load<IT, SW>(row));
                simd<float, SW> s1 = convert<float>(block_load<IT, SW>(row + SW));
                simd<float, SW> s2 = convert<float>(block_load<IT, SW>(row + 2 * SW));
                simd<float, SW> s3 = convert<float>(block_load<IT, SW>(row + 3 * SW));
                simd<float, SW> s4 = convert<float>(block_load<IT, SW>(row + 4 * SW));

                // 2. Sigmoid: 1 / (1 + exp(-x))
                simd<float, SW> one(1.0f);
                s0 = one / (one + __ESIMD_NS::exp(-s0));
                s1 = one / (one + __ESIMD_NS::exp(-s1));
                s2 = one / (one + __ESIMD_NS::exp(-s2));
                s3 = one / (one + __ESIMD_NS::exp(-s3));
                s4 = one / (one + __ESIMD_NS::exp(-s4));

                // 3. Add correction bias if present
                if (bias != nullptr) {
                    s0 += block_load<float, SW>(bias);
                    s1 += block_load<float, SW>(bias + SW);
                    s2 += block_load<float, SW>(bias + 2 * SW);
                    s3 += block_load<float, SW>(bias + 3 * SW);
                    s4 += block_load<float, SW>(bias + 4 * SW);
                }

                // 4. Top-k selection: 16 rounds of find-max-and-mask
                simd<float, TOPK> out_vals;
                simd<int, TOPK> out_ids;

                #pragma unroll
                for (int k = 0; k < TOPK; k++) {
                    float m0 = hmax32(s0);
                    float m1 = hmax32(s1);
                    float m2 = hmax32(s2);
                    float m3 = hmax32(s3);
                    float m4 = hmax32(s4);

                    // Find global max and its segment
                    float mx = m0; int seg = 0;
                    if (m1 > mx) { mx = m1; seg = 1; }
                    if (m2 > mx) { mx = m2; seg = 2; }
                    if (m3 > mx) { mx = m3; seg = 3; }
                    if (m4 > mx) { mx = m4; seg = 4; }

                    // Find lane within winning segment and mask out
                    int lane = 0;
                    switch (seg) {
                        case 0: lane = find_and_mask32(s0, mx); break;
                        case 1: lane = find_and_mask32(s1, mx); break;
                        case 2: lane = find_and_mask32(s2, mx); break;
                        case 3: lane = find_and_mask32(s3, mx); break;
                        case 4: lane = find_and_mask32(s4, mx); break;
                    }

                    out_vals[k] = mx;
                    out_ids[k] = seg * SW + lane;
                }

                // 5. Renormalize: weights / sum(weights)
                float wsum = 0.0f;
                #pragma unroll
                for (int k = 0; k < TOPK; k++) wsum += out_vals[k];
                if (wsum > 0.0f) out_vals = out_vals / wsum;

                // 6. Store results
                block_store<float, TOPK>(topk_weights + (size_t)tok * TOPK, out_vals);
                block_store<int, TOPK>(topk_ids + (size_t)tok * TOPK, out_ids);
            });
    });
}
