/* gdn_update.h — ESIMD kernel for Gated Delta Network (GDN) state update.
 *
 * Used in Qwen3.5-4B hybrid model for linear attention layers.
 * Replaces both the Triton chunk_gated_delta_rule (which has NaN on XPU)
 * and the PyTorch sequential fallback (which is slow).
 *
 * Algorithm per (sequence, value_head) thread:
 *   For each token t in sequence:
 *     g = -exp(A_log) * softplus(a[t] + dt_bias)
 *     beta = sigmoid(b[t])
 *     q_norm = L2_normalize(q[t]) * scale
 *     k_norm = L2_normalize(k[t])
 *     For each V-row v_idx:
 *       h[K] = state[hv][v_idx][:]          // f32
 *       h *= exp(g)                          // decay
 *       recon = dot(h, k_norm)               // scalar
 *       v_upd = (v[t][v_idx] - recon) * beta
 *       h += v_upd * k_norm                  // rank-1 update
 *       o[v_idx] = dot(h, q_norm)            // output
 *       store h back
 *
 * Grid: nd_range<1>(N * HV, 1) — one ESIMD thread per (seq, value_head).
 * Requires doubleGRF (512 registers).
 */

#include "utils.h"

/* ---- ESIMD scalar math helpers ---- */
/* In ESIMD context, scalar C math (expf/logf/sqrtf) is forbidden.
 * Use simd<float,8> ESIMD intrinsics and extract lane 0. */
ESIMD_INLINE float esimd_expf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::exp(v);
    return v[0];
}
ESIMD_INLINE float esimd_logf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::log(v);
    return v[0];
}
ESIMD_INLINE float esimd_sqrtf(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::sqrt(v);
    return v[0];
}

/* ---- Helpers ---- */

/* Dot product of two f32 vectors of size 128 via tree reduction. */
ESIMD_INLINE float gdn_dot128(simd<float, 64> a_lo, simd<float, 64> a_hi,
                               simd<float, 64> b_lo, simd<float, 64> b_hi) {
    simd<float, 64> p_lo = a_lo * b_lo;
    simd<float, 64> p_hi = a_hi * b_hi;
    p_lo += p_hi;
    // Reduce 64 -> 1
    p_lo.select<32,1>(0) += p_lo.select<32,1>(32);
    p_lo.select<16,1>(0) += p_lo.select<16,1>(16);
    p_lo.select<8,1>(0) += p_lo.select<8,1>(8);
    p_lo.select<4,1>(0) += p_lo.select<4,1>(4);
    p_lo.select<2,1>(0) += p_lo.select<2,1>(2);
    return p_lo[0] + p_lo[1];
}

/* Load 64 bf16 values from ptr, return as f32.
 * bf16 -> f32 conversion: left-shift by 16 bits. */
ESIMD_INLINE simd<float, 64> gdn_load_bf16_64(const unsigned short* ptr) {
    simd<unsigned short, 64> raw = block_load<unsigned short, 64>(ptr);
    simd<unsigned int, 64> bits = raw;
    bits <<= 16;
    return bits.template bit_cast_view<float>();
}

/* Store 64 f32 values as bf16 (truncation to upper 16 bits). */
ESIMD_INLINE void gdn_store_f32_as_bf16_64(unsigned short* ptr, simd<float, 64> val) {
    simd<unsigned int, 64> bits = val.template bit_cast_view<unsigned int>();
    // Round-to-nearest-even: add (bit16 + 0x7FFF)
    simd<unsigned int, 64> rounding = ((bits >> 16) & 1u) + 0x7FFFu;
    bits += rounding;
    simd<unsigned short, 64> bf16_bits = (bits >> 16);
    block_store<unsigned short, 64>(ptr, bf16_bits);
}

/* Load single bf16 scalar from an aligned region around the target index. */
ESIMD_INLINE float gdn_load_bf16_scalar(const unsigned short* base, int64_t idx) {
    // Align load to 16-element (32-byte) boundary
    int64_t aligned = idx & ~15;
    int lane = (int)(idx & 15);
    simd<unsigned short, 16> chunk = block_load<unsigned short, 16>(base + aligned);
    unsigned short raw = chunk[lane];
    // bf16 -> f32: shift left 16 bits
    union { unsigned int u; float f; } conv;
    conv.u = (unsigned int)raw << 16;
    return conv.f;
}

/* ---- Main kernel ---- */

/* Tensor layouts (batch dim 1 is squeezed):
 *   A_log:         [HV]                    f32
 *   dt_bias:       [HV]                    f32
 *   a:             [T_total, HV]           bf16
 *   b:             [T_total, HV]           bf16
 *   q:             [T_total, H, K]         bf16  (H = query/key heads)
 *   k:             [T_total, H, K]         bf16
 *   v:             [T_total, HV, V]        bf16  (HV = value heads)
 *   state:         [num_states, HV, V, K]  f32
 *   output:        [T_total, HV, V]        bf16
 *   cu_seqlens:    [N+1]                   i32
 *   state_indices: [N]                     i32
 *
 * K = V = 128 for Qwen3.5-4B.
 */
ESIMD_INLINE void gdn_update_kernel(
    const float* __restrict__ A_log_ptr,
    const float* __restrict__ dt_bias_ptr,
    const unsigned short* __restrict__ a_ptr,       // bf16 as u16
    const unsigned short* __restrict__ b_ptr,       // bf16 as u16
    const unsigned short* __restrict__ q_ptr,       // bf16 as u16
    const unsigned short* __restrict__ k_ptr,       // bf16 as u16
    const unsigned short* __restrict__ v_ptr,       // bf16 as u16
    float* __restrict__ state_ptr,
    unsigned short* __restrict__ output_ptr,         // bf16 as u16
    const int* __restrict__ cu_seqlens_ptr,
    const int* __restrict__ state_indices_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    float attn_scale,
    int inplace_state,
    nd_item<1>& ndi)
{
    const int tid = ndi.get_global_id(0);
    const int seq_idx = tid / HV;
    const int hv = tid % HV;

    if (seq_idx >= N) return;

    // Load sequence bounds
    const int bos = cu_seqlens_ptr[seq_idx];
    const int eos = cu_seqlens_ptr[seq_idx + 1];
    if (eos <= bos) return;

    const int state_idx = state_indices_ptr[seq_idx];
    if (state_idx < 0) return;

    // Head mapping: value head -> query/key head (GQA-style grouping)
    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;

    // Pre-load model parameters for this value head (constant across tokens)
    const float A_log_val = A_log_ptr[hv];
    const float dt_bias_val = dt_bias_ptr[hv];
    const float neg_exp_A = -esimd_expf(A_log_val);  // -exp(A_log), reused each token

    // State base offset: state[state_idx, hv, :, :]
    // Layout: state[s, hv, v, k] at offset: s * HV * V * K + hv * V * K + v * K + k
    const int64_t state_hv_offset =
        (int64_t)state_idx * HV * gdn_V * gdn_K + (int64_t)hv * gdn_V * gdn_K;

    // Sequential loop over tokens
    for (int t = bos; t < eos; t++) {

        // ---- 1. Gating computation ----
        float a_val = gdn_load_bf16_scalar(a_ptr, (int64_t)t * HV + hv);
        float b_val = gdn_load_bf16_scalar(b_ptr, (int64_t)t * HV + hv);

        float x = a_val + dt_bias_val;
        // softplus(x) = log(1 + exp(x)), with overflow protection
        float sp = (x > 20.0f) ? x : esimd_logf(1.0f + esimd_expf(x));
        float g = neg_exp_A * sp;   // g = -exp(A_log) * softplus(a + dt_bias)
        float exp_g = esimd_expf(g); // decay factor, in (0, 1)
        float beta = 1.0f / (1.0f + esimd_expf(-b_val));  // sigmoid(b)

        // ---- 2. Load and L2-normalize q, k ----
        // q[t, i_h, 0:K] and k[t, i_h, 0:K], each 128 bf16
        const unsigned short* q_row = q_ptr + (int64_t)t * H * gdn_K + (int64_t)i_h * gdn_K;
        const unsigned short* k_row = k_ptr + (int64_t)t * H * gdn_K + (int64_t)i_h * gdn_K;

        simd<float, 64> q_lo = gdn_load_bf16_64(q_row);
        simd<float, 64> q_hi = gdn_load_bf16_64(q_row + 64);
        simd<float, 64> k_lo = gdn_load_bf16_64(k_row);
        simd<float, 64> k_hi = gdn_load_bf16_64(k_row + 64);

        // L2 norm: ||q||, ||k||
        float q_norm_sq = gdn_dot128(q_lo, q_hi, q_lo, q_hi);
        float k_norm_sq = gdn_dot128(k_lo, k_hi, k_lo, k_hi);
        float q_inv_norm = 1.0f / esimd_sqrtf(q_norm_sq + 1e-6f);
        float k_inv_norm = 1.0f / esimd_sqrtf(k_norm_sq + 1e-6f);

        // Normalize
        q_lo *= q_inv_norm; q_hi *= q_inv_norm;
        k_lo *= k_inv_norm; k_hi *= k_inv_norm;

        // Apply attention scale to q
        q_lo *= attn_scale; q_hi *= attn_scale;

        // ---- 3. Load v[t, hv, 0:V] ----
        const unsigned short* v_row = v_ptr + (int64_t)t * HV * gdn_V + (int64_t)hv * gdn_V;
        // v is 128 bf16 values — we'll index element-by-element in the V-row loop
        // Pre-load into f32 for the inner loop
        simd<float, 64> v_lo = gdn_load_bf16_64(v_row);
        simd<float, 64> v_hi = gdn_load_bf16_64(v_row + 64);

        // ---- 4. Process each V-row independently ----
        // For each v_idx in [0, gdn_V): load h_row[K], decay, delta update, output
        simd<float, 64> o_lo, o_hi;  // output accumulator (128 f32 split)

        for (int v_idx = 0; v_idx < gdn_V; v_idx++) {
            // State row pointer: state[state_idx, hv, v_idx, 0:K]
            float* state_row = state_ptr + state_hv_offset + (int64_t)v_idx * gdn_K;

            // Load state row (128 f32 = 512 bytes, 2 loads of 256 bytes)
            simd<float, 64> h_lo = block_load<float, 64>(state_row);
            simd<float, 64> h_hi = block_load<float, 64>(state_row + 64);

            // Decay: h *= exp(g)
            h_lo *= exp_g;
            h_hi *= exp_g;

            // Reconstruction: recon = dot(h, k_norm)
            float recon = gdn_dot128(h_lo, h_hi, k_lo, k_hi);

            // Get v[t, hv, v_idx] as f32
            float v_val = (v_idx < 64) ? v_lo[v_idx] : v_hi[v_idx - 64];

            // Delta update: v_upd = (v_val - recon) * beta
            float v_upd = (v_val - recon) * beta;

            // Rank-1 update: h += v_upd * k_norm
            h_lo += v_upd * k_lo;
            h_hi += v_upd * k_hi;

            // Output: o[v_idx] = dot(h, q_norm)
            float o_val = gdn_dot128(h_lo, h_hi, q_lo, q_hi);
            if (v_idx < 64) {
                o_lo[v_idx] = o_val;
            } else {
                o_hi[v_idx - 64] = o_val;
            }

            // Store state row back
            block_store<float, 64>(state_row, h_lo);
            block_store<float, 64>(state_row + 64, h_hi);
        }

        // ---- 5. Store output[t, hv, 0:V] as bf16 ----
        unsigned short* out_row = output_ptr + (int64_t)t * HV * gdn_V + (int64_t)hv * gdn_V;
        gdn_store_f32_as_bf16_64(out_row, o_lo);
        gdn_store_f32_as_bf16_64(out_row + 64, o_hi);
    }
}
