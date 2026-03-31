/* vllm-kernel-custom: Named op declarations for all ESIMD kernels.
 * Replaces the generic esimd_kernel_uni dispatch with specific functions.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include <torch/torch.h>

/* ========== Non-LGRF ops (common_ops) ========== */

// AWQ dequantize (already named)
torch::Tensor awq_dequantize(torch::Tensor qweight, torch::Tensor scales, torch::Tensor qzeros);

// Fused residual add: c = a_bf16 * factor + b_fp16 (already named)
at::Tensor esimd_add(at::Tensor a, at::Tensor b, at::Tensor c, int64_t len, double factor);

// FP8 weight GEMV: output = input @ weight_fp8 (dequantized with scale) + bias
at::Tensor esimd_gemv_fp8(
    at::Tensor input, at::Tensor weight, at::Tensor weight_scale,
    at::Tensor bias, at::Tensor output,
    int64_t M, int64_t N, int64_t K, int64_t batch,
    int64_t scale_block_n, int64_t scale_block_k, int64_t has_bias);

// FP8 weight dequantize
at::Tensor esimd_fp8_dequant(
    at::Tensor weight_fp8, at::Tensor scale, at::Tensor output,
    int64_t N, int64_t K, int64_t scale_block_n, int64_t scale_block_k);

// Batch MM with FP8 weights
at::Tensor esimd_bmm_gemv_fp8(
    at::Tensor input, at::Tensor weight, at::Tensor output,
    int64_t M, int64_t N, int64_t K, int64_t stride, int64_t heads,
    double scale);

// SDPA normal with reduce
at::Tensor esimd_sdpa_normal(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor kv_indices, at::Tensor output,
    at::Tensor tmp_softmax, at::Tensor tmp_out, at::Tensor tmp_reduce,
    int64_t num_heads, int64_t seq_len, int64_t kv_len,
    int64_t head_dim, int64_t num_kv_heads,
    double attn_scale, double kv_scale);

// SDPA MLA mega (all-in-one MLA attention)
at::Tensor esimd_sdpa_mla(
    at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor kv_indices, at::Tensor output,
    at::Tensor tmp_softmax, at::Tensor tmp_out, at::Tensor tmp_reduce,
    at::Tensor tmp_extra0, at::Tensor tmp_extra1,
    int64_t num_heads, int64_t seq_len, int64_t kv_len,
    int64_t head_dim, int64_t num_kv_heads,
    int64_t qk_dim, int64_t v_dim, int64_t extra_param,
    double attn_scale, double kv_scale);

// Concatenate Q/K tensors for MLA
at::Tensor esimd_cat_qk(
    at::Tensor q_nope, at::Tensor q_pe,
    at::Tensor k_nope, at::Tensor k_pe,
    at::Tensor q_out, at::Tensor k_out,
    int64_t seq_len, int64_t q_heads, int64_t kv_heads,
    int64_t qk_head_dim, int64_t stride);

// RMSNorm with residual add
at::Tensor esimd_rms_norm(
    at::Tensor weight, at::Tensor residual, at::Tensor input, at::Tensor output,
    int64_t hidden_size, int64_t seq_len, int64_t add_residual, int64_t flag,
    double eps);

// RMSNorm for Q/K separately
at::Tensor esimd_rms_norm_qk(
    at::Tensor q_weight, at::Tensor kv_weight,
    at::Tensor q_input, at::Tensor kv_input,
    at::Tensor q_output, at::Tensor kv_output,
    int64_t q_dim, int64_t kv_dim, int64_t seq_len, int64_t stride,
    double q_eps, double kv_eps);

// Grouped Top-K
at::Tensor esimd_grouped_topk(
    at::Tensor scores, at::Tensor output_vals, at::Tensor output_ids, at::Tensor tmp,
    int64_t batch, int64_t experts, int64_t topk, int64_t groups, int64_t group_topk,
    double score_scale);

// Grouped Top-K fused with gate
at::Tensor esimd_grouped_topk_fused_gate(
    at::Tensor input, at::Tensor gate_weight, at::Tensor output_vals,
    at::Tensor output_ids, at::Tensor tmp0, at::Tensor tmp1, at::Tensor tmp2,
    int64_t batch, int64_t experts, int64_t topk, int64_t groups, int64_t group_topk,
    double score_scale);

// Grouped Top-K (Kimi variant)
at::Tensor esimd_grouped_topk_kimi(
    at::Tensor scores, at::Tensor output_vals, at::Tensor output_ids, at::Tensor tmp,
    int64_t batch, int64_t experts, int64_t topk, int64_t groups, int64_t group_topk,
    double score_scale);

// Grouped Top-K fused with gate (Kimi variant)
at::Tensor esimd_grouped_topk_fused_gate_kimi(
    at::Tensor input, at::Tensor gate_weight, at::Tensor output_vals,
    at::Tensor output_ids, at::Tensor tmp0, at::Tensor tmp1, at::Tensor tmp2,
    int64_t batch, int64_t experts, int64_t topk, int64_t groups, int64_t group_topk,
    double score_scale);

// Shared expert mega kernel
at::Tensor esimd_shared_expert(
    at::Tensor input, at::Tensor gate_weight, at::Tensor up_weight,
    at::Tensor down_weight, at::Tensor output, at::Tensor tmp0, at::Tensor tmp1,
    int64_t M, int64_t N, int64_t K);

// Rotary position embedding (dynamic shape)
at::Tensor esimd_rope(
    at::Tensor q, at::Tensor k, at::Tensor cos_sin_cache,
    at::Tensor positions, at::Tensor positions_k,
    int64_t q_heads, int64_t rope_dim, int64_t q_stride,
    int64_t kv_heads, int64_t k_rope_dim, int64_t k_stride,
    int64_t k_nope_stride, int64_t seq_len, int64_t flag);

// Update KV cache with indices
at::Tensor esimd_update_kv(
    at::Tensor kv_cache, at::Tensor new_kv, at::Tensor indices,
    int64_t seq_len, int64_t head_dim, int64_t num_heads);

// Top-K sort
at::Tensor esimd_topk_sort(
    at::Tensor input0, at::Tensor input1, at::Tensor input2,
    at::Tensor input3, at::Tensor input4, at::Tensor input5,
    at::Tensor input6, at::Tensor input7, at::Tensor output,
    int64_t param0, int64_t param1);

// DSA QK attention
at::Tensor esimd_dsa_qk_attn(
    at::Tensor q, at::Tensor k, at::Tensor output, at::Tensor indices, at::Tensor tmp,
    int64_t seq_len, int64_t head_dim, int64_t num_heads);

// Top-K multi round
at::Tensor esimd_topk_multi_round(
    at::Tensor input0, at::Tensor input1, at::Tensor input2,
    at::Tensor input3, at::Tensor input4, at::Tensor input5,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3, int64_t p4);

// Update KV with DSA index
at::Tensor esimd_update_kv_dsa(
    at::Tensor kv, at::Tensor new_kv, at::Tensor indices,
    at::Tensor tmp0, at::Tensor tmp1,
    int64_t param);

// RoPE + quantize + update K
at::Tensor esimd_rope_quant_updatek(
    at::Tensor t0, at::Tensor t1, at::Tensor t2, at::Tensor t3, at::Tensor t4,
    at::Tensor t5, at::Tensor t6, at::Tensor t7, at::Tensor t8, at::Tensor t9,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3,
    double f0);

// Weight quantization scale fuse
at::Tensor esimd_weight_qscale_fuse(
    at::Tensor weight, at::Tensor scale, at::Tensor output, at::Tensor tmp,
    int64_t param, double scale_factor);

// MLA mega (fused MLA pipeline)
at::Tensor esimd_mla_mega(
    at::Tensor input_norm_weight, at::Tensor lora_a_weight, at::Tensor lora_a_weight_scale,
    at::Tensor lora_a_norm_q_weight, at::Tensor lora_a_norm_kv_weight,
    at::Tensor lora_b_q_weight, at::Tensor lora_b_q_weight_scale,
    at::Tensor w_kc_fp8, at::Tensor cos_sin_cache, at::Tensor positions,
    at::Tensor lora_a_bias, at::Tensor lora_b_bias,
    at::Tensor hidden_states, at::Tensor residual,
    at::Tensor q_out, at::Tensor k_out, at::Tensor k_nope, at::Tensor intermedia,
    int64_t hidden_size, int64_t seq_len, int64_t add_residual,
    int64_t lora_a_dim, int64_t block_n, int64_t block_k,
    int64_t lora_a_has_bias, int64_t lora_b_has_bias,
    int64_t q_lora_rank, int64_t kv_lora_rank,
    int64_t q_header_num, int64_t kv_header_num,
    int64_t qk_head_dim, int64_t qk_rope_head_dim,
    double input_norm_eps, double q_a_norm_eps, double kv_a_norm_eps, double bmm_w_scale);

// DSA mega (fused DSA pipeline)
at::Tensor esimd_dsa_mega(
    at::Tensor x, at::Tensor q_lora, at::Tensor w_qb_weight, at::Tensor w_qb_scale,
    at::Tensor wk_weight, at::Tensor wk_scale,
    at::Tensor query, at::Tensor key,
    at::Tensor knorm_weight, at::Tensor knorm_bias,
    at::Tensor cos_sin_cache, at::Tensor positions,
    at::Tensor k_cache, at::Tensor k_scale,
    at::Tensor query_out, at::Tensor q_scale,
    at::Tensor weights, at::Tensor weights_proj_weight,
    at::Tensor topk_indices, at::Tensor index_score_rsv,
    at::Tensor out_idx, at::Tensor out_ordered,
    at::Tensor kv_indptr, at::Tensor kv_indices,
    at::Tensor kv_indptr_updated, at::Tensor kv_indices_new,
    int64_t batch_num, int64_t max_q_len,
    int64_t tbo_start_batch_idx, int64_t seq_lens_cpu_addr,
    int64_t tokens_per_batch, int64_t total_count_stride,
    int64_t real_total_count_reserved, int64_t groups,
    double knorm_eps, double softmax_scale);

/* ========== MoE decode ESIMD ops (common_ops) ========== */

// Fused MoE decode: up_gate_silu + down + gather (W4A16 GPTQ INT4 symmetric)
at::Tensor esimd_moe_decode(
    at::Tensor x, at::Tensor w13_qweight, at::Tensor w13_scales,
    at::Tensor w2_qweight, at::Tensor w2_scales,
    at::Tensor topk_weights, at::Tensor topk_ids,
    at::Tensor output, int64_t group_size);

// Same but with transposed scales: w13_scales_t [E, K/GS, 2*N], w2_scales_t [E, N/GS, K]
at::Tensor esimd_moe_decode_ts(
    at::Tensor x, at::Tensor w13_qweight, at::Tensor w13_scales_t,
    at::Tensor w2_qweight, at::Tensor w2_scales_t,
    at::Tensor topk_weights, at::Tensor topk_ids,
    at::Tensor output, int64_t group_size);

/* ========== W4A16 ESIMD GEMV (non-MoE, common_ops) ========== */

// General W4A16 GEMV: y[M,N] = dequant(W[N,K/2]) @ x[M,K]
at::Tensor esimd_w4a16_gemv(
    at::Tensor x, at::Tensor weight, at::Tensor scales,
    at::Tensor output, int64_t group_size);

// Fused gate+up+SiLU: y[M,N] = SiLU(W_gate @ x) * (W_up @ x)
// weight [2*N, K/2], scales [2*N, K/GS]
at::Tensor esimd_w4a16_gate_up_silu(
    at::Tensor x, at::Tensor weight, at::Tensor scales,
    at::Tensor output, int64_t N_half, int64_t group_size);

// FP16/BF16 GEMV: y[M,N] = weight[N,K] @ x[M,K]^T (unquantized)
at::Tensor esimd_fp16_gemv(
    at::Tensor x, at::Tensor weight, at::Tensor output);

/* ========== Fused sigmoid+topk for MoE routing ========== */

void esimd_moe_sigmoid_topk(
    at::Tensor logits, at::Tensor bias,
    at::Tensor topk_weights, at::Tensor topk_ids,
    int64_t num_experts, int64_t topk);

/* ========== MoE prefill ops ========== */

// MoE prefill master dispatch (common_ops, standard GRF)
at::Tensor esimd_moe_prefill(
    at::Tensor x, at::Tensor w13_qweight, at::Tensor w13_scales,
    at::Tensor w13_scales_t, at::Tensor w2_qweight, at::Tensor w2_scales,
    at::Tensor w2_scales_t, at::Tensor topk_weights, at::Tensor topk_ids,
    at::Tensor output, int64_t group_size);

// MoE prefill GGEMV sub-op (common_ops_lgrf, doubleGRF)
at::Tensor esimd_moe_prefill_ggemv(
    at::Tensor expert_states, at::Tensor w13_qweight, at::Tensor w13_scales,
    at::Tensor w2_qweight, at::Tensor w2_scales,
    at::Tensor gate_buf, at::Tensor intermediate, at::Tensor expert_output,
    at::Tensor chunks, int64_t hidden_size, int64_t intermediate_size);

/* ========== oneDNN ops (common_ops) ========== */

// oneDNN FP8 GEMM: FP16/BF16 x FP8_E4M3, per-N scale
at::Tensor onednn_w8a16_fp8(
    at::Tensor x, at::Tensor weight, at::Tensor scales,
    at::Tensor bias, at::Tensor output,
    int64_t M, int64_t N, int64_t K, int64_t has_bias);

// oneDNN FP8 GEMM: FP16/BF16 x FP8_E4M3, block scale [K/block_k, N/block_n]
at::Tensor onednn_w8a16_fp8_block(
    at::Tensor x, at::Tensor weight, at::Tensor scales,
    at::Tensor bias, at::Tensor output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_k, int64_t block_n, int64_t has_bias);

// oneDNN INT4 W4A16 GEMM (GPTQ-style group-quantized)
at::Tensor onednn_w4a16_int4(
    at::Tensor x, at::Tensor weight, at::Tensor scales,
    at::Tensor zp, at::Tensor bias, at::Tensor output,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size, int64_t has_zp, int64_t has_bias);

/* ========== LGRF ops (common_ops_lgrf) ========== */

// Element-wise multiply (already named)
at::Tensor esimd_kernel_mul_lgrf(at::Tensor a, at::Tensor b, at::Tensor c, int64_t flag, int64_t len);

// Full attention SDP (FP16, BF16, BF16io) — non-causal Flash Attention
at::Tensor esimd_sdp_fp16(
    at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor norm_alpha, at::Tensor output,
    int64_t q_len, int64_t kv_len, int64_t head_q, int64_t head_kv);
at::Tensor esimd_sdp_bf16(
    at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor norm_alpha, at::Tensor output,
    int64_t q_len, int64_t kv_len, int64_t head_q, int64_t head_kv);
at::Tensor esimd_sdp_bf16io(
    at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor norm_alpha, at::Tensor output,
    int64_t q_len, int64_t kv_len, int64_t head_q, int64_t head_kv);

// SDP MLA with XMX (large GRF)
at::Tensor esimd_sdp_mla_lgrf(
    at::Tensor q_extend, at::Tensor k_extend, at::Tensor v_extend,
    at::Tensor k_buffer, at::Tensor v_buffer,
    at::Tensor kv_indices, at::Tensor o_extend,
    int64_t num_heads, int64_t num_heads_kv,
    int64_t extend_seq_len, int64_t prefix_seq_len,
    int64_t qk_dim, int64_t v_dim,
    double attn_scale);

// Paged SDP attention — decode + prefill with causal masking (large GRF)
at::Tensor esimd_sdp_paged(
    at::Tensor query, at::Tensor kv_cache, at::Tensor output,
    at::Tensor block_table, at::Tensor seq_lens, at::Tensor query_start_loc,
    int64_t num_heads, int64_t num_kv_heads,
    int64_t head_dim, int64_t block_size,
    int64_t max_seq_len, double attn_scale,
    int64_t causal);

// InfLLMv2 sparse paged SDP (large GRF)
at::Tensor esimd_sdp_paged_sparse(
    at::Tensor query, at::Tensor kv_cache, at::Tensor output,
    at::Tensor block_table, at::Tensor seq_lens, at::Tensor query_start_loc,
    at::Tensor sparse_mask, at::Tensor sparse_mask_cnt,
    int64_t num_heads, int64_t num_kv_heads,
    int64_t head_dim, int64_t block_size,
    int64_t max_seq_len, double attn_scale,
    int64_t is_decode, int64_t num_sparse_blocks);

// InfLLMv2 K pooling (large GRF)
at::Tensor esimd_infllmv2_k_pooling(
    at::Tensor key_cache, at::Tensor key_pooled,
    int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_len, int64_t num_blocks,
    int64_t kernel_size, int64_t kernel_stride);

// InfLLMv2 pattern detection — prefill (large GRF)
at::Tensor esimd_infllmv2_pattern_prefill(
    at::Tensor query, at::Tensor key_pooled,
    at::Tensor block_scores, at::Tensor pooled_scores, at::Tensor topk_output,
    int64_t num_heads, int64_t num_kv_heads,
    int64_t seq_len, int64_t num_blocks,
    int64_t head_dim, int64_t num_pooled,
    int64_t cache_len, int64_t causal,
    int64_t init_block, int64_t local_block,
    int64_t topk);

// InfLLMv2 pattern detection — decode (large GRF)
at::Tensor esimd_infllmv2_pattern_decode(
    at::Tensor query, at::Tensor key_pooled,
    at::Tensor block_scores, at::Tensor kv_block_scores,
    at::Tensor pooled_scores, at::Tensor topk_output,
    int64_t num_heads, int64_t num_kv_heads,
    int64_t seq_len, int64_t num_blocks,
    int64_t head_dim, int64_t num_pooled,
    int64_t cache_len, int64_t causal,
    int64_t init_block, int64_t local_block,
    int64_t topk);

// InfLLMv2 mask convert (large GRF)
at::Tensor esimd_infllmv2_mask_convert(
    at::Tensor mask_orig, at::Tensor mask_out, at::Tensor mask_cnt_out,
    int64_t qlen, int64_t num_kv_heads, int64_t total_kv_blocks);

// InfLLMv2 paged K pooling (large GRF)
at::Tensor esimd_infllmv2_k_pooling_paged(
    at::Tensor kv_cache, at::Tensor key_pooled,
    at::Tensor block_table, at::Tensor seq_lens,
    int64_t num_kv_heads, int64_t head_dim,
    int64_t page_size, int64_t num_pooled_blocks,
    int64_t kernel_size, int64_t kernel_stride,
    int64_t start_pooled_block);

// InfLLMv2 force last block insertion (large GRF)
at::Tensor esimd_infllmv2_force_last_block(
    at::Tensor sparse_mask, at::Tensor seq_lens,
    int64_t num_kv_heads, int64_t sparse_block_size);

// GDN (Gated Delta Network) state update (large GRF)
at::Tensor esimd_gdn_update(
    at::Tensor A_log, at::Tensor dt_bias,
    at::Tensor a, at::Tensor b, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor state, at::Tensor output,
    at::Tensor cu_seqlens, at::Tensor state_indices,
    int64_t N, int64_t H, int64_t HV,
    int64_t K, int64_t V,
    double scale, int64_t inplace_state);
