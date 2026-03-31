/* vllm-kernel-custom: Torch extension registration for non-LGRF ESIMD kernels.
 * Each kernel has its own named op with proper typed parameters.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "vllm_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(vllm_kernel_custom, m) {

  /* === Already-named ops === */
  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");
  m.impl("awq_dequantize", torch::kXPU, &awq_dequantize);

  m.def("esimd_add(Tensor a, Tensor b, Tensor c, int len, float factor) -> Tensor");
  m.impl("esimd_add", torch::kXPU, &esimd_add);

  /* === Specific named ops (replacing esimd_kernel_uni dispatch) === */

  m.def("esimd_gemv_fp8(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor bias, Tensor output, "
        "int M, int N, int K, int batch, "
        "int scale_block_n, int scale_block_k, int has_bias) -> Tensor");
  m.impl("esimd_gemv_fp8", torch::kXPU, &esimd_gemv_fp8);

  m.def("esimd_fp8_dequant(Tensor weight_fp8, Tensor scale, Tensor output, "
        "int N, int K, int scale_block_n, int scale_block_k) -> Tensor");
  m.impl("esimd_fp8_dequant", torch::kXPU, &esimd_fp8_dequant);

  m.def("esimd_bmm_gemv_fp8(Tensor input, Tensor weight, Tensor output, "
        "int M, int N, int K, int stride, int heads, float scale) -> Tensor");
  m.impl("esimd_bmm_gemv_fp8", torch::kXPU, &esimd_bmm_gemv_fp8);

  m.def("esimd_sdpa_normal(Tensor q, Tensor k, Tensor v, "
        "Tensor kv_indices, Tensor output, "
        "Tensor tmp_softmax, Tensor tmp_out, Tensor tmp_reduce, "
        "int num_heads, int seq_len, int kv_len, int head_dim, int num_kv_heads, "
        "float attn_scale, float kv_scale) -> Tensor");
  m.impl("esimd_sdpa_normal", torch::kXPU, &esimd_sdpa_normal);

  m.def("esimd_sdpa_mla(Tensor q, Tensor k, Tensor v, "
        "Tensor kv_indices, Tensor output, "
        "Tensor tmp_softmax, Tensor tmp_out, Tensor tmp_reduce, "
        "Tensor tmp_extra0, Tensor tmp_extra1, "
        "int num_heads, int seq_len, int kv_len, int head_dim, int num_kv_heads, "
        "int qk_dim, int v_dim, int extra_param, "
        "float attn_scale, float kv_scale) -> Tensor");
  m.impl("esimd_sdpa_mla", torch::kXPU, &esimd_sdpa_mla);

  m.def("esimd_cat_qk(Tensor q_nope, Tensor q_pe, Tensor k_nope, Tensor k_pe, "
        "Tensor q_out, Tensor k_out, "
        "int seq_len, int q_heads, int kv_heads, int qk_head_dim, int stride) -> Tensor");
  m.impl("esimd_cat_qk", torch::kXPU, &esimd_cat_qk);

  m.def("esimd_rms_norm(Tensor weight, Tensor residual, Tensor input, Tensor output, "
        "int hidden_size, int seq_len, int add_residual, int flag, float eps) -> Tensor");
  m.impl("esimd_rms_norm", torch::kXPU, &esimd_rms_norm);

  m.def("esimd_rms_norm_qk(Tensor q_weight, Tensor kv_weight, "
        "Tensor q_input, Tensor kv_input, Tensor q_output, Tensor kv_output, "
        "int q_dim, int kv_dim, int seq_len, int stride, "
        "float q_eps, float kv_eps) -> Tensor");
  m.impl("esimd_rms_norm_qk", torch::kXPU, &esimd_rms_norm_qk);

  m.def("esimd_grouped_topk(Tensor scores, Tensor output_vals, Tensor output_ids, Tensor tmp, "
        "int batch, int experts, int topk, int groups, int group_topk, "
        "float score_scale) -> Tensor");
  m.impl("esimd_grouped_topk", torch::kXPU, &esimd_grouped_topk);

  m.def("esimd_grouped_topk_fused_gate(Tensor input, Tensor gate_weight, "
        "Tensor output_vals, Tensor output_ids, Tensor tmp0, Tensor tmp1, Tensor tmp2, "
        "int batch, int experts, int topk, int groups, int group_topk, "
        "float score_scale) -> Tensor");
  m.impl("esimd_grouped_topk_fused_gate", torch::kXPU, &esimd_grouped_topk_fused_gate);

  m.def("esimd_grouped_topk_kimi(Tensor scores, Tensor output_vals, Tensor output_ids, Tensor tmp, "
        "int batch, int experts, int topk, int groups, int group_topk, "
        "float score_scale) -> Tensor");
  m.impl("esimd_grouped_topk_kimi", torch::kXPU, &esimd_grouped_topk_kimi);

  m.def("esimd_grouped_topk_fused_gate_kimi(Tensor input, Tensor gate_weight, "
        "Tensor output_vals, Tensor output_ids, Tensor tmp0, Tensor tmp1, Tensor tmp2, "
        "int batch, int experts, int topk, int groups, int group_topk, "
        "float score_scale) -> Tensor");
  m.impl("esimd_grouped_topk_fused_gate_kimi", torch::kXPU, &esimd_grouped_topk_fused_gate_kimi);

  m.def("esimd_shared_expert(Tensor input, Tensor gate_weight, Tensor up_weight, "
        "Tensor down_weight, Tensor output, Tensor tmp0, Tensor tmp1, "
        "int M, int N, int K) -> Tensor");
  m.impl("esimd_shared_expert", torch::kXPU, &esimd_shared_expert);

  m.def("esimd_rope(Tensor q, Tensor k, Tensor cos_sin_cache, "
        "Tensor positions, Tensor positions_k, "
        "int q_heads, int rope_dim, int q_stride, "
        "int kv_heads, int k_rope_dim, int k_stride, "
        "int k_nope_stride, int seq_len, int flag) -> Tensor");
  m.impl("esimd_rope", torch::kXPU, &esimd_rope);

  m.def("esimd_update_kv(Tensor kv_cache, Tensor new_kv, Tensor indices, "
        "int seq_len, int head_dim, int num_heads) -> Tensor");
  m.impl("esimd_update_kv", torch::kXPU, &esimd_update_kv);

  m.def("esimd_topk_sort(Tensor i0, Tensor i1, Tensor i2, Tensor i3, "
        "Tensor i4, Tensor i5, Tensor i6, Tensor i7, Tensor output, "
        "int param0, int param1) -> Tensor");
  m.impl("esimd_topk_sort", torch::kXPU, &esimd_topk_sort);

  m.def("esimd_dsa_qk_attn(Tensor q, Tensor k, Tensor output, Tensor indices, Tensor tmp, "
        "int seq_len, int head_dim, int num_heads) -> Tensor");
  m.impl("esimd_dsa_qk_attn", torch::kXPU, &esimd_dsa_qk_attn);

  m.def("esimd_topk_multi_round(Tensor i0, Tensor i1, Tensor i2, "
        "Tensor i3, Tensor i4, Tensor i5, "
        "int p0, int p1, int p2, int p3, int p4) -> Tensor");
  m.impl("esimd_topk_multi_round", torch::kXPU, &esimd_topk_multi_round);

  m.def("esimd_update_kv_dsa(Tensor kv, Tensor new_kv, Tensor indices, "
        "Tensor tmp0, Tensor tmp1, int param) -> Tensor");
  m.impl("esimd_update_kv_dsa", torch::kXPU, &esimd_update_kv_dsa);

  m.def("esimd_rope_quant_updatek(Tensor t0, Tensor t1, Tensor t2, Tensor t3, Tensor t4, "
        "Tensor t5, Tensor t6, Tensor t7, Tensor t8, Tensor t9, "
        "int p0, int p1, int p2, int p3, float f0) -> Tensor");
  m.impl("esimd_rope_quant_updatek", torch::kXPU, &esimd_rope_quant_updatek);

  m.def("esimd_weight_qscale_fuse(Tensor weight, Tensor scale, Tensor output, Tensor tmp, "
        "int param, float scale_factor) -> Tensor");
  m.impl("esimd_weight_qscale_fuse", torch::kXPU, &esimd_weight_qscale_fuse);

  m.def("esimd_mla_mega(Tensor input_norm_weight, Tensor lora_a_weight, Tensor lora_a_weight_scale, "
        "Tensor lora_a_norm_q_weight, Tensor lora_a_norm_kv_weight, "
        "Tensor lora_b_q_weight, Tensor lora_b_q_weight_scale, "
        "Tensor w_kc_fp8, Tensor cos_sin_cache, Tensor positions, "
        "Tensor lora_a_bias, Tensor lora_b_bias, "
        "Tensor hidden_states, Tensor residual, "
        "Tensor q_out, Tensor k_out, Tensor k_nope, Tensor intermedia, "
        "int hidden_size, int seq_len, int add_residual, "
        "int lora_a_dim, int block_n, int block_k, "
        "int lora_a_has_bias, int lora_b_has_bias, "
        "int q_lora_rank, int kv_lora_rank, "
        "int q_header_num, int kv_header_num, "
        "int qk_head_dim, int qk_rope_head_dim, "
        "float input_norm_eps, float q_a_norm_eps, float kv_a_norm_eps, float bmm_w_scale) -> Tensor");
  m.impl("esimd_mla_mega", torch::kXPU, &esimd_mla_mega);

  /* === Fused sigmoid+topk for MoE routing === */
  m.def("esimd_moe_sigmoid_topk(Tensor logits, Tensor bias, "
        "Tensor topk_weights, Tensor topk_ids, "
        "int num_experts, int topk) -> ()");
  m.impl("esimd_moe_sigmoid_topk", torch::kXPU, &esimd_moe_sigmoid_topk);

  /* === W4A16 ESIMD GEMV (non-MoE) === */
  m.def("esimd_w4a16_gemv(Tensor x, Tensor weight, Tensor scales, "
        "Tensor output, int group_size) -> Tensor");
  m.impl("esimd_w4a16_gemv", torch::kXPU, &esimd_w4a16_gemv);

  m.def("esimd_w4a16_gate_up_silu(Tensor x, Tensor weight, Tensor scales, "
        "Tensor output, int N_half, int group_size) -> Tensor");
  m.impl("esimd_w4a16_gate_up_silu", torch::kXPU, &esimd_w4a16_gate_up_silu);

  /* === MoE decode fused ESIMD === */
  m.def("esimd_moe_decode(Tensor x, Tensor w13_qweight, Tensor w13_scales, "
        "Tensor w2_qweight, Tensor w2_scales, "
        "Tensor topk_weights, Tensor topk_ids, "
        "Tensor output, int group_size) -> Tensor");
  m.impl("esimd_moe_decode", torch::kXPU, &esimd_moe_decode);

  /* === MoE decode fused ESIMD (transposed scales) === */
  m.def("esimd_moe_decode_ts(Tensor x, Tensor w13_qweight, Tensor w13_scales_t, "
        "Tensor w2_qweight, Tensor w2_scales_t, "
        "Tensor topk_weights, Tensor topk_ids, "
        "Tensor output, int group_size) -> Tensor");
  m.impl("esimd_moe_decode_ts", torch::kXPU, &esimd_moe_decode_ts);

  /* === MoE prefill master dispatch === */
  m.def("esimd_moe_prefill(Tensor x, Tensor w13_qweight, Tensor w13_scales, "
        "Tensor w13_scales_t, Tensor w2_qweight, Tensor w2_scales, "
        "Tensor w2_scales_t, Tensor topk_weights, Tensor topk_ids, "
        "Tensor output, int group_size) -> Tensor");
  m.impl("esimd_moe_prefill", torch::kXPU, &esimd_moe_prefill);

  /* === oneDNN FP8 GEMM === */
  m.def("onednn_w8a16_fp8(Tensor x, Tensor weight, Tensor scales, "
        "Tensor bias, Tensor output, "
        "int M, int N, int K, int has_bias) -> Tensor");
  m.impl("onednn_w8a16_fp8", torch::kXPU, &onednn_w8a16_fp8);

  m.def("onednn_w8a16_fp8_block(Tensor x, Tensor weight, Tensor scales, "
        "Tensor bias, Tensor output, "
        "int M, int N, int K, int block_k, int block_n, int has_bias) -> Tensor");
  m.impl("onednn_w8a16_fp8_block", torch::kXPU, &onednn_w8a16_fp8_block);

  /* === oneDNN INT4 W4A16 GEMM (GPTQ) === */
  m.def("onednn_w4a16_int4(Tensor x, Tensor weight, Tensor scales, "
        "Tensor zp, Tensor bias, Tensor output, "
        "int M, int N, int K, int group_size, int has_zp, int has_bias) -> Tensor");
  m.impl("onednn_w4a16_int4", torch::kXPU, &onednn_w4a16_int4);

  m.def("esimd_dsa_mega(Tensor x, Tensor q_lora, Tensor w_qb_weight, Tensor w_qb_scale, "
        "Tensor wk_weight, Tensor wk_scale, Tensor query, Tensor key, "
        "Tensor knorm_weight, Tensor knorm_bias, Tensor cos_sin_cache, Tensor positions, "
        "Tensor k_cache, Tensor k_scale, Tensor query_out, Tensor q_scale, "
        "Tensor weights, Tensor weights_proj_weight, "
        "Tensor topk_indices, Tensor index_score_rsv, "
        "Tensor out_idx, Tensor out_ordered, "
        "Tensor kv_indptr, Tensor kv_indices, "
        "Tensor kv_indptr_updated, Tensor kv_indices_new, "
        "int batch_num, int max_q_len, "
        "int tbo_start_batch_idx, int seq_lens_cpu_addr, "
        "int tokens_per_batch, int total_count_stride, "
        "int real_total_count_reserved, int groups, "
        "float knorm_eps, float softmax_scale) -> Tensor");
  m.impl("esimd_dsa_mega", torch::kXPU, &esimd_dsa_mega);
}

REGISTER_EXTENSION(common_ops)
