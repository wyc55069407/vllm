/* vllm-kernel-custom: Torch extension registration for LGRF (doubleGRF) ESIMD kernels.
 * Each kernel has its own named op with proper typed parameters.
 */
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "vllm_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(vllm_kernel_custom, m) {

  /* === Element-wise multiply (already named) === */
  m.def("esimd_mul_lgrf(Tensor a, Tensor b, Tensor c, int flag, int len) -> Tensor");
  m.impl("esimd_mul_lgrf", torch::kXPU, &esimd_kernel_mul_lgrf);

  /* === Full attention SDP (FP16, BF16, BF16io) — non-causal Flash Attention === */
  m.def("esimd_sdp_fp16(Tensor Q, Tensor K, Tensor V, Tensor norm_alpha, Tensor output, "
        "int q_len, int kv_len, int head_q, int head_kv) -> Tensor");
  m.impl("esimd_sdp_fp16", torch::kXPU, &esimd_sdp_fp16);

  m.def("esimd_sdp_bf16(Tensor Q, Tensor K, Tensor V, Tensor norm_alpha, Tensor output, "
        "int q_len, int kv_len, int head_q, int head_kv) -> Tensor");
  m.impl("esimd_sdp_bf16", torch::kXPU, &esimd_sdp_bf16);

  m.def("esimd_sdp_bf16io(Tensor Q, Tensor K, Tensor V, Tensor norm_alpha, Tensor output, "
        "int q_len, int kv_len, int head_q, int head_kv) -> Tensor");
  m.impl("esimd_sdp_bf16io", torch::kXPU, &esimd_sdp_bf16io);

  /* === SDP MLA with XMX (specific named op replacing esimd_kernel_uni_lgrf) === */
  m.def("esimd_sdp_mla_lgrf(Tensor q_extend, Tensor k_extend, Tensor v_extend, "
        "Tensor k_buffer, Tensor v_buffer, "
        "Tensor kv_indices, Tensor o_extend, "
        "int num_heads, int num_heads_kv, "
        "int extend_seq_len, int prefix_seq_len, "
        "int qk_dim, int v_dim, "
        "float attn_scale) -> Tensor");
  m.impl("esimd_sdp_mla_lgrf", torch::kXPU, &esimd_sdp_mla_lgrf);

  /* === GDN (Gated Delta Network) state update === */
  m.def("esimd_gdn_update(Tensor A_log, Tensor dt_bias, "
        "Tensor a, Tensor b, Tensor q, Tensor k, Tensor v, "
        "Tensor state, Tensor output, "
        "Tensor cu_seqlens, Tensor state_indices, "
        "int N, int H, int HV, int K, int V, "
        "float scale, int inplace_state) -> Tensor");
  m.impl("esimd_gdn_update", torch::kXPU, &esimd_gdn_update);

  /* === Paged SDP (decode + prefill with causal mask) === */
  m.def("esimd_sdp_paged(Tensor query, Tensor kv_cache, Tensor output, "
        "Tensor block_table, Tensor seq_lens, Tensor query_start_loc, "
        "int num_heads, int num_kv_heads, "
        "int head_dim, int block_size, "
        "int max_seq_len, float attn_scale, "
        "int causal) -> Tensor");
  m.impl("esimd_sdp_paged", torch::kXPU, &esimd_sdp_paged);

  /* === InfLLMv2 Sparse Paged SDP === */
  m.def("esimd_sdp_paged_sparse(Tensor query, Tensor kv_cache, Tensor output, "
        "Tensor block_table, Tensor seq_lens, Tensor query_start_loc, "
        "Tensor sparse_mask, Tensor sparse_mask_cnt, "
        "int num_heads, int num_kv_heads, "
        "int head_dim, int block_size, "
        "int max_seq_len, float attn_scale, "
        "int is_decode, int num_sparse_blocks) -> Tensor");
  m.impl("esimd_sdp_paged_sparse", torch::kXPU, &esimd_sdp_paged_sparse);

  /* === InfLLMv2 K Pooling === */
  m.def("esimd_infllmv2_k_pooling(Tensor key_cache, Tensor key_pooled, "
        "int num_kv_heads, int head_dim, "
        "int kv_len, int num_blocks, "
        "int kernel_size, int kernel_stride) -> Tensor");
  m.impl("esimd_infllmv2_k_pooling", torch::kXPU, &esimd_infllmv2_k_pooling);

  /* === InfLLMv2 Pattern Detection — Prefill === */
  m.def("esimd_infllmv2_pattern_prefill(Tensor query, Tensor key_pooled, "
        "Tensor block_scores, Tensor pooled_scores, Tensor topk_output, "
        "int num_heads, int num_kv_heads, "
        "int seq_len, int num_blocks, "
        "int head_dim, int num_pooled, "
        "int cache_len, int causal, "
        "int init_block, int local_block, "
        "int topk) -> Tensor");
  m.impl("esimd_infllmv2_pattern_prefill", torch::kXPU, &esimd_infllmv2_pattern_prefill);

  /* === InfLLMv2 Pattern Detection — Decode === */
  m.def("esimd_infllmv2_pattern_decode(Tensor query, Tensor key_pooled, "
        "Tensor block_scores, Tensor kv_block_scores, "
        "Tensor pooled_scores, Tensor topk_output, "
        "int num_heads, int num_kv_heads, "
        "int seq_len, int num_blocks, "
        "int head_dim, int num_pooled, "
        "int cache_len, int causal, "
        "int init_block, int local_block, "
        "int topk) -> Tensor");
  m.impl("esimd_infllmv2_pattern_decode", torch::kXPU, &esimd_infllmv2_pattern_decode);

  /* === InfLLMv2 Mask Convert === */
  m.def("esimd_infllmv2_mask_convert(Tensor mask_orig, Tensor mask_out, "
        "Tensor mask_cnt_out, "
        "int qlen, int num_kv_heads, "
        "int total_kv_blocks) -> Tensor");
  m.impl("esimd_infllmv2_mask_convert", torch::kXPU, &esimd_infllmv2_mask_convert);

  /* === InfLLMv2 Paged K Pooling === */
  m.def("esimd_infllmv2_k_pooling_paged(Tensor kv_cache, Tensor key_pooled, "
        "Tensor block_table, Tensor seq_lens, "
        "int num_kv_heads, int head_dim, "
        "int page_size, int num_pooled_blocks, "
        "int kernel_size, int kernel_stride, "
        "int start_pooled_block) -> Tensor");
  m.impl("esimd_infllmv2_k_pooling_paged", torch::kXPU, &esimd_infllmv2_k_pooling_paged);

  /* === InfLLMv2 Force Last Block === */
  m.def("esimd_infllmv2_force_last_block(Tensor sparse_mask, Tensor seq_lens, "
        "int num_kv_heads, int sparse_block_size) -> Tensor");
  m.impl("esimd_infllmv2_force_last_block", torch::kXPU, &esimd_infllmv2_force_last_block);
}

REGISTER_EXTENSION(common_ops_lgrf)
