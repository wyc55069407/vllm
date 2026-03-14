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
}

REGISTER_EXTENSION(common_ops_lgrf)
