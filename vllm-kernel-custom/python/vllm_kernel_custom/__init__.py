import ctypes
import os

import torch

from vllm_kernel_custom import common_ops
from vllm_kernel_custom import common_ops_lgrf

from vllm_kernel_custom.gemm import (
    awq_dequantize,
)
from vllm_kernel_custom.esimd_ops import (
    # Non-LGRF ops
    esimd_mul_scale_factor_and_add,
    esimd_gemv_fp8,
    esimd_fp8_dequant,
    esimd_bmm_gemv_fp8,
    esimd_sdpa_normal,
    esimd_sdpa_mla,
    esimd_cat_qk,
    esimd_rms_norm,
    esimd_rms_norm_qk,
    esimd_grouped_topk,
    esimd_grouped_topk_fused_gate,
    esimd_grouped_topk_kimi,
    esimd_grouped_topk_fused_gate_kimi,
    esimd_shared_expert,
    esimd_rope,
    esimd_update_kv,
    esimd_topk_sort,
    esimd_dsa_qk_attn,
    esimd_topk_multi_round,
    esimd_update_kv_dsa,
    esimd_rope_quant_updatek,
    esimd_weight_qscale_fuse,
    esimd_mla_mega,
    esimd_dsa_mega,
    # oneDNN ops
    onednn_w8a16_fp8,
    onednn_w8a16_fp8_block,
    onednn_w4a16_int4,
    # LGRF ops
    esimd_mul_lgrf,
    esimd_sdp_fp16,
    esimd_sdp_bf16,
    esimd_sdp_bf16io,
    esimd_sdp_mla_lgrf,
    esimd_sdp_paged,
    esimd_gdn_update,
    # InfLLMv2 ops (LGRF)
    esimd_sdp_paged_sparse,
    esimd_infllmv2_k_pooling,
    esimd_infllmv2_pattern_prefill,
    esimd_infllmv2_pattern_decode,
    esimd_infllmv2_mask_convert,
)

from vllm_kernel_custom.version import __version__

build_tree_kernel = (
    None  # TODO(ying): remove this after updating the sglang python code.
)
