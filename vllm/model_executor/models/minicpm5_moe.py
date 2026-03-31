# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniCPM5 MoE model.

Architecture: GQA with gated attention + DeepSeek V3-style MoE.
- Layer 0: dense MLP (intermediate_size=8192)
- Layers 1-27: MoE (160 routed experts, 16 experts/tok, 1 shared expert)
- Gated attention: q_proj doubled, sigmoid gate on attention output
- Sigmoid scoring with e_score_correction_bias
"""

import os
import time
import typing
from collections.abc import Callable, Iterable

# ESIMD MoE kernels (optional, enabled by MINICPM5_ESIMD_MOE=1)
_esimd_moe_decode = None
_esimd_moe_decode_ts = None
_esimd_moe_prefill = None
_esimd_moe_sigmoid_topk = None
if os.environ.get("MINICPM5_ESIMD_MOE", "0") == "1":
    try:
        from vllm_kernel_custom.esimd_ops import esimd_moe_decode as _esimd_moe_decode
        from vllm_kernel_custom.esimd_ops import esimd_moe_decode_ts as _esimd_moe_decode_ts
        from vllm_kernel_custom.esimd_ops import esimd_moe_prefill as _esimd_moe_prefill
        from vllm_kernel_custom.esimd_ops import esimd_moe_sigmoid_topk as _esimd_moe_sigmoid_topk
    except ImportError:
        pass

# ESIMD GEMV kernels for shared expert (enabled by MINICPM5_ESIMD_GEMV=1)
_esimd_w4a16_gemv = None
_esimd_w4a16_gate_up_silu = None
_esimd_rope_available = False
if os.environ.get("MINICPM5_ESIMD_GEMV", "0") == "1":
    try:
        from vllm_kernel_custom.esimd_ops import esimd_w4a16_gemv as _esimd_w4a16_gemv
        from vllm_kernel_custom.esimd_ops import esimd_w4a16_gate_up_silu as _esimd_w4a16_gate_up_silu
        # esimd_rope is available once vllm_kernel_custom is loaded
        import torch as _torch
        _torch.ops.vllm_kernel_custom.esimd_rope
        _esimd_rope_available = True
    except (ImportError, AttributeError):
        pass

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import GateLinear, SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class MiniCPM5MoEAttention(nn.Module):
    """GQA attention with gated output (sigmoid gate on attention output)."""

    def __init__(
        self,
        config,
        model_config=None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.use_gated_attention = getattr(config, "use_gated_attention", False)

        # QKVParallelLinear: when gated, Q heads doubled to carry gate
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (2 if self.use_gated_attention else 1),
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        if self.use_gated_attention:
            # q_gate has 2x Q size (query + gate interleaved per head)
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

        num_tokens = q.shape[0]
        if (
            _esimd_rope_available
            and num_tokens <= 8
            and self.head_dim in (64, 128)
        ):
            # ESIMD fused neox-style RoPE: 1 kernel instead of 8 elementwise
            rotary_emb = self.rotary_emb
            # Get cos_sin_cache (Phi3LongRoPE uses long_short_cos_sin_cache)
            if hasattr(rotary_emb, 'long_short_cos_sin_cache'):
                cos_sin_cache = rotary_emb.long_short_cos_sin_cache
                # Compute effective positions for long rope
                if getattr(rotary_emb, 'use_long_rope', False):
                    eff_positions = positions + rotary_emb.original_max_position_embeddings
                else:
                    eff_positions = positions
            else:
                cos_sin_cache = rotary_emb.cos_sin_cache
                eff_positions = positions
            cos_sin_cache = cos_sin_cache.to(q.dtype)
            # flag=2 means neox style, no offset
            hd = self.head_dim
            torch.ops.vllm_kernel_custom.esimd_rope(
                q, k, cos_sin_cache,
                eff_positions, eff_positions,  # positions_k unused (no offset)
                self.num_heads, hd, hd,        # q_heads, rope_dim, q_stride
                self.num_kv_heads, hd, hd,     # kv_heads, k_rope_dim, k_stride
                self.num_kv_heads * hd,         # k_nope_stride (stride between seq pos in K)
                num_tokens, 2,                  # seq_len, flag=2 (neox, no offset)
            )
            if not getattr(MiniCPM5MoEAttention, '_esimd_rope_logged', False):
                MiniCPM5MoEAttention._esimd_rope_logged = True
                logger.warning("ESIMD neox RoPE active: M=%d, heads=%d, hd=%d",
                               num_tokens, self.num_heads, hd)
        else:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if self.use_gated_attention:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


class MiniCPM5MoEMLP(nn.Module):
    """Dense MLP with SiluAndMul activation (for layer 0 and shared experts)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniCPM5MoEMoE(nn.Module):
    """DeepSeek V3-style MoE with sigmoid scoring and e_score_correction_bias."""

    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            prefix=f"{prefix}.gate",
        )
        # MiniCPM5 always uses sigmoid scoring with e_score_correction_bias
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        # Load balancing settings
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )

        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MiniCPM5MoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=1.0
            if not self.is_rocm_aiter_moe_enabled
            else self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts
            if self.is_fusion_moe_shared_experts_enabled
            else None,
        )

        # MiniCPM5 vendor reference always uses fp32 for router gate output.
        # is_monolithic=False for GPTQ would default to bf16, losing routing
        # precision. Force fp32 to match vendor behavior.
        self.gate.set_out_dtype(torch.float32)

    _moe_inspected = False
    _esimd_logged = False

    def _ensure_transposed_scales(self):
        """Create transposed scale copies for ESIMD decode (lazy, once)."""
        if hasattr(self, '_w13_scales_t'):
            return
        # w13_scales: [E, 2*N, K/GS] -> [E, K/GS, 2*N]
        w13_s = self.experts.w13_scales
        w2_s = self.experts.w2_scales
        self._w13_scales_t = w13_s.permute(0, 2, 1).contiguous()
        self._w2_scales_t = w2_s.permute(0, 2, 1).contiguous()
        logger.warning(
            "ESIMD MoE: created transposed scales — "
            "w13_scales_t %s, w2_scales_t %s",
            list(self._w13_scales_t.shape),
            list(self._w2_scales_t.shape),
        )

    def _esimd_routing(
        self, router_logits: torch.Tensor, num_tokens: int
    ) -> tuple:
        """Fused sigmoid+topk routing or PyTorch fallback."""
        topk = self.experts.moe_config.experts_per_token
        num_experts = router_logits.size(-1)

        if (_esimd_moe_sigmoid_topk is not None
                and num_experts == 160 and topk == 16):
            # Fused ESIMD: sigmoid + bias + topk + renorm in one kernel
            bias = (self.gate.e_score_correction_bias.float()
                    if self.gate.e_score_correction_bias is not None
                    else torch.empty(0, dtype=torch.float32,
                                     device=router_logits.device))
            topk_weights = torch.empty(num_tokens, topk,
                                       dtype=torch.float32,
                                       device=router_logits.device)
            topk_ids = torch.empty(num_tokens, topk,
                                   dtype=torch.int32,
                                   device=router_logits.device)
            _esimd_moe_sigmoid_topk(
                router_logits.half(), bias,
                topk_weights, topk_ids,
                num_experts, topk)
        else:
            # PyTorch fallback
            scores = torch.sigmoid(router_logits.float())
            if self.gate.e_score_correction_bias is not None:
                scores = scores + self.gate.e_score_correction_bias.unsqueeze(0)
            topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(torch.float32)

        return topk_weights, topk_ids

    def _esimd_shared_expert(
        self, hidden_states: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Shared expert: ESIMD fused gate_up_silu + GEMV down, or fallback."""
        if (
            _esimd_w4a16_gate_up_silu is not None
            and _esimd_w4a16_gemv is not None
            and num_tokens <= 8
            and hasattr(self.shared_experts.gate_up_proj, 'linear_weights')
        ):
            try:
                gate_up_lw = self.shared_experts.gate_up_proj.linear_weights
                down_lw = self.shared_experts.down_proj.linear_weights
                # Check if ESIMD scales exist (set by gptq.py)
                if not hasattr(gate_up_lw, 'esimd_scales_fp16'):
                    return self.shared_experts(hidden_states)

                dtype = hidden_states.dtype
                device = hidden_states.device
                gs = gate_up_lw.onednn_group_size

                # gate_up: [M, K] -> [M, N_half]
                gu_w = gate_up_lw.onednn_weight      # [2*N, K/2]
                gu_s = (gate_up_lw.esimd_scales_bf16
                        if dtype == torch.bfloat16
                        else gate_up_lw.esimd_scales_fp16)
                N_half = gu_w.size(0) // 2
                intermediate = torch.empty(
                    num_tokens, N_half, dtype=dtype, device=device)
                _esimd_w4a16_gate_up_silu(
                    hidden_states, gu_w, gu_s, intermediate, N_half, gs)

                # down: [M, N_half] -> [M, K]
                dn_w = down_lw.onednn_weight          # [K, N_half/2]
                dn_s = (down_lw.esimd_scales_bf16
                        if dtype == torch.bfloat16
                        else down_lw.esimd_scales_fp16)
                K_out = dn_w.size(0)
                shared_output = torch.empty(
                    num_tokens, K_out, dtype=dtype, device=device)
                _esimd_w4a16_gemv(
                    intermediate, dn_w, dn_s, shared_output, gs)
                return shared_output
            except Exception:
                pass
        return self.shared_experts(hidden_states)

    def _esimd_forward(
        self, hidden_states: torch.Tensor, num_tokens: int, hidden_dim: int
    ) -> torch.Tensor:
        """ESIMD MoE decode fast path: routing + ESIMD kernel + shared expert."""
        # 1. Routing: fused sigmoid+topk or PyTorch fallback
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = self._esimd_routing(router_logits, num_tokens)

        # 2. ESIMD MoE decode kernel (routed experts)
        group_size = self.experts.group_size
        output = torch.zeros_like(hidden_states)

        # Use original (non-transposed) scale layout — faster than transposed
        # (block_load on contiguous K/GS dim beats strided scalar loads)
        # Transposed variant available via MINICPM5_ESIMD_MOE_TS=1 for testing
        use_ts = (os.environ.get("MINICPM5_ESIMD_MOE_TS", "0") == "1"
                  and _esimd_moe_decode_ts is not None)
        if use_ts:
            self._ensure_transposed_scales()
            w13_scales_t = self._w13_scales_t
            w2_scales_t = self._w2_scales_t
            if w13_scales_t.dtype != hidden_states.dtype:
                w13_scales_t = w13_scales_t.to(hidden_states.dtype)
                w2_scales_t = w2_scales_t.to(hidden_states.dtype)

            _esimd_moe_decode_ts(
                hidden_states,
                self.experts.w13_qweight, w13_scales_t,
                self.experts.w2_qweight, w2_scales_t,
                topk_weights, topk_ids,
                output, group_size,
            )
        else:
            w13_scales = self.experts.w13_scales
            w2_scales = self.experts.w2_scales
            if w13_scales.dtype != hidden_states.dtype:
                w13_scales = w13_scales.to(hidden_states.dtype)
                w2_scales = w2_scales.to(hidden_states.dtype)

            _esimd_moe_decode(
                hidden_states,
                self.experts.w13_qweight, w13_scales,
                self.experts.w2_qweight, w2_scales,
                topk_weights, topk_ids,
                output, group_size,
            )

        final_hidden_states = output

        # 3. Shared expert (ESIMD fused or PyTorch fallback)
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self._esimd_shared_expert(
                hidden_states, num_tokens)

        # 4. Apply routed_scaling_factor (vendor: always scale routed output)
        final_hidden_states = final_hidden_states * self.routed_scaling_factor

        # 5. Add shared expert output
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if not MiniCPM5MoEMoE._esimd_logged:
            MiniCPM5MoEMoE._esimd_logged = True
            logger.warning(
                "ESIMD MoE decode: M=%d, K=%d, topk=%d, group_size=%d, "
                "transposed_scales=%s, fused_topk=%s",
                num_tokens, hidden_dim,
                self.experts.moe_config.experts_per_token, group_size,
                _esimd_moe_decode_ts is not None,
                _esimd_moe_sigmoid_topk is not None,
            )

        return final_hidden_states.view(num_tokens, hidden_dim)

    _esimd_prefill_logged = False

    def _esimd_prefill_forward(
        self, hidden_states: torch.Tensor, num_tokens: int, hidden_dim: int
    ) -> torch.Tensor:
        """ESIMD MoE prefill fast path: routing + ESIMD GGEMV/oneDNN + shared expert."""
        # 1. Routing: fused sigmoid+topk or PyTorch fallback
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = self._esimd_routing(router_logits, num_tokens)

        # 2. ESIMD MoE prefill kernel (requires fp16 — kernel is hardcoded sycl::half)
        group_size = self.experts.group_size
        orig_dtype = hidden_states.dtype
        need_fp16_convert = (orig_dtype != torch.float16)

        if need_fp16_convert:
            hidden_states_fp16 = hidden_states.to(torch.float16)
        else:
            hidden_states_fp16 = hidden_states
        output = torch.zeros_like(hidden_states_fp16)

        # Need both non-transposed (GGEMV) and transposed (oneDNN) scales
        self._ensure_transposed_scales()
        w13_scales = self.experts.w13_scales
        w2_scales = self.experts.w2_scales
        w13_scales_t = self._w13_scales_t
        w2_scales_t = self._w2_scales_t

        # Scales must be fp16 to match kernel expectation
        if w13_scales.dtype != torch.float16:
            w13_scales = w13_scales.to(torch.float16)
            w2_scales = w2_scales.to(torch.float16)
            w13_scales_t = w13_scales_t.to(torch.float16)
            w2_scales_t = w2_scales_t.to(torch.float16)

        _esimd_moe_prefill(
            hidden_states_fp16,
            self.experts.w13_qweight, w13_scales, w13_scales_t,
            self.experts.w2_qweight, w2_scales, w2_scales_t,
            topk_weights, topk_ids,
            output, group_size,
        )

        if need_fp16_convert:
            output = output.to(orig_dtype)

        final_hidden_states = output

        # 3. Shared expert
        shared_output = self._esimd_shared_expert(hidden_states, num_tokens)

        # 4. Apply routed_scaling_factor (vendor: always scale routed output)
        final_hidden_states = final_hidden_states * self.routed_scaling_factor

        # 5. Add shared expert output
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if not MiniCPM5MoEMoE._esimd_prefill_logged:
            MiniCPM5MoEMoE._esimd_prefill_logged = True
            logger.warning(
                "ESIMD MoE prefill: M=%d, K=%d, topk=%d, group_size=%d",
                num_tokens, hidden_dim,
                self.experts.moe_config.experts_per_token, group_size,
            )

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # === ESIMD MoE prefill fast path (M > 64) ===
        # Needs M*topk large enough for GGEMV efficiency; short prompts use Triton
        if (
            _esimd_moe_prefill is not None
            and num_tokens > 64
            and not self.is_sequence_parallel
            and self.tp_size == 1
        ):
            return self._esimd_prefill_forward(
                hidden_states, num_tokens, hidden_dim)

        # === ESIMD MoE decode fast path (M <= 8) ===
        if (
            _esimd_moe_decode is not None
            and num_tokens <= 8
            and not self.is_sequence_parallel
            and self.tp_size == 1
        ):
            return self._esimd_forward(hidden_states, num_tokens, hidden_dim)

        # === MOE INSPECT: dump shapes on first decode call ===
        do_inspect = (
            os.environ.get("MINICPM5_MOE_INSPECT") == "1"
            and not MiniCPM5MoEMoE._moe_inspected
            and num_tokens == 1  # decode step
        )

        if do_inspect:
            MiniCPM5MoEMoE._moe_inspected = True
            import sys
            P = lambda *a: print(*a, file=sys.stderr, flush=True)
            P("\n" + "=" * 80)
            P("  MOE INSPECT (layer 1, first decode step)")
            P("=" * 80)
            P(f"\n  INPUT hidden_states: {hidden_states.shape} {hidden_states.dtype}")
            P(f"    values[:8]: {hidden_states[0,:8].tolist()}")

            # Gate weights
            P(f"\n  GATE weight: {self.gate.weight.shape} {self.gate.weight.dtype}")
            if hasattr(self.gate, 'e_score_correction_bias'):
                P(f"  e_score_correction_bias: "
                  f"{self.gate.e_score_correction_bias.shape} "
                  f"{self.gate.e_score_correction_bias.dtype}")

        if self.experts.is_internal_router:
            fused_moe_out = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)

            if do_inspect:
                P(f"\n  ROUTER LOGITS: {router_logits.shape} {router_logits.dtype}")
                P(f"    min={router_logits.min().item():.4f} "
                  f"max={router_logits.max().item():.4f}")
                topk_vals, topk_idx = torch.topk(router_logits[0], 16)
                P(f"    top-16 indices: {topk_idx.tolist()}")
                P(f"    top-16 values:  {[f'{v:.4f}' for v in topk_vals.tolist()]}")

            fused_moe_out = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        shared_output, final_hidden_states = fused_moe_out

        if do_inspect:
            P(f"\n  ROUTED OUTPUT (before scaling): {final_hidden_states.shape} "
              f"{final_hidden_states.dtype}")
            P(f"    values[:8]: {final_hidden_states[0,:8].tolist()}")
            P(f"  routed_scaling_factor: {self.routed_scaling_factor}")
            if shared_output is not None:
                P(f"  SHARED OUTPUT: {shared_output.shape} {shared_output.dtype}")
                P(f"    values[:8]: {shared_output[0,:8].tolist()}")

            # Dump expert weights
            experts_layer = self.experts
            P(f"\n  --- EXPERT WEIGHTS ---")
            for attr in ['w13_qweight', 'w2_qweight', 'w13_scales', 'w2_scales',
                         'w13_qzeros', 'w2_qzeros']:
                w = getattr(experts_layer, attr, None)
                if w is None:
                    # Check if nested in quant_method
                    pass
                else:
                    P(f"  {attr}: {w.shape} {w.dtype}")

            # Check nested in experts module
            for name, param in experts_layer.named_parameters():
                if any(k in name for k in ['qweight', 'scales', 'qzeros',
                                            'weight', 'w1', 'w2', 'w13']):
                    P(f"  experts.{name}: {param.shape} {param.dtype}")

            # Shared expert weights
            if self.shared_experts is not None:
                P(f"\n  --- SHARED EXPERT WEIGHTS ---")
                for name, param in self.shared_experts.named_parameters():
                    P(f"  shared.{name}: {param.shape} {param.dtype}")

        if self.shared_experts is None:
            assert shared_output is None

        # Apply routed_scaling_factor (vendor: always scale routed output)
        if not self.is_rocm_aiter_moe_enabled:
            final_hidden_states *= self.routed_scaling_factor

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if do_inspect:
            P(f"\n  FINAL OUTPUT (after scaling + shared): "
              f"{final_hidden_states.shape} {final_hidden_states.dtype}")
            P(f"    values[:8]: {final_hidden_states[0,:8].tolist()}")
            P("=" * 80 + "\n")

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniCPM5MoEDecoderLayer(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_idx = extract_layer_index(prefix)

        self.self_attn = MiniCPM5MoEAttention(
            config,
            model_config=vllm_config.model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            config.n_routed_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = MiniCPM5MoEMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiniCPM5MoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    _profile_enabled = None  # class-level cache

    @classmethod
    def _is_profile_enabled(cls):
        if cls._profile_enabled is None:
            cls._profile_enabled = os.environ.get("MINICPM5_PROFILE", "0") == "1"
            if cls._profile_enabled:
                cls._profile_step = 0
                cls._profile_data = []  # list of (step, layer, attn_ms, mlp_ms, total_ms, n_tokens)
        return cls._profile_enabled

    _nan_check = os.environ.get("MINICPM5_NAN_CHECK", "0") == "1"
    _nan_step = 0

    @staticmethod
    def _check_tensor(name, t, layer_idx, step):
        """Lightweight NaN/inf check — samples first token only."""
        if t is None:
            return
        sample = t[0] if t.dim() > 1 else t
        has_nan = torch.isnan(sample).any().item()
        has_inf = torch.isinf(sample).any().item()
        if has_nan or has_inf:
            absmax = t.abs().max().item()
            print(f"[NAN_CHECK] step={step} layer={layer_idx} {name}: "
                  f"nan={has_nan} inf={has_inf} absmax={absmax} "
                  f"shape={list(t.shape)} dtype={t.dtype}", flush=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        do_profile = self._is_profile_enabled()
        do_nan = self._nan_check

        if do_profile:
            n_tokens = hidden_states.shape[0]
            torch.xpu.synchronize()
            t_start = time.perf_counter()

        if do_nan and self.layer_idx == 0:
            self._check_tensor("input", hidden_states, 0,
                               self.__class__._nan_step)

        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if do_profile:
            torch.xpu.synchronize()
            t_attn_start = time.perf_counter()

        hidden_states = self.self_attn(positions, hidden_states)

        if do_nan:
            self._check_tensor("post_attn", hidden_states, self.layer_idx,
                               self.__class__._nan_step)

        if do_profile:
            torch.xpu.synchronize()
            t_attn_end = time.perf_counter()

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        if do_profile:
            torch.xpu.synchronize()
            t_mlp_start = time.perf_counter()

        hidden_states = self.mlp(hidden_states)

        if do_nan:
            self._check_tensor("post_mlp", hidden_states, self.layer_idx,
                               self.__class__._nan_step)
            self._check_tensor("residual", residual, self.layer_idx,
                               self.__class__._nan_step)
            # Increment step counter after last layer
            if self.layer_idx == 27:
                self.__class__._nan_step += 1

        if do_profile:
            torch.xpu.synchronize()
            t_end = time.perf_counter()
            step = self.__class__._profile_step
            attn_ms = (t_attn_end - t_attn_start) * 1000
            mlp_ms = (t_end - t_mlp_start) * 1000
            total_ms = (t_end - t_start) * 1000
            self.__class__._profile_data.append(
                (step, self.layer_idx, attn_ms, mlp_ms, total_ms, n_tokens)
            )
            # Increment step counter after last layer
            if self.layer_idx == 27:
                self.__class__._profile_step += 1

        return hidden_states, residual


@support_torch_compile
class MiniCPM5MoEModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniCPM5MoEDecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        from itertools import islice
        import torch as _torch

        is_prefill = hidden_states.shape[0] > 1
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
            # Flush GPU command queue each layer during prefill to avoid
            # xe driver job_timeout (default 5s on BMG). Without this,
            # async kernel submissions accumulate and total GPU time
            # exceeds the timeout, causing DEVICE_LOST.
            if is_prefill and hasattr(_torch, "xpu"):
                _torch.xpu.synchronize()

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiniCPM5MoEForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = MiniCPM5MoEModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        # Write profiling data to temp file for external analysis
        if MiniCPM5MoEDecoderLayer._is_profile_enabled():
            data = MiniCPM5MoEDecoderLayer._profile_data
            if len(data) >= 28 and len(data) % 28 == 0:
                import json as _json
                profile_path = os.environ.get(
                    "MINICPM5_PROFILE_PATH", "/tmp/minicpm5_profile.jsonl")
                with open(profile_path, "w") as f:
                    for row in data:
                        f.write(_json.dumps(row) + "\n")

        return hidden_states

    @staticmethod
    def _dump_profile_summary(data, total_steps):
        """Print profiling summary grouped by prefill vs decode."""
        import collections
        import sys

        if not data:
            return

        by_step = collections.defaultdict(list)
        for step, layer_idx, attn_ms, mlp_ms, total_ms, n_tokens in data:
            by_step[step].append((layer_idx, attn_ms, mlp_ms, total_ms, n_tokens))

        steps = sorted(by_step.keys())
        if not steps:
            return

        prefill_steps = []
        decode_steps = []
        for s in steps:
            n_tok = by_step[s][0][4]
            if n_tok > 1:
                prefill_steps.append(s)
            else:
                decode_steps.append(s)

        print(f"\n{'='*80}", file=sys.stderr)
        print(f"  MINICPM5 PROFILE: {len(prefill_steps)} prefill, "
              f"{len(decode_steps)} decode steps", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)

        for phase, step_list in [("PREFILL", prefill_steps),
                                  ("DECODE", decode_steps)]:
            if not step_list:
                continue

            attn_total = 0.0
            mlp_total = 0.0
            all_total = 0.0
            n_tokens_first = by_step[step_list[0]][0][4]

            attn_per_layer = collections.defaultdict(float)
            mlp_per_layer = collections.defaultdict(float)

            for s in step_list:
                for layer_idx, attn_ms, mlp_ms, total_ms, n_tokens in by_step[s]:
                    attn_total += attn_ms
                    mlp_total += mlp_ms
                    all_total += total_ms
                    attn_per_layer[layer_idx] += attn_ms
                    mlp_per_layer[layer_idx] += mlp_ms

            n_steps = len(step_list)
            other_total = all_total - attn_total - mlp_total

            print(f"\n  --- {phase} ({n_steps} steps, {n_tokens_first} tok/step) ---",
                  file=sys.stderr)
            print(f"  Total: {all_total:.1f} ms", file=sys.stderr)
            print(f"    Attention: {attn_total:.1f} ms "
                  f"({attn_total/all_total*100:.1f}%)", file=sys.stderr)
            print(f"    MLP/MoE:   {mlp_total:.1f} ms "
                  f"({mlp_total/all_total*100:.1f}%)", file=sys.stderr)
            print(f"    Other:     {other_total:.1f} ms "
                  f"({other_total/all_total*100:.1f}%)", file=sys.stderr)

            if n_steps > 0:
                per_step = all_total / n_steps
                attn_per = attn_total / n_steps
                mlp_per = mlp_total / n_steps
                other_per = other_total / n_steps
                print(f"  Per step: {per_step:.2f} ms = {attn_per:.2f} attn"
                      f" + {mlp_per:.2f} mlp + {other_per:.2f} other",
                      file=sys.stderr)

            for li in [0, 1, 14, 27]:
                ltype = "dense" if li == 0 else "MoE"
                a = attn_per_layer[li] / n_steps if n_steps else 0
                m = mlp_per_layer[li] / n_steps if n_steps else 0
                print(f"    L{li:2d}({ltype}): attn={a:.2f} mlp={m:.2f} ms",
                      file=sys.stderr)

        print(f"{'='*80}\n", file=sys.stderr)
        sys.stderr.flush()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip expert weights (handled below)
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models
                if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    # Skip loading extra bias for GPTQ models
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            if name is not None:
                loaded_params.add(name)

        return loaded_params
