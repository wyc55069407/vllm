# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ESIMD attention backend for Intel XPU (BMG).

Uses SYCL-ESIMD paged SDP kernel from vllm-kernel-custom.
Supports decode and prefill with causal masking, paged KV cache,
and GQA. Compatible with NHD KV cache layout.
"""
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class EsimdAttentionBackend(AttentionBackend):
    """ESIMD attention backend for Intel XPU with paged KV cache."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
        torch.float16,
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 256]

    @staticmethod
    def get_name() -> str:
        return "ESIMD_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["EsimdAttentionImpl"]:
        return EsimdAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["EsimdAttentionMetadataBuilder"]:
        return EsimdAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # NHD layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class EsimdAttentionMetadata:
    """Metadata for ESIMD paged SDP attention."""
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor   # [batch+1] i32
    max_seq_len: int
    seq_lens: torch.Tensor           # [batch] i32
    block_table: torch.Tensor        # [batch, max_blocks_per_seq] i32
    slot_mapping: torch.Tensor       # [num_tokens] i32
    causal: bool = True


class EsimdAttentionMetadataBuilder(
    AttentionMetadataBuilder[EsimdAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> EsimdAttentionMetadata:
        return EsimdAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
        )


class EsimdAttentionImpl(AttentionImpl):
    """ESIMD paged SDP attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        if alibi_slopes is not None:
            logger.warning("ESIMD_ATTN does not support ALiBi slopes")
        if sliding_window is not None:
            logger.warning("ESIMD_ATTN does not support sliding window")
        if logits_soft_cap is not None:
            logger.warning("ESIMD_ATTN does not support logits soft cap")

        # Lazy import to avoid import errors when not on XPU
        self._esimd_sdp_paged = None

    def _get_kernel(self):
        if self._esimd_sdp_paged is None:
            try:
                from vllm_kernel_custom import esimd_sdp_paged
                self._esimd_sdp_paged = esimd_sdp_paged
            except ImportError:
                raise RuntimeError(
                    "ESIMD_ATTN backend requires vllm-kernel-custom with "
                    "esimd_sdp_paged support. Build with: "
                    "cd vllm-kernel-custom && python setup_sycl.py install"
                )
        return self._esimd_sdp_paged

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: EsimdAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using ESIMD paged SDP kernel.

        Args:
            query: [num_tokens, num_heads, head_size] bf16/fp16
            key: [num_tokens, num_kv_heads, head_size] bf16/fp16
            value: [num_tokens, num_kv_heads, head_size] bf16/fp16
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: ESIMD attention metadata
            output: [num_tokens, num_heads, head_size] bf16/fp16

        Returns:
            output: [num_tokens, num_heads * head_size] bf16/fp16
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        esimd_sdp_paged = self._get_kernel()

        # Determine block_size from KV cache shape
        block_size = kv_cache.shape[2]

        # ESIMD kernel natively supports both bf16 and fp16
        q_slice = query[:num_actual_tokens]
        o_slice = output[:num_actual_tokens]

        # Call ESIMD paged SDP kernel (stride-aware, handles any layout)
        esimd_sdp_paged(
            q_slice,
            kv_cache,
            o_slice,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            block_size,
            attn_metadata.max_seq_len,
            self.scale,
            1 if attn_metadata.causal else 0,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY,
                              AttentionType.ENCODER):
            return
        # NHD layout: kv_cache[2, num_blocks, block_size, num_kv_heads, head]
        # unbind on dim=0 to get key_cache and value_cache
        key_cache, value_cache = kv_cache.unbind(0)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
