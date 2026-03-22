# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InfLLMv2 sparse attention backend for Intel XPU (BMG).

Uses ESIMD kernels for:
- Dense paged SDP when seq_len <= dense_len (8192)
- Sparse paged SDP (decode + prefill) when seq_len > dense_len

Sparse attention flow:
  Decode: K_new → pool_buf → pattern_detect → topk → sparse_decode_SDP
  Prefill: K_all → k_pooling → pattern_detect → topk → mask_convert → sparse_prefill_SDP
"""
from dataclasses import dataclass, field
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

# InfLLMv2 constants (MiniCPM4-8B config)
INFLLMV2_KERNEL_SIZE = 32       # k_pooling window size
INFLLMV2_KERNEL_STRIDE = 16    # k_pooling stride
INFLLMV2_SPARSE_BLOCK = 64     # tokens per sparse block
INFLLMV2_TOPK = 64             # number of KV blocks selected per token
INFLLMV2_INIT_BLOCKS = 2       # initial blocks always attended
INFLLMV2_LOCAL_BLOCKS = 4      # local attention window (in pooled blocks)
INFLLMV2_DENSE_LEN = 8192      # threshold: use dense below this


class InfLLMv2EsimdAttentionBackend(AttentionBackend):
    """InfLLMv2 sparse attention backend for Intel XPU."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
        torch.float16,
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "INFLLMV2_ESIMD_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["InfLLMv2EsimdAttentionImpl"]:
        return InfLLMv2EsimdAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["InfLLMv2EsimdAttentionMetadataBuilder"]:
        return InfLLMv2EsimdAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class InfLLMv2EsimdAttentionMetadata:
    """Metadata for InfLLMv2 sparse attention."""
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class InfLLMv2EsimdAttentionMetadataBuilder(
    AttentionMetadataBuilder[InfLLMv2EsimdAttentionMetadata]
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
    ) -> InfLLMv2EsimdAttentionMetadata:
        return InfLLMv2EsimdAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
        )


class InfLLMv2EsimdAttentionImpl(AttentionImpl):
    """InfLLMv2 sparse attention with ESIMD kernels."""

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

        # InfLLMv2 parameters
        self.dense_len = INFLLMV2_DENSE_LEN
        self.kernel_size = INFLLMV2_KERNEL_SIZE
        self.kernel_stride = INFLLMV2_KERNEL_STRIDE
        self.sparse_block = INFLLMV2_SPARSE_BLOCK
        self.topk = INFLLMV2_TOPK
        self.init_blocks = INFLLMV2_INIT_BLOCKS
        self.local_blocks = INFLLMV2_LOCAL_BLOCKS

        # Per-layer compressed K state (lazily initialized)
        self._pooled_k: dict[int, torch.Tensor] = {}  # req_idx -> pooled_k
        self._pooled_k_len: dict[int, int] = {}        # req_idx -> num_blocks

        # Lazy kernel imports
        self._kernels_loaded = False
        self._esimd_sdp_paged = None
        self._esimd_sdp_paged_sparse = None
        self._esimd_k_pooling = None
        self._esimd_pattern_prefill = None
        self._esimd_pattern_decode = None
        self._esimd_mask_convert = None

    def _load_kernels(self):
        if self._kernels_loaded:
            return
        try:
            from vllm_kernel_custom import (
                esimd_sdp_paged,
                esimd_sdp_paged_sparse,
                esimd_infllmv2_k_pooling,
                esimd_infllmv2_pattern_prefill,
                esimd_infllmv2_pattern_decode,
                esimd_infllmv2_mask_convert,
            )
            self._esimd_sdp_paged = esimd_sdp_paged
            self._esimd_sdp_paged_sparse = esimd_sdp_paged_sparse
            self._esimd_k_pooling = esimd_infllmv2_k_pooling
            self._esimd_pattern_prefill = esimd_infllmv2_pattern_prefill
            self._esimd_pattern_decode = esimd_infllmv2_pattern_decode
            self._esimd_mask_convert = esimd_infllmv2_mask_convert
            self._kernels_loaded = True
        except ImportError:
            raise RuntimeError(
                "INFLLMV2_ESIMD_ATTN backend requires vllm-kernel-custom "
                "with InfLLMv2 kernels. Build with: "
                "cd vllm-kernel-custom && python setup_sycl.py install"
            )

    def _forward_dense(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: InfLLMv2EsimdAttentionMetadata,
    ) -> torch.Tensor:
        """Dense paged SDP for short sequences."""
        block_size = kv_cache.shape[2]
        self._esimd_sdp_paged(
            query, kv_cache, output,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            self.num_heads, self.num_kv_heads,
            self.head_size, block_size,
            attn_metadata.max_seq_len, self.scale,
            1 if attn_metadata.causal else 0,
        )
        return output

    def _forward_sparse_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: InfLLMv2EsimdAttentionMetadata,
    ) -> torch.Tensor:
        """Sparse decode: k_pool → pattern_detect → topk → sparse SDP."""
        batch = attn_metadata.seq_lens.shape[0]
        block_size = kv_cache.shape[2]
        device = query.device
        nkvh = self.num_kv_heads
        nh = self.num_heads
        hd = self.head_size

        seq_lens_cpu = attn_metadata.seq_lens.cpu()
        max_sl = int(seq_lens_cpu.max().item())

        # Dimensions for pattern detection
        num_pooled_blocks = max(
            1, (max_sl - self.kernel_size + self.kernel_stride)
            // self.kernel_stride)
        pooling_stride = self.sparse_block // self.kernel_stride  # 4
        num_pooled = max(
            1, (num_pooled_blocks + pooling_stride - 1) // pooling_stride)

        # Step 1: Extract K from paged cache into contiguous buffer
        # K cache: kv_cache[0] shape [num_blocks, block_size, nkvh, hd]
        k_contiguous = torch.zeros(
            batch, nkvh, max_sl, hd,
            dtype=torch.float16, device=device)

        bt_cpu = attn_metadata.block_table.cpu()
        key_cache = kv_cache[0]
        for b in range(batch):
            sl = int(seq_lens_cpu[b].item())
            num_pages = (sl + block_size - 1) // block_size
            for p in range(num_pages):
                phys_page = int(bt_cpu[b, p].item())
                start = p * block_size
                end = min(start + block_size, sl)
                length = end - start
                k_page = key_cache[phys_page, :length]
                k_contiguous[b, :, start:end, :] = k_page.permute(
                    1, 0, 2).half()

        # Step 2: K pooling
        k_pooled = torch.zeros(
            batch, nkvh, num_pooled_blocks, hd,
            dtype=torch.float16, device=device)

        self._esimd_k_pooling(
            k_contiguous, k_pooled,
            nkvh, hd,
            max_sl, num_pooled_blocks,
            self.kernel_size, self.kernel_stride)

        # Step 3: Pattern detection (decode)
        # query shape: [num_tokens, nh, hd] where num_tokens = batch (1 per req)
        q_for_pattern = query.view(batch, nh, hd).unsqueeze(2)  # [B, nh, 1, hd]
        if q_for_pattern.dtype != torch.float16:
            q_for_pattern = q_for_pattern.half()

        block_scores = torch.zeros(
            batch, nh, 1, num_pooled_blocks,
            dtype=torch.float16, device=device)
        kv_block_scores = torch.zeros(
            batch, nkvh, 1, num_pooled_blocks,
            dtype=torch.float16, device=device)
        pooled_scores = torch.zeros(
            batch, nkvh, 1, num_pooled,
            dtype=torch.float16, device=device)
        topk_output = torch.zeros(
            batch, nkvh, 1, self.topk,
            dtype=torch.int32, device=device)

        cache_len = max_sl - 1  # history before this token

        self._esimd_pattern_decode(
            q_for_pattern, k_pooled,
            block_scores, kv_block_scores,
            pooled_scores, topk_output,
            nh, nkvh,
            1, num_pooled_blocks,  # seq_len=1, num_blocks
            hd, num_pooled,
            cache_len, 1,  # causal=True
            self.init_blocks, self.local_blocks,
            self.topk)

        # Step 4: Sparse decode SDP
        # topk_output: [batch, nkvh, 1, 64] — these are kv_block indices
        # Reshape to [batch, nkvh, 64] for the kernel
        sparse_mask = topk_output.squeeze(2).int()

        # For sparse decode, we pass num_sparse_blocks to the kernel
        # mask_cnt is not used for decode (only for prefill)
        dummy_mask_cnt = torch.zeros(1, dtype=torch.int32, device=device)

        self._esimd_sdp_paged_sparse(
            query, kv_cache, output,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            sparse_mask, dummy_mask_cnt,
            nh, nkvh,
            hd, block_size,
            max_sl, self.scale,
            1,  # is_decode=True
            self.topk)

        return output

    def _forward_sparse_prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: InfLLMv2EsimdAttentionMetadata,
    ) -> torch.Tensor:
        """Sparse prefill: k_pooling → pattern_detect → mask_convert → sparse SDP."""
        batch = attn_metadata.seq_lens.shape[0]
        block_size = kv_cache.shape[2]
        device = query.device
        dtype = query.dtype
        q_len = query.shape[0]

        seq_lens_cpu = attn_metadata.seq_lens.cpu()
        max_sl = int(seq_lens_cpu.max().item())

        # Step 1: Extract K from paged cache for pooling
        # K cache: kv_cache[0, :, :, :, :]
        # For pooling, we need contiguous [bsz, nkvh, kv_len, hd] layout
        # Extract from paged format using block_table
        # TODO: Direct paged K pooling kernel would be more efficient

        # For now, gather K into contiguous buffer
        nkvh = self.num_kv_heads
        hd = self.head_size

        # Contiguous K buffer for pooling
        k_contiguous = torch.zeros(
            batch, nkvh, max_sl, hd,
            dtype=torch.float16, device=device)

        bt_cpu = attn_metadata.block_table.cpu()
        key_cache = kv_cache[0]  # [num_blocks, block_size, nkvh, hd]

        for b in range(batch):
            sl = int(seq_lens_cpu[b].item())
            num_pages = (sl + block_size - 1) // block_size
            for p in range(num_pages):
                phys_page = int(bt_cpu[b, p].item())
                start = p * block_size
                end = min(start + block_size, sl)
                length = end - start
                # key_cache[phys_page, :length, :, :] -> k_contiguous[b, :, start:end, :]
                # key_cache shape: [num_blocks, block_size, nkvh, hd]
                k_page = key_cache[phys_page, :length]  # [length, nkvh, hd]
                k_contiguous[b, :, start:end, :] = k_page.permute(1, 0, 2)

        # Convert to fp16 if needed (pooling kernels expect fp16)
        if k_contiguous.dtype != torch.float16:
            k_contiguous = k_contiguous.half()

        # Step 2: K pooling
        num_pooled_blocks = (max_sl - self.kernel_size + self.kernel_stride) // self.kernel_stride
        if num_pooled_blocks <= 0:
            num_pooled_blocks = 1

        k_pooled = torch.zeros(
            batch, nkvh, num_pooled_blocks, hd,
            dtype=torch.float16, device=device)

        self._esimd_k_pooling(
            k_contiguous, k_pooled,
            nkvh, hd,
            max_sl, num_pooled_blocks,
            self.kernel_size, self.kernel_stride)

        # Step 3: Pattern detection (prefill)
        # Query needs to be in [bsz, nh, seq_len, hd] format for pattern detection
        nh = self.num_heads
        q_for_pattern = query.view(q_len, nh, hd).permute(1, 0, 2).unsqueeze(0)
        # [1, nh, q_len, hd]
        if q_for_pattern.dtype != torch.float16:
            q_for_pattern = q_for_pattern.half()

        num_kv_blocks = (max_sl + self.sparse_block - 1) // self.sparse_block
        pooling_stride = self.sparse_block // self.kernel_stride
        num_pooled_out = (num_pooled_blocks + pooling_stride - 1) // pooling_stride

        block_scores = torch.zeros(
            1, nkvh, q_len, num_pooled_blocks,
            dtype=torch.float16, device=device)
        pooled_scores = torch.zeros(
            1, nkvh, q_len, num_pooled_out,
            dtype=torch.float16, device=device)
        topk_per_token = torch.zeros(
            1, nkvh, q_len, self.topk,
            dtype=torch.int32, device=device)

        history_len = max_sl - q_len
        cache_len = history_len

        self._esimd_pattern_prefill(
            q_for_pattern, k_pooled,
            block_scores, pooled_scores, topk_per_token,
            nh, nkvh,
            q_len, num_pooled_blocks,
            hd, num_pooled_out,
            cache_len, 1,  # causal=True
            self.init_blocks, self.local_blocks,
            self.topk)

        # Step 4: Mask convert (per-token → per-q-block union)
        # topk_per_token: [1, nkvh, q_len, 64] -> reshape to [nkvh, q_len, 64]
        mask_orig = topk_per_token.squeeze(0).int()
        q_blocks = (q_len + 15) // 16
        mask_out = torch.zeros(
            nkvh, q_blocks, 1024,
            dtype=torch.int32, device=device)
        mask_cnt_out = torch.zeros(
            nkvh, q_blocks,
            dtype=torch.int32, device=device)

        total_kv_blocks = (max_sl + self.sparse_block - 1) // self.sparse_block
        if total_kv_blocks == 0:
            total_kv_blocks = 1

        self._esimd_mask_convert(
            mask_orig, mask_out, mask_cnt_out,
            q_len, nkvh, total_kv_blocks)

        # Step 5: Sparse prefill SDP
        sparse_mask_cnt = mask_cnt_out.int()
        sparse_mask = mask_out.int()

        # Debug: check mask stats
        mask_cnt_cpu = mask_cnt_out.cpu()
        logger.info(
            "Sparse prefill mask stats: q_blocks=%d, "
            "mask_cnt min=%d max=%d mean=%.1f, "
            "topk_min=%d topk_max=%d, "
            "num_pooled_blocks=%d, num_pooled_out=%d, "
            "total_kv_blocks=%d",
            q_blocks,
            int(mask_cnt_cpu.min().item()),
            int(mask_cnt_cpu.max().item()),
            float(mask_cnt_cpu.float().mean().item()),
            int(topk_per_token.min().item()),
            int(topk_per_token.max().item()),
            num_pooled_blocks, num_pooled_out,
            total_kv_blocks)

        # query_start_loc for single request
        qsl = torch.tensor([0, q_len], dtype=torch.int32, device=device)

        self._esimd_sdp_paged_sparse(
            query[:q_len], kv_cache, output[:q_len],
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            qsl,
            sparse_mask, sparse_mask_cnt,
            nh, nkvh,
            hd, block_size,
            max_sl, self.scale,
            0,  # is_decode=False
            self.topk)

        # Debug: compare sparse (all-blocks mask) vs dense output
        if not hasattr(self, '_debug_compared'):
            self._debug_compared = True

            sparse_out = output[:q_len].clone()

            # Test with ALL blocks mask to isolate kernel vs mask issue
            all_blocks_mask = torch.zeros(
                nkvh, q_blocks, 1024,
                dtype=torch.int32, device=device)
            all_blocks_cnt = torch.zeros(
                nkvh, q_blocks,
                dtype=torch.int32, device=device)

            # Fill mask with all block IDs [0, total_kv_blocks-1]
            for h_idx in range(nkvh):
                for qb in range(q_blocks):
                    all_blocks_mask[h_idx, qb, :total_kv_blocks] = \
                        torch.arange(total_kv_blocks, dtype=torch.int32,
                                     device=device)
                    all_blocks_cnt[h_idx, qb] = total_kv_blocks

            allblk_out = torch.zeros_like(output[:q_len])
            self._esimd_sdp_paged_sparse(
                query[:q_len], kv_cache, allblk_out,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                qsl,
                all_blocks_mask, all_blocks_cnt,
                nh, nkvh,
                hd, block_size,
                max_sl, self.scale,
                0,  # is_decode=False
                total_kv_blocks)

            # Run dense reference
            dense_out = torch.zeros_like(output[:q_len])
            self._esimd_sdp_paged(
                query[:q_len], kv_cache, dense_out,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                qsl,
                nh, nkvh,
                hd, block_size,
                max_sl, self.scale,
                1,  # causal=True
            )

            # Compare sparse(original) vs dense
            diff_sparse = (sparse_out.float() - dense_out.float()).abs()
            logger.info(
                "Sparse(topk) vs Dense: abs_diff mean=%.6f max=%.6f",
                float(diff_sparse.mean().item()),
                float(diff_sparse.max().item()))

            # Compare sparse(all-blocks) vs dense
            diff_all = (allblk_out.float() - dense_out.float()).abs()
            logger.info(
                "Sparse(all-blocks) vs Dense: abs_diff mean=%.6f max=%.6f",
                float(diff_all.mean().item()),
                float(diff_all.max().item()))

            # Per-Q comparison for first 4 positions
            for qi in range(min(4, q_len)):
                d = dense_out[qi].float()
                a = allblk_out[qi].float()
                s = sparse_out[qi].float()
                logger.info(
                    "  Q[%d]: dense_norm=%.4f allblk_norm=%.4f "
                    "sparse_norm=%.4f diff_allblk=%.6f diff_sparse=%.6f",
                    qi,
                    float(d.norm().item()),
                    float(a.norm().item()),
                    float(s.norm().item()),
                    float((a - d).abs().mean().item()),
                    float((s - d).abs().mean().item()))

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: InfLLMv2EsimdAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output

        self._load_kernels()

        num_actual_tokens = attn_metadata.num_actual_tokens
        q_slice = query[:num_actual_tokens]
        o_slice = output[:num_actual_tokens]

        max_seq_len = attn_metadata.max_seq_len
        is_decode = (attn_metadata.max_query_len == 1)

        if max_seq_len <= self.dense_len:
            # Short sequence: use dense attention
            self._forward_dense(q_slice, kv_cache, o_slice, attn_metadata)
        elif is_decode:
            # Long decode: sparse attention
            logger.info_once(
                "InfLLMv2 sparse decode activated: max_seq_len=%d, "
                "batch=%d, num_tokens=%d",
                max_seq_len, attn_metadata.seq_lens.shape[0],
                num_actual_tokens)
            self._forward_sparse_decode(
                q_slice, kv_cache, o_slice, attn_metadata)
        else:
            # Long prefill: sparse prefill
            logger.info_once(
                "InfLLMv2 sparse prefill activated: max_seq_len=%d, "
                "q_len=%d", max_seq_len, num_actual_tokens)
            self._forward_sparse_prefill(
                q_slice, kv_cache, o_slice, attn_metadata)

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
        key_cache, value_cache = kv_cache.unbind(0)
        triton_reshape_and_cache_flash(
            key, value,
            key_cache, value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
