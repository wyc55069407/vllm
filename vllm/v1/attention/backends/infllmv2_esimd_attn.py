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
import os

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

# Debug toggles: set to "0" to disable sparse SDP for that phase
# (pattern detection still runs, but dense SDP is used instead)
# INFLLMV2_SPARSE_PREFILL=0  → run pattern detection, use dense SDP for prefill
# INFLLMV2_SPARSE_DECODE=0   → run pattern detection, use dense SDP for decode
INFLLMV2_USE_SPARSE_PREFILL = os.environ.get("INFLLMV2_SPARSE_PREFILL", "1") != "0"
# Sparse decode: controllable via env var. Default ON (last-block fix verified).
INFLLMV2_USE_SPARSE_DECODE = os.environ.get("INFLLMV2_SPARSE_DECODE", "1") != "0"
INFLLMV2_DEBUG_TRACE = os.environ.get("INFLLMV2_DEBUG_TRACE", "0") != "0"
INFLLMV2_HOST_TIMING = os.environ.get("INFLLMV2_HOST_TIMING", "0") != "0"
_trace_call_count = 0  # module-level counter for sampling

# Lightweight host timing accumulator (toggled by INFLLMV2_HOST_TIMING=1)
import time as _time
_host_timers: dict[str, list[float]] = {}
_host_timer_step = 0


def _ht_reset():
    global _host_timers, _host_timer_step
    _host_timers.clear()
    _host_timer_step = 0


def _ht_record(name: str, dt: float):
    if name not in _host_timers:
        _host_timers[name] = []
    _host_timers[name].append(dt)


def _ht_report(num_layers: int = 28):
    """Print per-decode-step timing summary."""
    global _host_timer_step
    if not _host_timers:
        return
    print(f"\n=== InfLLMv2 Host Timing ({_host_timer_step} steps, "
          f"{num_layers} layers) ===")
    for name, vals in sorted(_host_timers.items()):
        total = sum(vals)
        per_step = total / max(1, _host_timer_step) * 1000  # ms
        avg = total / len(vals) * 1000  # ms
        print(f"  {name:30s}: {total*1000:8.1f}ms total, "
              f"{per_step:6.2f}ms/step, {avg:6.3f}ms/call ({len(vals)} calls)")
    total_all = sum(sum(v) for v in _host_timers.values())
    print(f"  {'TOTAL':30s}: {total_all*1000:8.1f}ms total, "
          f"{total_all/_host_timer_step*1000:6.2f}ms/step")
    print()


def _check_tensor(name: str, t: torch.Tensor, step: int):
    """Log NaN/inf/stats for a tensor. Used in debug trace."""
    f = t.float()
    has_nan = torch.isnan(f).any().item()
    has_inf = torch.isinf(f).any().item()
    nan_cnt = int(torch.isnan(f).sum().item()) if has_nan else 0
    inf_cnt = int(torch.isinf(f).sum().item()) if has_inf else 0
    mn = f[~torch.isnan(f)].min().item() if not has_nan or f.numel() > nan_cnt else float('nan')
    mx = f[~torch.isnan(f)].max().item() if not has_nan or f.numel() > nan_cnt else float('nan')
    avg = f[~torch.isnan(f)].mean().item() if not has_nan or f.numel() > nan_cnt else float('nan')
    tag = "OK" if not has_nan and not has_inf else "*** BAD ***"
    logger.info(
        "DIAG step=%d %s %s shape=%s nan=%d inf=%d "
        "min=%.4f max=%.4f mean=%.4f",
        step, name, tag, list(t.shape), nan_cnt, inf_cnt, mn, mx, avg)


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

        # Per-layer incremental K pooling state (keyed by batch index)
        # Cached across decode steps to avoid full recomputation
        self._cached_pooled_k: dict[int, torch.Tensor] = {}  # batch_idx -> pooled_k
        self._cached_seq_len: dict[int, int] = {}             # batch_idx -> seq_len when cached

        # Pre-allocated decode buffers (lazily sized on first use)
        self._decode_bufs_ready = False
        self._decode_batch = 0
        self._decode_num_pooled_blocks = 0
        self._decode_num_pooled = 0
        self._buf_block_scores = None
        self._buf_kv_block_scores = None
        self._buf_pooled_scores = None
        self._buf_topk_output = None
        self._buf_dummy_mask_cnt = None

        # Lazy kernel imports
        self._kernels_loaded = False
        self._esimd_sdp_paged = None
        self._esimd_sdp_paged_sparse = None
        self._esimd_k_pooling = None
        self._esimd_k_pooling_paged = None
        self._esimd_force_last_block = None
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
                esimd_infllmv2_k_pooling_paged,
                esimd_infllmv2_force_last_block,
                esimd_infllmv2_pattern_prefill,
                esimd_infllmv2_pattern_decode,
                esimd_infllmv2_mask_convert,
            )
            self._esimd_sdp_paged = esimd_sdp_paged
            self._esimd_sdp_paged_sparse = esimd_sdp_paged_sparse
            self._esimd_k_pooling = esimd_infllmv2_k_pooling
            self._esimd_k_pooling_paged = esimd_infllmv2_k_pooling_paged
            self._esimd_force_last_block = esimd_infllmv2_force_last_block
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

    def _ensure_decode_bufs(self, batch, num_pooled_blocks, num_pooled, device):
        """Pre-allocate decode intermediate buffers (reused across steps).

        No zeroing needed — all buffers are fully overwritten by the
        pattern_decode kernels (confirmed by kernel code analysis).
        """
        if (self._decode_bufs_ready
                and self._decode_batch >= batch
                and self._decode_num_pooled_blocks >= num_pooled_blocks
                and self._decode_num_pooled >= num_pooled):
            return

        nh, nkvh, hd = self.num_heads, self.num_kv_heads, self.head_size
        self._decode_batch = batch
        self._decode_num_pooled_blocks = num_pooled_blocks
        self._decode_num_pooled = num_pooled
        self._buf_block_scores = torch.empty(
            batch, nh, 1, num_pooled_blocks,
            dtype=torch.float16, device=device)
        self._buf_kv_block_scores = torch.empty(
            batch, nkvh, 1, num_pooled_blocks,
            dtype=torch.float16, device=device)
        self._buf_pooled_scores = torch.empty(
            batch, nkvh, 1, num_pooled,
            dtype=torch.float16, device=device)
        self._buf_topk_output = torch.empty(
            batch, nkvh, 1, self.topk,
            dtype=torch.int32, device=device)
        if self._buf_dummy_mask_cnt is None:
            self._buf_dummy_mask_cnt = torch.zeros(
                1, dtype=torch.int32, device=device)
        self._decode_bufs_ready = True

    def _forward_sparse_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: InfLLMv2EsimdAttentionMetadata,
    ) -> torch.Tensor:
        """Sparse decode with incremental K pooling and pre-allocated buffers.

        Caches pooled_k across decode steps. Only recomputes tail 2 pooled
        blocks when seq_len increments by 1 (typical decode). Falls back to
        full recompute on mismatch (new request, after prefill, etc.).
        """
        _ht = INFLLMV2_HOST_TIMING
        batch = attn_metadata.seq_lens.shape[0]
        block_size = kv_cache.shape[2]
        device = query.device
        nkvh = self.num_kv_heads
        nh = self.num_heads
        hd = self.head_size
        max_sl = attn_metadata.max_seq_len

        # Dimensions for pattern detection
        num_pooled_blocks = max(
            1, (max_sl - self.kernel_size + self.kernel_stride)
            // self.kernel_stride)
        pooling_stride = self.sparse_block // self.kernel_stride  # 4
        num_pooled = max(
            1, (num_pooled_blocks + pooling_stride - 1) // pooling_stride)

        # Step 1: Incremental K pooling
        if _ht:
            torch.xpu.synchronize()
            _t0 = _time.perf_counter()
        # Check if we can reuse cached pooled_k (batch=1 typical case)
        cached_sl = self._cached_seq_len.get(0, 0)
        if (batch == 1 and cached_sl > 0 and max_sl == cached_sl + 1
                and 0 in self._cached_pooled_k):
            # Incremental: reuse cached, update only tail blocks
            old_k = self._cached_pooled_k[0]
            old_npb = old_k.shape[2]
            if old_npb < num_pooled_blocks:
                # pooled_k grew by 1 block — extend
                k_pooled = torch.zeros(
                    1, nkvh, num_pooled_blocks, hd,
                    dtype=torch.float16, device=device)
                k_pooled[:, :, :old_npb, :] = old_k
            else:
                k_pooled = old_k
            # Recompute only last 2 blocks (affected by new token)
            start_block = max(0, num_pooled_blocks - 2)
            self._esimd_k_pooling_paged(
                kv_cache, k_pooled,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                nkvh, hd,
                block_size, num_pooled_blocks,
                self.kernel_size, self.kernel_stride,
                start_block)
        else:
            # Full recompute (first decode, batch>1, or mismatch)
            k_pooled = torch.zeros(
                batch, nkvh, num_pooled_blocks, hd,
                dtype=torch.float16, device=device)
            self._esimd_k_pooling_paged(
                kv_cache, k_pooled,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                nkvh, hd,
                block_size, num_pooled_blocks,
                self.kernel_size, self.kernel_stride,
                0)
        if _ht:
            torch.xpu.synchronize()
            _ht_record("k_pooling", _time.perf_counter() - _t0)

        # Cache for next decode step
        if batch == 1:
            self._cached_pooled_k[0] = k_pooled
            self._cached_seq_len[0] = max_sl

        # NaN/inf check on intermediates
        if INFLLMV2_DEBUG_TRACE:
            global _trace_call_count
            _trace_call_count += 1
            if _trace_call_count <= 5 or _trace_call_count % 200 == 0:
                _check_tensor("decode_k_pooled", k_pooled,
                              _trace_call_count)
                _check_tensor("decode_query", query, _trace_call_count)

        # Step 2: Pattern detection (decode) — use pre-allocated buffers
        if _ht:
            _t0 = _time.perf_counter()
        q_for_pattern = query.view(batch, nh, hd).unsqueeze(2)  # [B, nh, 1, hd]
        if q_for_pattern.dtype != torch.float16:
            q_for_pattern = q_for_pattern.half()
        if _ht:
            torch.xpu.synchronize()
            _ht_record("q_reshape+half", _time.perf_counter() - _t0)

        if _ht:
            _t0 = _time.perf_counter()
        self._ensure_decode_bufs(batch, num_pooled_blocks, num_pooled, device)
        if _ht:
            torch.xpu.synchronize()
            _ht_record("zero_bufs", _time.perf_counter() - _t0)

        cache_len = max_sl - 1  # history before this token

        if _ht:
            _t0 = _time.perf_counter()
        self._esimd_pattern_decode(
            q_for_pattern, k_pooled,
            self._buf_block_scores[:batch, :, :, :num_pooled_blocks],
            self._buf_kv_block_scores[:batch, :, :, :num_pooled_blocks],
            self._buf_pooled_scores[:batch, :, :, :num_pooled],
            self._buf_topk_output[:batch],
            nh, nkvh,
            1, num_pooled_blocks,
            hd, num_pooled,
            cache_len, 1,
            self.init_blocks, self.local_blocks,
            self.topk)
        if _ht:
            torch.xpu.synchronize()
            _ht_record("pattern_decode", _time.perf_counter() - _t0)

        # NaN/inf check on pattern detection outputs
        if INFLLMV2_DEBUG_TRACE:
            if _trace_call_count <= 5 or _trace_call_count % 200 == 0:
                _check_tensor("decode_block_scores",
                              self._buf_block_scores[:batch],
                              _trace_call_count)

        # Step 3: SDP — sparse or dense depending on toggle
        if INFLLMV2_USE_SPARSE_DECODE:
            # topk_output: [batch, nkvh, 1, 64] → [batch, nkvh, 64]
            sparse_mask = self._buf_topk_output[:batch].squeeze(2).int()

            # Force-insert last sparse block on GPU (no CPU sync)
            if _ht:
                _t0 = _time.perf_counter()
            self._esimd_force_last_block(
                sparse_mask, attn_metadata.seq_lens,
                nkvh, self.sparse_block)

            self._esimd_sdp_paged_sparse(
                query, kv_cache, output,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                attn_metadata.query_start_loc,
                sparse_mask, self._buf_dummy_mask_cnt,
                nh, nkvh,
                hd, block_size,
                max_sl, self.scale,
                1,  # is_decode=True
                self.topk)
            if _ht:
                torch.xpu.synchronize()
                _ht_record("force+sparse_sdp", _time.perf_counter() - _t0)

            # Debug: check sparse decode output
            if INFLLMV2_DEBUG_TRACE:
                if _trace_call_count <= 5 or _trace_call_count % 200 == 0:
                    _check_tensor("decode_sparse_output", output[:batch],
                                  _trace_call_count)
        else:
            # Dense fallback (pattern detection ran but we ignore its output)
            if _ht:
                _t0 = _time.perf_counter()
            self._forward_dense(query, kv_cache, output, attn_metadata)
            if _ht:
                torch.xpu.synchronize()
                _ht_record("dense_sdp_fallback", _time.perf_counter() - _t0)

        return output

    def _forward_sparse_prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: InfLLMv2EsimdAttentionMetadata,
    ) -> torch.Tensor:
        """Sparse prefill: k_pooling_paged → pattern_detect → mask_convert → sparse SDP.

        All operations on GPU — no CPU waits or .item() calls.
        """
        batch = attn_metadata.seq_lens.shape[0]
        block_size = kv_cache.shape[2]
        device = query.device
        q_len = query.shape[0]
        max_sl = attn_metadata.max_seq_len

        nkvh = self.num_kv_heads
        hd = self.head_size

        # Step 1: Paged K pooling — reads directly from paged KV cache
        num_pooled_blocks = (max_sl - self.kernel_size + self.kernel_stride) // self.kernel_stride
        if num_pooled_blocks <= 0:
            num_pooled_blocks = 1

        k_pooled = torch.zeros(
            batch, nkvh, num_pooled_blocks, hd,
            dtype=torch.float16, device=device)

        self._esimd_k_pooling_paged(
            kv_cache, k_pooled,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            nkvh, hd,
            block_size, num_pooled_blocks,
            self.kernel_size, self.kernel_stride,
            0)  # start_pooled_block=0 (full compute)

        # Cache pooled_k for subsequent decode steps (incremental update)
        if batch == 1:
            self._cached_pooled_k[0] = k_pooled
            self._cached_seq_len[0] = max_sl

        # NaN/inf check on prefill intermediates
        if INFLLMV2_DEBUG_TRACE:
            _check_tensor("prefill_k_pooled", k_pooled, 0)
            _check_tensor("prefill_query", query, 0)

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

        # NaN/inf check on pattern detection outputs
        if INFLLMV2_DEBUG_TRACE:
            _check_tensor("prefill_block_scores", block_scores, 0)
            _check_tensor("prefill_pooled_scores", pooled_scores, 0)

        # Debug trace: log prefill block selection
        if INFLLMV2_DEBUG_TRACE:
            tk = topk_per_token.cpu()  # [1, nkvh, q_len, 64]
            # Sample a few query positions
            sample_qs = [0, q_len // 4, q_len // 2, 3 * q_len // 4, q_len - 1]
            sample_qs = [q for q in sample_qs if q < q_len]
            for qi in sample_qs:
                for h in range(nkvh):
                    blks = tk[0, h, qi].numpy()
                    sorted_blks = sorted(blks)
                    q_blk_abs = (qi + cache_len) // self.sparse_block
                    has_init = [x for x in sorted_blks if x < self.init_blocks]
                    has_local = [x for x in sorted_blks if q_blk_abs - self.local_blocks <= x <= q_blk_abs]
                    logger.info(
                        "TRACE prefill q=%d/%d kvh=%d seq=%d q_blk=%d "
                        "cache_len=%d init_blks=%s local_blks=%s "
                        "topk_range=[%d,%d] blocks=%s",
                        qi, q_len, h, max_sl, q_blk_abs,
                        cache_len, has_init, has_local,
                        min(sorted_blks), max(sorted_blks),
                        sorted_blks[:20])
            # Also log k_pooled stats
            kp_cpu = k_pooled.cpu().float()
            logger.info(
                "TRACE prefill k_pooled shape=%s mean=%.4f std=%.4f "
                "min=%.4f max=%.4f nonzero=%d/%d",
                list(k_pooled.shape),
                kp_cpu.mean().item(), kp_cpu.std().item(),
                kp_cpu.min().item(), kp_cpu.max().item(),
                (kp_cpu.abs() > 1e-6).sum().item(), kp_cpu.numel())
            # Log pooled_scores stats
            ps_cpu = pooled_scores.cpu().float()
            logger.info(
                "TRACE prefill pooled_scores shape=%s mean=%.6f std=%.6f "
                "min=%.6f max=%.6f",
                list(pooled_scores.shape),
                ps_cpu.mean().item(), ps_cpu.std().item(),
                ps_cpu.min().item(), ps_cpu.max().item())

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

        if INFLLMV2_DEBUG_TRACE:
            mc = mask_cnt_out.cpu()
            mo = mask_out.cpu()
            logger.info(
                "TRACE mask_convert q_blocks=%d total_kv_blocks=%d "
                "mask_cnt: min=%d max=%d mean=%.1f",
                q_blocks, total_kv_blocks,
                mc.min().item(), mc.max().item(), mc.float().mean().item())
            # Show first and last q_block's selected blocks
            for qb_idx in [0, q_blocks - 1]:
                for h in range(nkvh):
                    cnt = int(mc[h, qb_idx].item())
                    blks = mo[h, qb_idx, :cnt].numpy().tolist()
                    logger.info(
                        "TRACE mask_convert kvh=%d qblk=%d cnt=%d "
                        "blocks=%s",
                        h, qb_idx, cnt, blks[:30])

        # Step 5: SDP — sparse or dense depending on toggle
        if INFLLMV2_USE_SPARSE_PREFILL:
            sparse_mask_cnt = mask_cnt_out.int()
            sparse_mask = mask_out.int()
            qsl = torch.tensor([0, q_len], dtype=torch.int32, device=device)

            # Debug: compare fast vs old kernel outputs in-place
            _cmp_count = getattr(self, '_cmp_count', 0)
            if _cmp_count < 1 and os.environ.get("INFLLMV2_CMP_KERNELS", "0") == "1":
                # Save KV cache pages for this layer (first time only)
                _kv_dump_done = getattr(self, '_kv_dump_done', False)
                if not _kv_dump_done:
                    # Save compact KV: only pages referenced by block table
                    bt_cpu = attn_metadata.block_table.cpu()
                    sl_val = int(attn_metadata.seq_lens.cpu().max().item())
                    n_pages = (sl_val + block_size - 1) // block_size
                    used_pages = bt_cpu[0, :n_pages].tolist()
                    kv_pages = kv_cache[:, used_pages].cpu().clone()
                    kv_dump = {
                        'kv_pages': kv_pages,  # [2, n_pages, bs, nkvh, hd]
                        'used_pages': used_pages,
                        'kv_full_shape': kv_cache.shape,
                        'kv_strides': kv_cache.stride(),
                        'query': query[:q_len].cpu().clone(),
                        'block_table': attn_metadata.block_table.cpu().clone(),
                        'seq_lens': attn_metadata.seq_lens.cpu().clone(),
                        'sparse_mask': sparse_mask.cpu().clone(),
                        'sparse_mask_cnt': sparse_mask_cnt.cpu().clone(),
                        'nh': nh, 'nkvh': nkvh, 'hd': hd, 'bs': block_size,
                        'max_sl': max_sl, 'scale': self.scale, 'topk': self.topk,
                    }
                    torch.save(kv_dump, '/tmp/sparse_kv_dump.pt')
                    logger.info("KV_DUMP saved: n_pages=%d kv_pages=%s used=%s",
                               n_pages, kv_pages.shape, used_pages[:10])
                    self._kv_dump_done = True

                # Log block table and mask details
                bt = attn_metadata.block_table
                bt_cpu = bt.cpu()
                logger.info(
                    "CMP_DETAIL: bt_shape=%s bt[:20]=%s max_pg=%d "
                    "mask_cnt_range=[%d,%d] q_len=%d max_sl=%d hist=%d "
                    "kv_strides=%s",
                    bt.shape, bt_cpu[0, :20].tolist(), bt_cpu.max().item(),
                    sparse_mask_cnt.min().item(), sparse_mask_cnt.max().item(),
                    q_len, max_sl, max_sl - q_len,
                    kv_cache.stride())

                # Run old kernel
                out_old = torch.zeros_like(output[:q_len])
                os.environ["INFLLMV2_FAST_PREFILL"] = "0"
                self._esimd_sdp_paged_sparse(
                    query[:q_len], kv_cache, out_old,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    qsl,
                    sparse_mask, sparse_mask_cnt,
                    nh, nkvh, hd, block_size,
                    max_sl, self.scale, 0, self.topk)
                torch.xpu.synchronize()

                # Run fast kernel
                out_fast = torch.zeros_like(output[:q_len])
                os.environ["INFLLMV2_FAST_PREFILL"] = "1"
                self._esimd_sdp_paged_sparse(
                    query[:q_len], kv_cache, out_fast,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    qsl,
                    sparse_mask, sparse_mask_cnt,
                    nh, nkvh, hd, block_size,
                    max_sl, self.scale, 0, self.topk)
                torch.xpu.synchronize()

                diff = (out_old.float() - out_fast.float()).abs()
                # Find which Q positions have largest diff
                row_diff = diff.max(dim=-1).values.max(dim=-1).values  # [q_len]
                top5_rows = row_diff.topk(5)
                logger.info(
                    "CMP layer=%d: old_max=%.4f fast_max=%.4f "
                    "diff_mean=%.6f diff_max=%.4f "
                    "worst_rows=%s worst_diffs=%s",
                    getattr(self, '_layer_idx', -1),
                    out_old.float().abs().max().item(),
                    out_fast.float().abs().max().item(),
                    diff.mean().item(), diff.max().item(),
                    top5_rows.indices.tolist(),
                    [f"{v:.2f}" for v in top5_rows.values.tolist()])

                # Check first and last q_blocks specifically
                for qb in [0, 1, (q_len - 1) // 16]:
                    start = qb * 16
                    end = min(start + 16, q_len)
                    d = diff[start:end]
                    logger.info(
                        "CMP qblock=%d: diff_max=%.4f old_max=%.4f fast_max=%.4f",
                        qb, d.max().item(), out_old[start:end].float().abs().max().item(),
                        out_fast[start:end].float().abs().max().item())

                # Use old kernel output for the actual computation
                output[:q_len].copy_(out_old)
            self._cmp_count = _cmp_count + 1

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

            # Debug: check for NaN in sparse prefill output
            if INFLLMV2_DEBUG_TRACE:
                o_check = output[:q_len]  # [q_len, nh, hd]
                nan_per_row = torch.isnan(o_check).any(dim=-1).any(dim=-1)  # [q_len]
                nan_rows = nan_per_row.nonzero(as_tuple=False).squeeze(-1)
                if nan_rows.numel() > 0:
                    first_r = int(nan_rows[0].item())
                    last_r = int(nan_rows[-1].item())
                    nr = nan_rows.numel()
                    # Check which heads have NaN at first_r
                    first_row_nan = torch.isnan(o_check[first_r]).any(dim=-1)
                    nan_heads = first_row_nan.nonzero(as_tuple=False).squeeze(-1).tolist()
                    logger.info(
                        "TRACE sparse_prefill: NaN rows=%d first=%d last=%d "
                        "q_len=%d last_blk_start=%d "
                        "first_row_nan_heads=%s",
                        nr, first_r, last_r, q_len,
                        (q_len // 16) * 16, str(nan_heads[:10]))
                else:
                    logger.info(
                        "TRACE sparse_prefill: OK mean=%.4f abs_max=%.4f",
                        o_check.float().mean().item(),
                        o_check.float().abs().max().item())
        else:
            # Dense fallback (pattern detection ran but we ignore its output)
            self._forward_dense(
                query[:q_len], kv_cache, output[:q_len], attn_metadata)

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

        _ht = INFLLMV2_HOST_TIMING

        # Track decode steps: increment when first layer sees decode
        if _ht and is_decode and not hasattr(self, '_ht_layer_idx'):
            # Assign layer index based on creation order
            if not hasattr(InfLLMv2EsimdAttentionImpl, '_ht_next_idx'):
                InfLLMv2EsimdAttentionImpl._ht_next_idx = 0
            self._ht_layer_idx = InfLLMv2EsimdAttentionImpl._ht_next_idx
            InfLLMv2EsimdAttentionImpl._ht_next_idx += 1
        if _ht and is_decode and getattr(self, '_ht_layer_idx', -1) == 0:
            global _host_timer_step
            _host_timer_step += 1
            if _host_timer_step % 16 == 0:
                _ht_report(num_layers=28)

        if max_seq_len <= self.dense_len:
            # Short sequence: use dense attention
            if _ht:
                torch.xpu.synchronize()
                _t0 = _time.perf_counter()
            self._forward_dense(q_slice, kv_cache, o_slice, attn_metadata)
            if _ht:
                torch.xpu.synchronize()
                _ht_record("short_dense_sdp", _time.perf_counter() - _t0)
        elif is_decode:
            if not INFLLMV2_USE_SPARSE_DECODE:
                # Sparse decode disabled — skip pattern detection entirely
                logger.info_once(
                    "InfLLMv2 decode: dense fallback (max_seq_len=%d)",
                    max_seq_len)
                if _ht:
                    torch.xpu.synchronize()
                    _t0 = _time.perf_counter()
                self._forward_dense(
                    q_slice, kv_cache, o_slice, attn_metadata)
                if _ht:
                    torch.xpu.synchronize()
                    _ht_record("long_dense_sdp", _time.perf_counter() - _t0)
            else:
                # Long decode: pattern detection + sparse SDP
                logger.info_once(
                    "InfLLMv2 decode path: max_seq_len=%d, batch=%d, "
                    "sparse_sdp=%s",
                    max_seq_len, attn_metadata.seq_lens.shape[0],
                    INFLLMV2_USE_SPARSE_DECODE)
                self._forward_sparse_decode(
                    q_slice, kv_cache, o_slice, attn_metadata)
        else:
            # Long prefill: clear incremental cache (prefill recomputes all)
            self._cached_pooled_k.clear()
            self._cached_seq_len.clear()
            # Long prefill: pattern detection + sparse/dense SDP
            logger.info_once(
                "InfLLMv2 prefill path: max_seq_len=%d, q_len=%d, "
                "sparse_sdp=%s",
                max_seq_len, num_actual_tokens,
                INFLLMV2_USE_SPARSE_PREFILL)
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
