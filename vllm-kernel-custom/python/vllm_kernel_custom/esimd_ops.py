"""vllm-kernel-custom: Python wrappers for ESIMD kernels.

Each op has a specific named function with proper typed parameters,
replacing the old generic esimd_kernel_uni dispatch.
"""
import torch

_ops = torch.ops.vllm_kernel_custom

# ============================================================
# Non-LGRF ops (common_ops)
# ============================================================

def esimd_mul_scale_factor_and_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, length: int, factor: float
) -> torch.Tensor:
    """Fused residual add: c = a_bf16 * factor + b_fp16."""
    return _ops.esimd_add(a, b, c, length, factor)


def esimd_gemv_fp8(
    input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
    bias: torch.Tensor, output: torch.Tensor,
    M: int, N: int, K: int, batch: int,
    scale_block_n: int, scale_block_k: int, has_bias: int,
) -> torch.Tensor:
    """FP8 weight GEMV: output = input @ dequant(weight_fp8, scale) + bias."""
    return _ops.esimd_gemv_fp8(input, weight, weight_scale, bias, output,
                               M, N, K, batch, scale_block_n, scale_block_k, has_bias)


def esimd_fp8_dequant(
    weight_fp8: torch.Tensor, scale: torch.Tensor, output: torch.Tensor,
    N: int, K: int, scale_block_n: int, scale_block_k: int,
) -> torch.Tensor:
    """Dequantize FP8 weights."""
    return _ops.esimd_fp8_dequant(weight_fp8, scale, output, N, K, scale_block_n, scale_block_k)


def esimd_bmm_gemv_fp8(
    input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
    M: int, N: int, K: int, stride: int, heads: int, scale: float,
) -> torch.Tensor:
    """Batch matrix-multiply with FP8 weights."""
    return _ops.esimd_bmm_gemv_fp8(input, weight, output, M, N, K, stride, heads, scale)


def esimd_sdpa_normal(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    kv_indices: torch.Tensor, output: torch.Tensor,
    tmp_softmax: torch.Tensor, tmp_out: torch.Tensor, tmp_reduce: torch.Tensor,
    num_heads: int, seq_len: int, kv_len: int, head_dim: int, num_kv_heads: int,
    attn_scale: float, kv_scale: float,
) -> torch.Tensor:
    """Scaled dot-product attention with reduce."""
    return _ops.esimd_sdpa_normal(q, k, v, kv_indices, output,
                                  tmp_softmax, tmp_out, tmp_reduce,
                                  num_heads, seq_len, kv_len, head_dim, num_kv_heads,
                                  attn_scale, kv_scale)


def esimd_sdpa_mla(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    kv_indices: torch.Tensor, output: torch.Tensor,
    tmp_softmax: torch.Tensor, tmp_out: torch.Tensor, tmp_reduce: torch.Tensor,
    tmp_extra0: torch.Tensor, tmp_extra1: torch.Tensor,
    num_heads: int, seq_len: int, kv_len: int, head_dim: int, num_kv_heads: int,
    qk_dim: int, v_dim: int, extra_param: int,
    attn_scale: float, kv_scale: float,
) -> torch.Tensor:
    """SDPA with Multi-head Latent Attention (MLA)."""
    return _ops.esimd_sdpa_mla(q, k, v, kv_indices, output,
                               tmp_softmax, tmp_out, tmp_reduce, tmp_extra0, tmp_extra1,
                               num_heads, seq_len, kv_len, head_dim, num_kv_heads,
                               qk_dim, v_dim, extra_param, attn_scale, kv_scale)


def esimd_cat_qk(
    q_nope: torch.Tensor, q_pe: torch.Tensor,
    k_nope: torch.Tensor, k_pe: torch.Tensor,
    q_out: torch.Tensor, k_out: torch.Tensor,
    seq_len: int, q_heads: int, kv_heads: int, qk_head_dim: int, stride: int,
) -> torch.Tensor:
    """Concatenate Q/K nope and pe parts for MLA."""
    return _ops.esimd_cat_qk(q_nope, q_pe, k_nope, k_pe, q_out, k_out,
                             seq_len, q_heads, kv_heads, qk_head_dim, stride)


def esimd_rms_norm(
    weight: torch.Tensor, residual: torch.Tensor,
    input: torch.Tensor, output: torch.Tensor,
    hidden_size: int, seq_len: int, add_residual: int, flag: int, eps: float,
) -> torch.Tensor:
    """RMSNorm with optional residual addition."""
    return _ops.esimd_rms_norm(weight, residual, input, output,
                               hidden_size, seq_len, add_residual, flag, eps)


def esimd_rms_norm_qk(
    q_weight: torch.Tensor, kv_weight: torch.Tensor,
    q_input: torch.Tensor, kv_input: torch.Tensor,
    q_output: torch.Tensor, kv_output: torch.Tensor,
    q_dim: int, kv_dim: int, seq_len: int, stride: int,
    q_eps: float, kv_eps: float,
) -> torch.Tensor:
    """RMSNorm for Q and K separately."""
    return _ops.esimd_rms_norm_qk(q_weight, kv_weight, q_input, kv_input,
                                   q_output, kv_output, q_dim, kv_dim, seq_len, stride,
                                   q_eps, kv_eps)


def esimd_grouped_topk(
    scores: torch.Tensor, output_vals: torch.Tensor,
    output_ids: torch.Tensor, tmp: torch.Tensor,
    batch: int, experts: int, topk: int, groups: int, group_topk: int,
    score_scale: float,
) -> torch.Tensor:
    """Grouped top-K selection."""
    return _ops.esimd_grouped_topk(scores, output_vals, output_ids, tmp,
                                   batch, experts, topk, groups, group_topk, score_scale)


def esimd_grouped_topk_fused_gate(
    input: torch.Tensor, gate_weight: torch.Tensor,
    output_vals: torch.Tensor, output_ids: torch.Tensor,
    tmp0: torch.Tensor, tmp1: torch.Tensor, tmp2: torch.Tensor,
    batch: int, experts: int, topk: int, groups: int, group_topk: int,
    score_scale: float,
) -> torch.Tensor:
    """Grouped top-K fused with gate computation."""
    return _ops.esimd_grouped_topk_fused_gate(input, gate_weight, output_vals, output_ids,
                                               tmp0, tmp1, tmp2,
                                               batch, experts, topk, groups, group_topk,
                                               score_scale)


def esimd_grouped_topk_kimi(
    scores: torch.Tensor, output_vals: torch.Tensor,
    output_ids: torch.Tensor, tmp: torch.Tensor,
    batch: int, experts: int, topk: int, groups: int, group_topk: int,
    score_scale: float,
) -> torch.Tensor:
    """Grouped top-K (Kimi variant)."""
    return _ops.esimd_grouped_topk_kimi(scores, output_vals, output_ids, tmp,
                                        batch, experts, topk, groups, group_topk, score_scale)


def esimd_grouped_topk_fused_gate_kimi(
    input: torch.Tensor, gate_weight: torch.Tensor,
    output_vals: torch.Tensor, output_ids: torch.Tensor,
    tmp0: torch.Tensor, tmp1: torch.Tensor, tmp2: torch.Tensor,
    batch: int, experts: int, topk: int, groups: int, group_topk: int,
    score_scale: float,
) -> torch.Tensor:
    """Grouped top-K fused with gate (Kimi variant)."""
    return _ops.esimd_grouped_topk_fused_gate_kimi(input, gate_weight, output_vals, output_ids,
                                                     tmp0, tmp1, tmp2,
                                                     batch, experts, topk, groups, group_topk,
                                                     score_scale)


def esimd_shared_expert(
    input: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor,
    down_weight: torch.Tensor, output: torch.Tensor,
    tmp0: torch.Tensor, tmp1: torch.Tensor,
    M: int, N: int, K: int,
) -> torch.Tensor:
    """Shared expert mega kernel for MoE."""
    return _ops.esimd_shared_expert(input, gate_weight, up_weight, down_weight,
                                    output, tmp0, tmp1, M, N, K)


def esimd_rope(
    q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor,
    positions: torch.Tensor, positions_k: torch.Tensor,
    q_heads: int, rope_dim: int, q_stride: int,
    kv_heads: int, k_rope_dim: int, k_stride: int,
    k_nope_stride: int, seq_len: int, flag: int,
) -> torch.Tensor:
    """Rotary position embedding (dynamic shape)."""
    return _ops.esimd_rope(q, k, cos_sin_cache, positions, positions_k,
                           q_heads, rope_dim, q_stride, kv_heads, k_rope_dim, k_stride,
                           k_nope_stride, seq_len, flag)


def esimd_update_kv(
    kv_cache: torch.Tensor, new_kv: torch.Tensor, indices: torch.Tensor,
    seq_len: int, head_dim: int, num_heads: int,
) -> torch.Tensor:
    """Update KV cache with indexed values."""
    return _ops.esimd_update_kv(kv_cache, new_kv, indices, seq_len, head_dim, num_heads)


def esimd_topk_sort(
    i0: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor,
    i3: torch.Tensor, i4: torch.Tensor, i5: torch.Tensor,
    i6: torch.Tensor, i7: torch.Tensor, output: torch.Tensor,
    param0: int, param1: int,
) -> torch.Tensor:
    """Top-K radix sort."""
    return _ops.esimd_topk_sort(i0, i1, i2, i3, i4, i5, i6, i7, output, param0, param1)


def esimd_dsa_qk_attn(
    q: torch.Tensor, k: torch.Tensor, output: torch.Tensor,
    indices: torch.Tensor, tmp: torch.Tensor,
    seq_len: int, head_dim: int, num_heads: int,
) -> torch.Tensor:
    """DSA QK attention."""
    return _ops.esimd_dsa_qk_attn(q, k, output, indices, tmp, seq_len, head_dim, num_heads)


def esimd_topk_multi_round(
    i0: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor,
    i3: torch.Tensor, i4: torch.Tensor, i5: torch.Tensor,
    p0: int, p1: int, p2: int, p3: int, p4: int,
) -> torch.Tensor:
    """Multi-round top-K."""
    return _ops.esimd_topk_multi_round(i0, i1, i2, i3, i4, i5, p0, p1, p2, p3, p4)


def esimd_update_kv_dsa(
    kv: torch.Tensor, new_kv: torch.Tensor, indices: torch.Tensor,
    tmp0: torch.Tensor, tmp1: torch.Tensor, param: int,
) -> torch.Tensor:
    """Update KV cache with DSA indices."""
    return _ops.esimd_update_kv_dsa(kv, new_kv, indices, tmp0, tmp1, param)


def esimd_rope_quant_updatek(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
    t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor,
    t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, t9: torch.Tensor,
    p0: int, p1: int, p2: int, p3: int, f0: float,
) -> torch.Tensor:
    """Fused RoPE + quantize + K update."""
    return _ops.esimd_rope_quant_updatek(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
                                          p0, p1, p2, p3, f0)


def esimd_weight_qscale_fuse(
    weight: torch.Tensor, scale: torch.Tensor,
    output: torch.Tensor, tmp: torch.Tensor,
    param: int, scale_factor: float,
) -> torch.Tensor:
    """Fuse weight with quantization scale."""
    return _ops.esimd_weight_qscale_fuse(weight, scale, output, tmp, param, scale_factor)


def esimd_mla_mega(
    input_norm_weight: torch.Tensor, lora_a_weight: torch.Tensor,
    lora_a_weight_scale: torch.Tensor,
    lora_a_norm_q_weight: torch.Tensor, lora_a_norm_kv_weight: torch.Tensor,
    lora_b_q_weight: torch.Tensor, lora_b_q_weight_scale: torch.Tensor,
    w_kc_fp8: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor,
    lora_a_bias: torch.Tensor, lora_b_bias: torch.Tensor,
    hidden_states: torch.Tensor, residual: torch.Tensor,
    q_out: torch.Tensor, k_out: torch.Tensor,
    k_nope: torch.Tensor, intermedia: torch.Tensor,
    hidden_size: int, seq_len: int, add_residual: int,
    lora_a_dim: int, block_n: int, block_k: int,
    lora_a_has_bias: int, lora_b_has_bias: int,
    q_lora_rank: int, kv_lora_rank: int,
    q_header_num: int, kv_header_num: int,
    qk_head_dim: int, qk_rope_head_dim: int,
    input_norm_eps: float, q_a_norm_eps: float,
    kv_a_norm_eps: float, bmm_w_scale: float,
) -> torch.Tensor:
    """Fused MLA pipeline: norm -> LoRA A -> norm Q/K -> LoRA B -> BMM -> RoPE -> cat."""
    return _ops.esimd_mla_mega(
        input_norm_weight, lora_a_weight, lora_a_weight_scale,
        lora_a_norm_q_weight, lora_a_norm_kv_weight,
        lora_b_q_weight, lora_b_q_weight_scale,
        w_kc_fp8, cos_sin_cache, positions,
        lora_a_bias, lora_b_bias, hidden_states, residual,
        q_out, k_out, k_nope, intermedia,
        hidden_size, seq_len, add_residual,
        lora_a_dim, block_n, block_k,
        lora_a_has_bias, lora_b_has_bias,
        q_lora_rank, kv_lora_rank,
        q_header_num, kv_header_num,
        qk_head_dim, qk_rope_head_dim,
        input_norm_eps, q_a_norm_eps, kv_a_norm_eps, bmm_w_scale)


def esimd_dsa_mega(
    x: torch.Tensor, q_lora: torch.Tensor,
    w_qb_weight: torch.Tensor, w_qb_scale: torch.Tensor,
    wk_weight: torch.Tensor, wk_scale: torch.Tensor,
    query: torch.Tensor, key: torch.Tensor,
    knorm_weight: torch.Tensor, knorm_bias: torch.Tensor,
    cos_sin_cache: torch.Tensor, positions: torch.Tensor,
    k_cache: torch.Tensor, k_scale: torch.Tensor,
    query_out: torch.Tensor, q_scale: torch.Tensor,
    weights: torch.Tensor, weights_proj_weight: torch.Tensor,
    topk_indices: torch.Tensor, index_score_rsv: torch.Tensor,
    out_idx: torch.Tensor, out_ordered: torch.Tensor,
    kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
    kv_indptr_updated: torch.Tensor, kv_indices_new: torch.Tensor,
    batch_num: int, max_q_len: int,
    tbo_start_batch_idx: int, seq_lens_cpu_addr: int,
    tokens_per_batch: int, total_count_stride: int,
    real_total_count_reserved: int, groups: int,
    knorm_eps: float, softmax_scale: float,
) -> torch.Tensor:
    """Fused DSA pipeline."""
    return _ops.esimd_dsa_mega(
        x, q_lora, w_qb_weight, w_qb_scale, wk_weight, wk_scale,
        query, key, knorm_weight, knorm_bias, cos_sin_cache, positions,
        k_cache, k_scale, query_out, q_scale,
        weights, weights_proj_weight, topk_indices, index_score_rsv,
        out_idx, out_ordered, kv_indptr, kv_indices,
        kv_indptr_updated, kv_indices_new,
        batch_num, max_q_len, tbo_start_batch_idx, seq_lens_cpu_addr,
        tokens_per_batch, total_count_stride, real_total_count_reserved, groups,
        knorm_eps, softmax_scale)


# ============================================================
# oneDNN ops (common_ops)
# ============================================================

def onednn_w8a16_fp8(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
    bias: torch.Tensor, output: torch.Tensor,
    M: int, N: int, K: int, has_bias: int,
) -> torch.Tensor:
    """oneDNN FP8 GEMM: output = x @ dequant(weight_fp8, scales) [+ bias].
    x: [M,K] FP16/BF16, weight: [N,K] FP8_E4M3, scales: [N] FP32."""
    return _ops.onednn_w8a16_fp8(x, weight, scales, bias, output, M, N, K, has_bias)


def onednn_w8a16_fp8_block(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
    bias: torch.Tensor, output: torch.Tensor,
    M: int, N: int, K: int,
    block_k: int, block_n: int, has_bias: int,
) -> torch.Tensor:
    """oneDNN FP8 GEMM with block quantization.
    x: [M,K] FP16/BF16, weight: [N,K] FP8_E4M3.

    Scales can be either:
      - [K/block_k, N] FP32 (K-grouped, per-N) — passed directly
      - [K/block_k, N/block_n] FP32 (2D block) — auto-expanded to [K/block_k, N]
        via repeat_interleave along N dim

    oneDNN on BMG/Xe2 only supports K-grouped + per-N natively (mask=3, groups={block_k,1}).
    2D block scales are emulated by repeating each scale block_n times along N."""
    if scales.ndim == 2 and scales.shape[1] != N:
        # 2D block scales [K/bk, N/bn] -> expand to [K/bk, N]
        scales = scales.repeat_interleave(block_n, dim=1)[:, :N].contiguous()
    return _ops.onednn_w8a16_fp8_block(
        x, weight, scales, bias, output, M, N, K, block_k, block_n, has_bias)


def onednn_w4a16_int4(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
    zp: torch.Tensor, bias: torch.Tensor, output: torch.Tensor,
    M: int, N: int, K: int,
    group_size: int, has_zp: int, has_bias: int,
) -> torch.Tensor:
    """oneDNN INT4 W4A16 GEMM for GPTQ-style group quantization.
    x: [M,K] FP16/BF16, weight: [N,K] u4 packed (N*K/2 bytes),
    scales: [K/group_size, N] FP32,
    zp: [K/group_size, N] u4 packed (optional, for asymmetric quant)."""
    return _ops.onednn_w4a16_int4(
        x, weight, scales, zp, bias, output, M, N, K,
        group_size, has_zp, has_bias)


# ============================================================
# LGRF ops (common_ops_lgrf) — compiled with doubleGRF
# ============================================================

def esimd_mul_lgrf(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, flag: int, length: int
) -> torch.Tensor:
    """Element-wise multiply: c = a * b (fp16, LGRF)."""
    return _ops.esimd_mul_lgrf(a, b, c, flag, length)


def esimd_sdp_fp16(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    norm_alpha: torch.Tensor, output: torch.Tensor,
    q_len: int, kv_len: int, head_q: int, head_kv: int,
) -> torch.Tensor:
    """Full attention SDP FP16 (non-causal Flash Attention, LGRF/doubleGRF).
    Input: [L, H, 128] contiguous. head_dim=128."""
    return _ops.esimd_sdp_fp16(Q, K, V, norm_alpha, output,
                                q_len, kv_len, head_q, head_kv)


def esimd_sdp_bf16(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    norm_alpha: torch.Tensor, output: torch.Tensor,
    q_len: int, kv_len: int, head_q: int, head_kv: int,
) -> torch.Tensor:
    """Full attention SDP BF16 (non-causal Flash Attention, LGRF/doubleGRF).
    Input: [L, H, 128] contiguous. head_dim=128."""
    return _ops.esimd_sdp_bf16(Q, K, V, norm_alpha, output,
                                q_len, kv_len, head_q, head_kv)


def esimd_sdp_bf16io(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    norm_alpha: torch.Tensor, output: torch.Tensor,
    q_len: int, kv_len: int, head_q: int, head_kv: int,
) -> torch.Tensor:
    """Full attention SDP BF16io hybrid (non-causal Flash Attention, LGRF/doubleGRF).
    BF16 in/out, hybrid BF16/FP16 internal DPAS. Fastest variant."""
    return _ops.esimd_sdp_bf16io(Q, K, V, norm_alpha, output,
                                  q_len, kv_len, head_q, head_kv)


def esimd_sdp_mla_lgrf(
    q_extend: torch.Tensor, k_extend: torch.Tensor, v_extend: torch.Tensor,
    k_buffer: torch.Tensor, v_buffer: torch.Tensor,
    kv_indices: torch.Tensor, o_extend: torch.Tensor,
    num_heads: int, num_heads_kv: int,
    extend_seq_len: int, prefix_seq_len: int,
    qk_dim: int, v_dim: int, attn_scale: float,
) -> torch.Tensor:
    """SDP MLA with XMX systolic arrays (LGRF, doubleGRF)."""
    return _ops.esimd_sdp_mla_lgrf(q_extend, k_extend, v_extend,
                                    k_buffer, v_buffer, kv_indices, o_extend,
                                    num_heads, num_heads_kv,
                                    extend_seq_len, prefix_seq_len,
                                    qk_dim, v_dim, attn_scale)


def esimd_sdp_paged(
    query: torch.Tensor, kv_cache: torch.Tensor, output: torch.Tensor,
    block_table: torch.Tensor, seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_heads: int, num_kv_heads: int,
    head_dim: int, block_size: int,
    max_seq_len: int, attn_scale: float,
    causal: int,
) -> torch.Tensor:
    """Paged SDP attention (LGRF, doubleGRF).

    Handles both decode (query_len=1) and prefill (query_len>1) with
    causal masking and paged KV cache.

    Args:
        query: [num_tokens, num_heads, head_dim] bf16
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim] bf16 (NHD)
        output: [num_tokens, num_heads, head_dim] bf16 — output buffer
        block_table: [batch, max_blocks_per_seq] i32
        seq_lens: [batch] i32 — total KV length per request
        query_start_loc: [batch+1] i32 — cumulative query token offsets
        num_heads: number of query heads
        num_kv_heads: number of KV heads (GQA)
        head_dim: 128 or 256
        block_size: KV cache block size (power of 2)
        max_seq_len: maximum sequence length in batch
        attn_scale: 1/sqrt(head_dim)
        causal: 1 for causal masking
    """
    return _ops.esimd_sdp_paged(
        query, kv_cache, output,
        block_table, seq_lens, query_start_loc,
        num_heads, num_kv_heads,
        head_dim, block_size,
        max_seq_len, attn_scale, causal)


def esimd_sdp_paged_sparse(
    query: torch.Tensor, kv_cache: torch.Tensor, output: torch.Tensor,
    block_table: torch.Tensor, seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    sparse_mask: torch.Tensor, sparse_mask_cnt: torch.Tensor,
    num_heads: int, num_kv_heads: int,
    head_dim: int, block_size: int,
    max_seq_len: int, attn_scale: float,
    is_decode: int, num_sparse_blocks: int,
) -> torch.Tensor:
    """Sparse paged SDP attention for InfLLMv2 (LGRF, doubleGRF).

    Decode: mask = [batch, nkvh, num_sparse_blocks] u32 block indices
    Prefill: mask = [nkvh, q_blocks, 1024] u32 union block indices
             mask_cnt = [nkvh, q_blocks] u32 count
    """
    return _ops.esimd_sdp_paged_sparse(
        query, kv_cache, output,
        block_table, seq_lens, query_start_loc,
        sparse_mask, sparse_mask_cnt,
        num_heads, num_kv_heads,
        head_dim, block_size,
        max_seq_len, attn_scale,
        is_decode, num_sparse_blocks)


def esimd_infllmv2_k_pooling(
    key_cache: torch.Tensor, key_pooled: torch.Tensor,
    num_kv_heads: int, head_dim: int,
    kv_len: int, num_blocks: int,
    kernel_size: int, kernel_stride: int,
) -> torch.Tensor:
    """InfLLMv2 K cache sliding window mean pooling."""
    return _ops.esimd_infllmv2_k_pooling(
        key_cache, key_pooled,
        num_kv_heads, head_dim,
        kv_len, num_blocks,
        kernel_size, kernel_stride)


def esimd_infllmv2_pattern_prefill(
    query: torch.Tensor, key_pooled: torch.Tensor,
    block_scores: torch.Tensor, pooled_scores: torch.Tensor,
    topk_output: torch.Tensor,
    num_heads: int, num_kv_heads: int,
    seq_len: int, num_blocks: int,
    head_dim: int, num_pooled: int,
    cache_len: int, causal: int,
    init_block: int, local_block: int,
    topk: int,
) -> torch.Tensor:
    """InfLLMv2 prefill pattern detection: qk_gemm → max_pool → topk."""
    return _ops.esimd_infllmv2_pattern_prefill(
        query, key_pooled,
        block_scores, pooled_scores, topk_output,
        num_heads, num_kv_heads,
        seq_len, num_blocks,
        head_dim, num_pooled,
        cache_len, causal,
        init_block, local_block,
        topk)


def esimd_infllmv2_pattern_decode(
    query: torch.Tensor, key_pooled: torch.Tensor,
    block_scores: torch.Tensor, kv_block_scores: torch.Tensor,
    pooled_scores: torch.Tensor, topk_output: torch.Tensor,
    num_heads: int, num_kv_heads: int,
    seq_len: int, num_blocks: int,
    head_dim: int, num_pooled: int,
    cache_len: int, causal: int,
    init_block: int, local_block: int,
    topk: int,
) -> torch.Tensor:
    """InfLLMv2 decode pattern detection: qk_gemm → softmax → pool → max_pool → topk."""
    return _ops.esimd_infllmv2_pattern_decode(
        query, key_pooled,
        block_scores, kv_block_scores,
        pooled_scores, topk_output,
        num_heads, num_kv_heads,
        seq_len, num_blocks,
        head_dim, num_pooled,
        cache_len, causal,
        init_block, local_block,
        topk)


def esimd_infllmv2_mask_convert(
    mask_orig: torch.Tensor, mask_out: torch.Tensor,
    mask_cnt_out: torch.Tensor,
    qlen: int, num_kv_heads: int,
    total_kv_blocks: int,
) -> torch.Tensor:
    """InfLLMv2 mask convert: per-token [nkvh, qlen, 64] -> per-q-block union [nkvh, q_blocks, 1024]."""
    return _ops.esimd_infllmv2_mask_convert(
        mask_orig, mask_out, mask_cnt_out,
        qlen, num_kv_heads, total_kv_blocks)


def esimd_gdn_update(
    A_log: torch.Tensor, dt_bias: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    state: torch.Tensor, output: torch.Tensor,
    cu_seqlens: torch.Tensor, state_indices: torch.Tensor,
    N: int, H: int, HV: int, K: int, V: int,
    scale: float, inplace_state: int,
) -> torch.Tensor:
    """GDN (Gated Delta Network) state update (LGRF, doubleGRF).

    Replaces both Triton chunk_gated_delta_rule (NaN on XPU) and
    PyTorch sequential fallback (slow). Handles both decode (T=1)
    and prefill (T>1) via sequential token processing.

    Args:
        A_log: [HV] f32 — log decay parameters
        dt_bias: [HV] f32 — softplus bias
        a: [T_total, HV] bf16 — gate input
        b: [T_total, HV] bf16 — beta input
        q: [T_total, H, K] bf16 — query (H = query/key heads)
        k: [T_total, H, K] bf16 — key
        v: [T_total, HV, V] bf16 — value (HV = value heads)
        state: [num_states, HV, V, K] f32 — recurrent state (updated in-place)
        output: [T_total, HV, V] bf16 — output buffer
        cu_seqlens: [N+1] i32 — cumulative sequence lengths
        state_indices: [N] i32 — index into state for each sequence
        N: number of sequences
        H: number of query/key heads
        HV: number of value heads
        K: key dimension (128)
        V: value dimension (128)
        scale: attention scale (typically K^-0.5)
        inplace_state: 1 to update state in-place
    """
    return _ops.esimd_gdn_update(
        A_log, dt_bias, a, b, q, k, v,
        state, output, cu_seqlens, state_indices,
        N, H, HV, K, V, scale, inplace_state)
