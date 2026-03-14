"""
Test vllm-xpu-kernels Flash Attention 2 (cutlass_paged_decode_xe2 / chunk_prefill)
from vllm_xpu_kernels.flash_attn_interface.

Tests:
  1. Correctness: small shapes, causal=True and causal=False, vs PyTorch SDPA
  2. Non-paged (cu_seqlens_k path) full attention
  3. Paged (block_table + seqused_k path) decode attention
  4. Performance: qlen=kvlen 8192 and 16384, FP16 and BF16
     - FA2 non-paged (non-causal / causal)
     - FA2 paged (non-causal)
     - ESIMD SDP kernels from vllm-kernel-custom (non-causal, non-paged)
     - torch SDPA reference (non-causal)
     All TFLOPS use full-attention FLOP count (4*H*q*kv*D).
     Causal kernels show adjusted TFLOPS (÷2) since they skip ~half the work.

FA2 kernel input layout: [total_tokens, num_heads, head_dim]
  - cu_seqlens_q: [batch_size+1] cumulative query lengths
  - cu_seqlens_k: [batch_size+1] cumulative KV lengths (non-paged)
  - seqused_k:   [batch_size] per-seq KV length (paged)
  - block_table:  [batch_size, max_blocks] page table (paged)

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_vllm_fa2_paged.py
"""
import time

import torch
import torch.nn.functional as F

device = torch.device("xpu")

D = 128  # head_dim
BLOCK_SIZE = 64  # XPU FA2 paged KV cache block size


def rel_rms(a, b):
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    return ((a_f - b_f).pow(2).mean().sqrt() / (b_f.pow(2).mean().sqrt() + 1e-8)).item()


def sdp_flops(H, q_len, kv_len, D):
    """Standard full-attention FLOPs: 4 * H * q_len * kv_len * D (B=1)."""
    return 4 * H * q_len * kv_len * D


def torch_sdpa_ref(Q_lhd, K_lhd, V_lhd, is_causal=False):
    """PyTorch SDPA reference. Input [L, H, D] -> output [L, H, D].
    Handles GQA by expanding K/V heads."""
    H_q, H_kv = Q_lhd.shape[1], K_lhd.shape[1]
    if H_kv != H_q:
        K_lhd = K_lhd.repeat_interleave(H_q // H_kv, dim=1)
        V_lhd = V_lhd.repeat_interleave(H_q // H_kv, dim=1)
    Q_bhld = Q_lhd.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    K_bhld = K_lhd.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    V_bhld = V_lhd.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    ref = F.scaled_dot_product_attention(Q_bhld, K_bhld, V_bhld, is_causal=is_causal)
    return ref.permute(0, 2, 1, 3).squeeze(0).contiguous()


# ============================================================
# FA2 wrapper helpers
# ============================================================

def fa2_non_paged(q, k, v, causal=False, out=None):
    """Call FA2 without paging: use cu_seqlens_k path.
    q: [total_q, H_q, D], k: [total_k, H_kv, D], v: [total_k, H_kv, D].
    Single sequence (batch=1).
    """
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

    q_len = q.shape[0]
    kv_len = k.shape[0]
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

    if out is None:
        out = torch.empty_like(q)

    result = flash_attn_varlen_func(
        q=q, k=k, v=v,
        out=out,
        max_seqlen_q=q_len,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=kv_len,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
    )
    return result


def fa2_paged(q, key_cache, value_cache, block_table, seqused_k, causal=False, out=None):
    """Call FA2 with paged KV cache.
    q: [total_q, H_q, D]
    key_cache:   [num_blocks, block_size, H_kv, D] (NHD layout)
    value_cache: [num_blocks, block_size, H_kv, D] (NHD layout)
    block_table: [batch, max_blocks_per_seq]
    seqused_k: [batch]
    """
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

    q_len = q.shape[0]
    max_kv_len = int(seqused_k.max().item())
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)

    if out is None:
        out = torch.empty_like(q)

    result = flash_attn_varlen_func(
        q=q, k=key_cache, v=value_cache,
        out=out,
        max_seqlen_q=q_len,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_kv_len,
        seqused_k=seqused_k,
        block_table=block_table,
        causal=causal,
    )
    return result


def build_paged_kv(K_flat, V_flat, kv_len, H_kv, block_size=BLOCK_SIZE, extra_blocks=0):
    """Build paged KV cache from flat K/V tensors.
    Returns (key_cache, value_cache, block_table, seqused_k) for batch=1.
    """
    num_blocks_needed = (kv_len + block_size - 1) // block_size
    total_blocks = num_blocks_needed + extra_blocks
    dtype = K_flat.dtype

    key_cache = torch.zeros(total_blocks, block_size, H_kv, D,
                            dtype=dtype, device=device)
    value_cache = torch.zeros(total_blocks, block_size, H_kv, D,
                              dtype=dtype, device=device)

    if extra_blocks > 0:
        perm = torch.randperm(total_blocks)[:num_blocks_needed]
    else:
        perm = torch.arange(num_blocks_needed)

    for i in range(num_blocks_needed):
        start = i * block_size
        end = min(start + block_size, kv_len)
        length = end - start
        page_idx = perm[i].item()
        key_cache[page_idx, :length] = K_flat[start:end]
        value_cache[page_idx, :length] = V_flat[start:end]

    block_table = perm.to(torch.int32).unsqueeze(0).to(device)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)
    return key_cache, value_cache, block_table, seqused_k


# ============================================================
# Correctness tests
# ============================================================

def test_fa2_non_paged_causal():
    """FA2 non-paged causal vs PyTorch SDPA reference."""
    H_q, H_kv = 32, 32

    for q_len, kv_len, label in [(64, 64, "64x64"), (256, 256, "256x256")]:
        torch.manual_seed(42)
        Q = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
        K = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        V = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)

        with torch.no_grad():
            out = fa2_non_paged(Q, K, V, causal=True)
            torch.xpu.synchronize()
            ref = torch_sdpa_ref(Q, K, V, is_causal=True)

        err = rel_rms(out, ref)
        assert err < 0.02, f"[causal {label}] rel_rms={err:.4f} > 0.02"
        print(f"[PASS] fa2_non_paged causal [{label}] — rel_rms={err:.4f}")


def test_fa2_non_paged_non_causal():
    """FA2 non-paged non-causal (full attention) vs PyTorch SDPA reference."""
    H_q, H_kv = 32, 32

    for q_len, kv_len, label in [(64, 64, "64x64"), (256, 256, "256x256")]:
        torch.manual_seed(42)
        Q = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
        K = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        V = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)

        with torch.no_grad():
            out = fa2_non_paged(Q, K, V, causal=False)
            torch.xpu.synchronize()
            ref = torch_sdpa_ref(Q, K, V, is_causal=False)

        err = rel_rms(out, ref)
        assert err < 0.02, f"[non-causal {label}] rel_rms={err:.4f} > 0.02"
        print(f"[PASS] fa2_non_paged non-causal [{label}] — rel_rms={err:.4f}")


def test_fa2_non_paged_bf16():
    """FA2 non-paged BF16 vs PyTorch SDPA reference."""
    H_q, H_kv = 32, 32

    for q_len, kv_len, label in [(64, 64, "64x64"), (256, 256, "256x256")]:
        torch.manual_seed(42)
        Q = torch.randn(q_len, H_q, D, dtype=torch.bfloat16, device=device)
        K = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        V = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            out = fa2_non_paged(Q, K, V, causal=False)
            torch.xpu.synchronize()
            ref = torch_sdpa_ref(Q, K, V, is_causal=False)

        err = rel_rms(out, ref)
        assert err < 0.05, f"[bf16 non-causal {label}] rel_rms={err:.4f} > 0.05"
        print(f"[PASS] fa2_non_paged bf16 non-causal [{label}] — rel_rms={err:.4f}")


def test_fa2_paged_decode():
    """FA2 paged decode: fill KV cache into pages, query with block_table."""
    H_q, H_kv = 32, 32
    kv_len = 256
    q_len = 1  # decode: single token query

    torch.manual_seed(42)
    Q = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
    K_flat = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
    V_flat = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)

    key_cache, value_cache, block_table, seqused_k = build_paged_kv(
        K_flat, V_flat, kv_len, H_kv)

    with torch.no_grad():
        out = fa2_paged(Q, key_cache, value_cache, block_table, seqused_k, causal=False)
        torch.xpu.synchronize()
        ref = fa2_non_paged(Q, K_flat, V_flat, causal=False)
        torch.xpu.synchronize()

    err = rel_rms(out, ref)
    assert err < 0.01, f"[paged decode] rel_rms={err:.4f} > 0.01"
    print(f"[PASS] fa2_paged_decode — kv_len={kv_len}, rel_rms={err:.4f}")


def test_fa2_paged_decode_shuffled():
    """FA2 paged decode with shuffled page order."""
    H_q, H_kv = 32, 32
    kv_len = 512
    q_len = 1

    torch.manual_seed(123)
    Q = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
    K_flat = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
    V_flat = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)

    key_cache, value_cache, block_table, seqused_k = build_paged_kv(
        K_flat, V_flat, kv_len, H_kv, extra_blocks=8)

    with torch.no_grad():
        out = fa2_paged(Q, key_cache, value_cache, block_table, seqused_k, causal=False)
        torch.xpu.synchronize()
        ref = fa2_non_paged(Q, K_flat, V_flat, causal=False)
        torch.xpu.synchronize()

    err = rel_rms(out, ref)
    assert err < 0.01, f"[paged decode shuffled] rel_rms={err:.4f} > 0.01"
    print(f"[PASS] fa2_paged_decode_shuffled — kv_len={kv_len}, rel_rms={err:.4f}")


# ============================================================
# Benchmark helper
# ============================================================

def _bench_one(label, fn, flops, warmup, iters, causal=False):
    """Run warmup+timed loop, return (ms, tflops). causal halves effective flops."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - t0
    ms = elapsed / iters * 1e3
    effective_flops = flops / 2 if causal else flops
    tflops = effective_flops / (elapsed / iters) / 1e12
    return ms, tflops


# ============================================================
# Performance benchmarks
# ============================================================

def bench_sdp(H_q, H_kv, title):
    """Benchmark FA2 (non-paged + paged) vs ESIMD SDP vs torch SDPA.
    All TFLOPS are effective: causal kernels use flops/2.
    """
    WARMUP = 20
    ITERS = 100

    PERF_SHAPES = [
        (8192, 8192, "8Kx8K"),
        (16384, 16384, "16Kx16K"),
    ]

    # Try loading ESIMD SDP kernels
    esimd_available = False
    try:
        from vllm_kernel_custom import esimd_sdp_fp16, esimd_sdp_bf16io
        esimd_available = True
    except ImportError:
        print("  (vllm_kernel_custom not available, skipping ESIMD comparison)")

    print("=" * 100)
    print(f"{title}  B=1  H_q={H_q}  H_kv={H_kv}  D={D}  "
          f"warmup={WARMUP}  iters={ITERS}")
    print(f"  TFLOPS = effective (causal uses flops/2, non-causal uses full flops)")
    print(f"{'Kernel':<32s} {'Path':<10s} {'Shape':<10s} {'Time (ms)':>10s} {'TFLOPS':>8s}")
    print("-" * 100)

    for q_len, kv_len, label in PERF_SHAPES:
        flops = sdp_flops(H_q, q_len, kv_len, D)

        torch.manual_seed(42)
        Q_fp16 = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
        K_fp16 = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        V_fp16 = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        out_fp16 = torch.empty_like(Q_fp16)

        Q_bf16 = torch.randn(q_len, H_q, D, dtype=torch.bfloat16, device=device)
        K_bf16 = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        V_bf16 = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        out_bf16 = torch.empty_like(Q_bf16)

        # Build paged KV cache for FA2 paged benchmarks
        kc_fp16, vc_fp16, bt, suk = build_paged_kv(K_fp16, V_fp16, kv_len, H_kv)
        out_paged_fp16 = torch.empty_like(Q_fp16)
        kc_bf16, vc_bf16, bt_bf, suk_bf = build_paged_kv(K_bf16, V_bf16, kv_len, H_kv)
        out_paged_bf16 = torch.empty_like(Q_bf16)

        # --- FA2 FP16 non-causal, non-paged ---
        ms, tf = _bench_one("fa2_fp16",
            lambda: fa2_non_paged(Q_fp16, K_fp16, V_fp16, causal=False, out=out_fp16),
            flops, WARMUP, ITERS)
        print(f"{'fa2_fp16_non_causal':<32s} {'flat':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        # --- FA2 FP16 causal, non-paged ---
        ms, tf = _bench_one("fa2_fp16_causal",
            lambda: fa2_non_paged(Q_fp16, K_fp16, V_fp16, causal=True, out=out_fp16),
            flops, WARMUP, ITERS, causal=True)
        print(f"{'fa2_fp16_causal':<32s} {'flat':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        # --- FA2 FP16 non-causal, paged ---
        ms, tf = _bench_one("fa2_fp16_paged",
            lambda: fa2_paged(Q_fp16, kc_fp16, vc_fp16, bt, suk, causal=False, out=out_paged_fp16),
            flops, WARMUP, ITERS)
        print(f"{'fa2_fp16_paged_non_causal':<32s} {'paged':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        # --- FA2 BF16 non-causal, non-paged ---
        ms, tf = _bench_one("fa2_bf16",
            lambda: fa2_non_paged(Q_bf16, K_bf16, V_bf16, causal=False, out=out_bf16),
            flops, WARMUP, ITERS)
        print(f"{'fa2_bf16_non_causal':<32s} {'flat':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        # --- FA2 BF16 non-causal, paged ---
        ms, tf = _bench_one("fa2_bf16_paged",
            lambda: fa2_paged(Q_bf16, kc_bf16, vc_bf16, bt_bf, suk_bf, causal=False, out=out_paged_bf16),
            flops, WARMUP, ITERS)
        print(f"{'fa2_bf16_paged_non_causal':<32s} {'paged':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        # --- ESIMD SDP (vllm-kernel-custom, non-causal, non-paged) ---
        if esimd_available:
            norm_alpha = torch.ones(H_q, D, dtype=torch.float32, device=device)

            out_esimd = torch.empty_like(Q_fp16)
            ms, tf = _bench_one("esimd_sdp_fp16",
                lambda: esimd_sdp_fp16(Q_fp16, K_fp16, V_fp16, norm_alpha, out_esimd,
                                       q_len, kv_len, H_q, H_kv),
                flops, WARMUP, ITERS)
            print(f"{'esimd_sdp_fp16':<32s} {'flat':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

            out_esimd_bf = torch.empty_like(Q_bf16)
            ms, tf = _bench_one("esimd_sdp_bf16io",
                lambda: esimd_sdp_bf16io(Q_bf16, K_bf16, V_bf16, norm_alpha, out_esimd_bf,
                                         q_len, kv_len, H_q, H_kv),
                flops, WARMUP, ITERS)
            print(f"{'esimd_sdp_bf16io':<32s} {'flat':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        # --- torch SDPA FP16 (non-causal) ---
        if H_kv != H_q:
            K_exp = K_fp16.repeat_interleave(H_q // H_kv, dim=1)
            V_exp = V_fp16.repeat_interleave(H_q // H_kv, dim=1)
        else:
            K_exp, V_exp = K_fp16, V_fp16
        Q_bhld = Q_fp16.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        K_bhld = K_exp.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        V_bhld = V_exp.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        ms, tf = _bench_one("torch_sdpa_fp16",
            lambda: F.scaled_dot_product_attention(Q_bhld, K_bhld, V_bhld, is_causal=False),
            flops, WARMUP, ITERS)
        print(f"{'torch_sdpa_fp16':<32s} {'flat':<10s} {label:<10s} {ms:10.3f} {tf:8.2f}")

        print("-" * 100)

    print()


if __name__ == "__main__":
    print("=" * 70)
    print("vllm-xpu-kernels: FA2 Paged Attention Tests (libattn_kernels_xe_2)")
    print("=" * 70)

    print("\n--- Correctness tests ---")
    test_fa2_non_paged_causal()
    test_fa2_non_paged_non_causal()
    test_fa2_non_paged_bf16()
    test_fa2_paged_decode()
    test_fa2_paged_decode_shuffled()

    print("\n--- Performance benchmarks (MHA: H_q=32, H_kv=32) ---")
    bench_sdp(32, 32, "MHA Performance")

    print("\n--- Performance benchmarks (GQA: H_q=32, H_kv=2) ---")
    bench_sdp(32, 2, "GQA Performance")

    print("=" * 70)
    print("ALL FA2 TESTS PASSED")
    print("=" * 70)
