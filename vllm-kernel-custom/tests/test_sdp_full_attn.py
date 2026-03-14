"""
Test ESIMD Flash Attention SDP kernels (non-causal).

Three variants compiled with doubleGRF (LGRF):
  - sdp_fp16: FP16 optimized, barrier+interleave optimization
  - sdp_bf16: BF16 pure, BF16 DPAS for both QK and SxV
  - sdp_bf16io: BF16 I/O hybrid, BF16 DPAS for QK, FP16 DPAS for SxV (fastest)

Kernel input layout: [L, H, D] contiguous (squeezed from [B=1, L, H, D])
head_dim = 128, hardcoded scale = 1/sqrt(128)
normAlpha: float32[H, 128], set to 1.0 for standard attention

Reference: PyTorch scaled_dot_product_attention on XPU
  SDPA layout: [B, H, L, D]

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_sdp_full_attn.py
"""
import torch
import torch.nn.functional as F

device = torch.device("xpu")

D = 128  # head_dim, hardcoded


def rel_rms(a, b):
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    return ((a_f - b_f).pow(2).mean().sqrt() / (b_f.pow(2).mean().sqrt() + 1e-8)).item()


def test_sdp_fp16_correctness():
    """FP16 SDP vs PyTorch SDPA reference."""
    from vllm_kernel_custom import esimd_sdp_fp16

    H_q = 32
    H_kv = 32

    for q_len, kv_len, label in [(512, 512, "512x512"), (1024, 1024, "1Kx1K")]:
        torch.manual_seed(42)
        # Kernel layout: [L, H, D]
        Q = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
        K = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        V = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        norm_alpha = torch.ones(H_q, D, dtype=torch.float32, device=device)
        output = torch.empty_like(Q)

        with torch.no_grad():
            esimd_sdp_fp16(Q, K, V, norm_alpha, output, q_len, kv_len, H_q, H_kv)
            torch.xpu.synchronize()

            # Reference: [1, H, L, D]
            Q_ref = Q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            K_ref = K.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            V_ref = V.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=False)
            ref_lhd = ref.permute(0, 2, 1, 3).squeeze(0).contiguous()

        err = rel_rms(output, ref_lhd)
        assert err < 0.05, f"[fp16 {label}] rel_rms={err:.4f} > 0.05"
        print(f"[PASS] test_sdp_fp16 [{label}] — q_len={q_len}, kv_len={kv_len}, rel_rms={err:.4f}")


def test_sdp_bf16_correctness():
    """BF16 SDP vs PyTorch SDPA reference."""
    from vllm_kernel_custom import esimd_sdp_bf16

    H_q = 32
    H_kv = 32

    for q_len, kv_len, label in [(512, 512, "512x512"), (1024, 1024, "1Kx1K")]:
        torch.manual_seed(42)
        Q = torch.randn(q_len, H_q, D, dtype=torch.bfloat16, device=device)
        K = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        V = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        norm_alpha = torch.ones(H_q, D, dtype=torch.float32, device=device)
        output = torch.empty_like(Q)

        with torch.no_grad():
            esimd_sdp_bf16(Q, K, V, norm_alpha, output, q_len, kv_len, H_q, H_kv)
            torch.xpu.synchronize()

            Q_ref = Q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            K_ref = K.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            V_ref = V.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=False)
            ref_lhd = ref.permute(0, 2, 1, 3).squeeze(0).contiguous()

        err = rel_rms(output, ref_lhd)
        assert err < 0.05, f"[bf16 {label}] rel_rms={err:.4f} > 0.05"
        print(f"[PASS] test_sdp_bf16 [{label}] — q_len={q_len}, kv_len={kv_len}, rel_rms={err:.4f}")


def test_sdp_bf16io_correctness():
    """BF16io SDP vs PyTorch SDPA reference."""
    from vllm_kernel_custom import esimd_sdp_bf16io

    H_q = 32
    H_kv = 32

    for q_len, kv_len, label in [(512, 512, "512x512"), (1024, 1024, "1Kx1K")]:
        torch.manual_seed(42)
        Q = torch.randn(q_len, H_q, D, dtype=torch.bfloat16, device=device)
        K = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        V = torch.randn(kv_len, H_kv, D, dtype=torch.bfloat16, device=device)
        norm_alpha = torch.ones(H_q, D, dtype=torch.float32, device=device)
        output = torch.empty_like(Q)

        with torch.no_grad():
            esimd_sdp_bf16io(Q, K, V, norm_alpha, output, q_len, kv_len, H_q, H_kv)
            torch.xpu.synchronize()

            Q_ref = Q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            K_ref = K.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            V_ref = V.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=False)
            ref_lhd = ref.permute(0, 2, 1, 3).squeeze(0).contiguous()

        err = rel_rms(output, ref_lhd)
        assert err < 0.05, f"[bf16io {label}] rel_rms={err:.4f} > 0.05"
        print(f"[PASS] test_sdp_bf16io [{label}] — q_len={q_len}, kv_len={kv_len}, rel_rms={err:.4f}")


def sdp_flops(H, q_len, kv_len, D):
    """Standard attention FLOPs: 4 * H * q_len * kv_len * D (B=1)."""
    return 4 * H * q_len * kv_len * D


def bench_sdp():
    """Performance benchmark: 20 warmup + 100 iters, sync only at end."""
    import time
    from vllm_kernel_custom import esimd_sdp_fp16, esimd_sdp_bf16, esimd_sdp_bf16io

    B = 1
    H_q = 32
    H_kv = 32
    WARMUP = 20
    ITERS = 100

    PERF_SHAPES = [
        (8192, 8192, "8Kx8K"),
        (8192, 16384, "8Kx16K"),
    ]

    kernels = [
        ("fp16",   torch.float16,  esimd_sdp_fp16),
        ("bf16",   torch.bfloat16, esimd_sdp_bf16),
        ("bf16io", torch.bfloat16, esimd_sdp_bf16io),
    ]

    # Also benchmark torch SDPA as reference
    def torch_sdpa_fn(Q_bhld, K_bhld, V_bhld):
        return F.scaled_dot_product_attention(Q_bhld, K_bhld, V_bhld, is_causal=False)

    print("=" * 80)
    print(f"Performance benchmark  B={B} H={H_q} D={D}  warmup={WARMUP} iters={ITERS}")
    print("=" * 80)

    for q_len, kv_len, label in PERF_SHAPES:
        flops = sdp_flops(H_q, q_len, kv_len, D)
        print(f"\n  --- {label}  q_len={q_len}  kv_len={kv_len} ---")

        for kname, dtype, kernel_fn in kernels:
            torch.manual_seed(42)
            Q = torch.randn(q_len, H_q, D, dtype=dtype, device=device)
            K = torch.randn(kv_len, H_kv, D, dtype=dtype, device=device)
            V = torch.randn(kv_len, H_kv, D, dtype=dtype, device=device)
            norm_alpha = torch.ones(H_q, D, dtype=torch.float32, device=device)
            output = torch.empty_like(Q)

            with torch.no_grad():
                # warmup
                for _ in range(WARMUP):
                    kernel_fn(Q, K, V, norm_alpha, output, q_len, kv_len, H_q, H_kv)
                torch.xpu.synchronize()

                # timed
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    kernel_fn(Q, K, V, norm_alpha, output, q_len, kv_len, H_q, H_kv)
                torch.xpu.synchronize()
                elapsed = time.perf_counter() - t0

            ms = elapsed / ITERS * 1e3
            tflops = flops / (elapsed / ITERS) / 1e12
            print(f"    esimd_{kname:6s}  {ms:8.3f} ms    {tflops:7.2f} TFLOPS")

        # torch SDPA FP16 reference
        torch.manual_seed(42)
        Q_fp16 = torch.randn(q_len, H_q, D, dtype=torch.float16, device=device)
        K_fp16 = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        V_fp16 = torch.randn(kv_len, H_kv, D, dtype=torch.float16, device=device)
        Q_bhld = Q_fp16.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        K_bhld = K_fp16.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
        V_bhld = V_fp16.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        with torch.no_grad():
            for _ in range(WARMUP):
                torch_sdpa_fn(Q_bhld, K_bhld, V_bhld)
            torch.xpu.synchronize()

            t0 = time.perf_counter()
            for _ in range(ITERS):
                torch_sdpa_fn(Q_bhld, K_bhld, V_bhld)
            torch.xpu.synchronize()
            elapsed = time.perf_counter() - t0

        ms = elapsed / ITERS * 1e3
        tflops = flops / (elapsed / ITERS) / 1e12
        print(f"    torch_sdpa    {ms:8.3f} ms    {tflops:7.2f} TFLOPS  (FP16 ref)")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: Full Attention SDP Tests (LGRF)")
    print("=" * 60)
    test_sdp_fp16_correctness()
    test_sdp_bf16_correctness()
    test_sdp_bf16io_correctness()
    print()
    bench_sdp()
    print("=" * 60)
    print("ALL SDP FULL ATTENTION TESTS PASSED")
    print("=" * 60)
