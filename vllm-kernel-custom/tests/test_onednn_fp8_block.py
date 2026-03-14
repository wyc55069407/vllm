"""
Test oneDNN FP8 block-scale GEMM (onednn_w8a16_fp8_block).

Supports two scale formats:
  1. [K/block_k, N] — K-grouped, per-N (native oneDNN path)
  2. [K/block_k, N/block_n] — 2D block (auto-expanded to [K/block_k, N])

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_onednn_fp8_block.py
"""
import torch

device = torch.device("xpu")
FP8_MAX = 448.0


def quant_fp8_block_1d(weight, block_k=128):
    """K-grouped, per-N: scales [K/bk, N]."""
    N, K = weight.shape
    w_f32 = weight.float()
    scale_k = (K + block_k - 1) // block_k
    scales = torch.zeros(scale_k, N, dtype=torch.float32)
    for ki in range(scale_k):
        k_s, k_e = ki * block_k, min((ki + 1) * block_k, K)
        block = w_f32[:, k_s:k_e]
        scales[ki, :] = (block.abs().max(dim=1).values / FP8_MAX).clamp(min=1e-12)
    qweight = torch.zeros_like(w_f32)
    for ki in range(scale_k):
        k_s, k_e = ki * block_k, min((ki + 1) * block_k, K)
        qweight[:, k_s:k_e] = (w_f32[:, k_s:k_e] / scales[ki].unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX)
    return qweight.to(torch.float8_e4m3fn), scales


def quant_fp8_block_2d(weight, block_k=128, block_n=128):
    """2D block: scales [K/bk, N/bn]."""
    N, K = weight.shape
    w_f32 = weight.float()
    scale_k = (K + block_k - 1) // block_k
    scale_n = (N + block_n - 1) // block_n
    scales = torch.zeros(scale_k, scale_n, dtype=torch.float32)
    for ki in range(scale_k):
        for ni in range(scale_n):
            k_s, k_e = ki * block_k, min((ki + 1) * block_k, K)
            n_s, n_e = ni * block_n, min((ni + 1) * block_n, N)
            max_val = w_f32[n_s:n_e, k_s:k_e].abs().max().item()
            scales[ki, ni] = max(max_val / FP8_MAX, 1e-12)
    qweight = torch.zeros_like(w_f32)
    for ki in range(scale_k):
        for ni in range(scale_n):
            k_s, k_e = ki * block_k, min((ki + 1) * block_k, K)
            n_s, n_e = ni * block_n, min((ni + 1) * block_n, N)
            qweight[n_s:n_e, k_s:k_e] = (
                w_f32[n_s:n_e, k_s:k_e] / scales[ki, ni]
            ).clamp(-FP8_MAX, FP8_MAX)
    return qweight.to(torch.float8_e4m3fn), scales


def rel_rms(a, b):
    a_f, b_f = a.float().cpu(), b.float().cpu()
    return ((a_f - b_f).pow(2).mean().sqrt() / (b_f.pow(2).mean().sqrt() + 1e-8)).item()


def run_test(M, N, K, block_k, block_n, use_2d, label):
    from vllm_kernel_custom import onednn_w8a16_fp8_block

    weight = torch.randn(N, K, dtype=torch.float16)
    if use_2d:
        qweight, scales = quant_fp8_block_2d(weight, block_k, block_n)
    else:
        qweight, scales = quant_fp8_block_1d(weight, block_k)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    bias = torch.zeros(N, dtype=torch.float16, device=device)
    output = torch.zeros(M, N, dtype=torch.float16, device=device)

    onednn_w8a16_fp8_block(
        x, qweight.to(device), scales.to(device), bias, output,
        M, N, K, block_k, block_n, 0
    )

    ref = torch.nn.functional.linear(x, weight.to(device))
    err = rel_rms(output, ref)
    mode = "2D" if use_2d else "1D"
    status = "PASS" if err < 0.05 else "FAIL"
    print(f"[{status}] {label} ({mode} scales {list(scales.shape)}) — "
          f"M={M}, N={N}, K={K}, rel_rms={err:.4f}")
    assert err < 0.05, f"rel_rms too high: {err:.4f}"


if __name__ == "__main__":
    print("=" * 70)
    print("vllm-kernel-custom: oneDNN FP8 Block-Scale GEMM Tests")
    print("=" * 70)

    # 1D scales [K/bk, N] — native path
    run_test(4, 1024, 1024, 128, 128, False, "basic_1d")
    run_test(1, 4096, 4096, 128, 128, False, "attn_1d")
    run_test(4, 10240, 4096, 128, 128, False, "ffn_up_1d")

    # 2D scales [K/bk, N/bn] — auto-expanded
    run_test(4, 1024, 1024, 128, 128, True, "basic_2d")
    run_test(1, 4096, 4096, 128, 128, True, "attn_2d")
    run_test(4, 10240, 4096, 128, 128, True, "ffn_up_2d")
    run_test(4, 4096, 10240, 128, 128, True, "ffn_down_2d")
    run_test(4, 4000, 4096, 128, 128, True, "non_div_N_2d")

    print("=" * 70)
    print("ALL BLOCK FP8 TESTS PASSED")
    print("=" * 70)
