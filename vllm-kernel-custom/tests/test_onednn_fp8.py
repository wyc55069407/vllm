"""
Test oneDNN FP8 GEMM kernel (onednn_w8a16_fp8).

FP16/BF16 activations x FP8_E4M3 weights via oneDNN matmul primitive.
Per-N (per-output-channel) scaling.

Wrapper: onednn_w8a16_fp8(x, weight, scales, bias, output, M, N, K, has_bias)

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_onednn_fp8.py
"""
import torch

device = torch.device("xpu")

FP8_MAX = 448.0


def quant_fp8_per_n(weight):
    """Per-output-channel absmax quantisation -> FP8 E4M3.
    weight: [N, K] any float -> qweight: [N, K] fp8, scales: [N] fp32.
    """
    w_f32 = weight.float()
    max_vals = w_f32.abs().max(dim=1).values  # [N]
    scales = (max_vals / FP8_MAX).clamp(min=1e-12)  # [N]
    qweight = (w_f32 / scales.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return qweight, scales.float()


def rel_rms(a, b):
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    return ((a_f - b_f).pow(2).mean().sqrt() / (b_f.pow(2).mean().sqrt() + 1e-8)).item()


def test_onednn_fp8_fp16_no_bias():
    """FP16 x FP8 GEMM without bias."""
    from vllm_kernel_custom import onednn_w8a16_fp8

    M, N, K = 4, 1024, 1024

    weight = torch.randn(N, K, dtype=torch.float16)
    qweight, scales = quant_fp8_per_n(weight)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    qweight_xpu = qweight.to(device)
    scales_xpu = scales.to(device)
    bias = torch.zeros(N, dtype=torch.float16, device=device)
    output = torch.zeros(M, N, dtype=torch.float16, device=device)

    onednn_w8a16_fp8(x, qweight_xpu, scales_xpu, bias, output, M, N, K, 0)

    # Reference: dequant + matmul
    weight_xpu = weight.to(device)
    ref = torch.nn.functional.linear(x, weight_xpu)

    err = rel_rms(output, ref)
    assert err < 0.05, f"FP16 no-bias rel_rms too high: {err:.4f}"
    print(f"[PASS] test_onednn_fp8_fp16_no_bias — M={M}, N={N}, K={K}, rel_rms={err:.4f}")


def test_onednn_fp8_fp16_with_bias():
    """FP16 x FP8 GEMM with bias."""
    from vllm_kernel_custom import onednn_w8a16_fp8

    M, N, K = 4, 512, 512

    weight = torch.randn(N, K, dtype=torch.float16)
    qweight, scales = quant_fp8_per_n(weight)

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    qweight_xpu = qweight.to(device)
    scales_xpu = scales.to(device)
    bias = torch.randn(N, dtype=torch.float16, device=device)
    output = torch.zeros(M, N, dtype=torch.float16, device=device)

    onednn_w8a16_fp8(x, qweight_xpu, scales_xpu, bias, output, M, N, K, 1)

    weight_xpu = weight.to(device)
    ref = torch.nn.functional.linear(x, weight_xpu, bias)

    err = rel_rms(output, ref)
    assert err < 0.08, f"FP16 with-bias rel_rms too high: {err:.4f}"
    print(f"[PASS] test_onednn_fp8_fp16_with_bias — M={M}, N={N}, K={K}, rel_rms={err:.4f}")


def test_onednn_fp8_large_shapes():
    """FP16 x FP8 GEMM with typical LLM shapes."""
    from vllm_kernel_custom import onednn_w8a16_fp8

    shapes = [
        (1, 4096, 4096, "attn to_q/k/v"),
        (4, 10240, 4096, "FFN up-proj"),
        (4, 4096, 10240, "FFN down-proj"),
    ]

    for M, N, K, label in shapes:
        weight = torch.randn(N, K, dtype=torch.float16)
        qweight, scales = quant_fp8_per_n(weight)

        x = torch.randn(M, K, dtype=torch.float16, device=device)
        qweight_xpu = qweight.to(device)
        scales_xpu = scales.to(device)
        bias = torch.zeros(N, dtype=torch.float16, device=device)
        output = torch.zeros(M, N, dtype=torch.float16, device=device)

        onednn_w8a16_fp8(x, qweight_xpu, scales_xpu, bias, output, M, N, K, 0)

        weight_xpu = weight.to(device)
        ref = torch.nn.functional.linear(x, weight_xpu)

        err = rel_rms(output, ref)
        assert err < 0.05, f"[{label}] rel_rms too high: {err:.4f}"
        print(f"[PASS] test_onednn_fp8_large_shapes [{label}] — M={M}, N={N}, K={K}, rel_rms={err:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: oneDNN FP8 GEMM Tests")
    print("=" * 60)
    test_onednn_fp8_fp16_no_bias()
    test_onednn_fp8_fp16_with_bias()
    test_onednn_fp8_large_shapes()
    print("=" * 60)
    print("ALL ONEDNN_FP8 TESTS PASSED")
    print("=" * 60)
