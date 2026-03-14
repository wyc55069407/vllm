"""
Test esimd_gemv_fp8 kernel.

FP8 weight GEMV: output = input @ dequant(weight_fp8, scale) + bias
M must be <= 8 (GEMV, not GEMM).

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_gemv_fp8.py
"""
import torch

device = torch.device("xpu")


def test_gemv_fp8_no_bias():
    """FP8 GEMV without bias: output = input @ dequant(weight_fp8)."""
    from vllm_kernel_custom import esimd_gemv_fp8

    M = 1       # single token (GEMV)
    N = 1024    # output dim
    K = 1024    # input dim
    batch = 1
    block_n = 128
    block_k = 128

    # Create reference weight, quantize to fp8
    weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
    weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
    weight_scale = torch.ones(N // block_n, K // block_k, dtype=torch.float16, device=device)

    input_t = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    bias = torch.zeros(N, dtype=torch.float16, device=device)
    output = torch.zeros(M, N, dtype=torch.float16, device=device)

    esimd_gemv_fp8(input_t, weight_fp8, weight_scale, bias, output,
                   M, N, K, batch, block_n, block_k, 0)

    # Reference: dequant fp8 to fp16 (scale=1), then matmul
    weight_dequant = weight_fp8.to(torch.float16)
    ref = input_t.float() @ weight_dequant.float().T

    torch.testing.assert_close(output.float(), ref.half().float(), rtol=0.1, atol=0.05)
    print(f"[PASS] test_gemv_fp8_no_bias — M={M}, N={N}, K={K}")


def test_gemv_fp8_with_bias():
    """FP8 GEMV with bias."""
    from vllm_kernel_custom import esimd_gemv_fp8

    M = 1
    N = 512
    K = 512
    batch = 1
    block_n = 128
    block_k = 128

    weight_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
    weight_fp8 = weight_ref.to(torch.float8_e4m3fn)
    weight_scale = torch.ones(N // block_n, K // block_k, dtype=torch.float16, device=device)

    input_t = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    bias = torch.randn(N, dtype=torch.float16, device=device) * 0.05
    output = torch.zeros(M, N, dtype=torch.float16, device=device)

    esimd_gemv_fp8(input_t, weight_fp8, weight_scale, bias, output,
                   M, N, K, batch, block_n, block_k, 1)

    weight_dequant = weight_fp8.to(torch.float16)
    ref = (input_t.float() @ weight_dequant.float().T + bias.float())

    torch.testing.assert_close(output.float(), ref.half().float(), rtol=0.15, atol=0.1)
    print(f"[PASS] test_gemv_fp8_with_bias — M={M}, N={N}, K={K}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: GEMV FP8 Tests")
    print("=" * 60)
    test_gemv_fp8_no_bias()
    test_gemv_fp8_with_bias()
    print("=" * 60)
    print("ALL GEMV_FP8 TESTS PASSED")
    print("=" * 60)
