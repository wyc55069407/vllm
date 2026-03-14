"""
Test esimd_fp8_dequant kernel.

Dequantizes FP8 E4M3 weights to FP16 using per-block scales:
  output[i,j] = fp8_to_float(weight_fp8[i,j]) * scale[i//block_n, j//block_k]

Block size is hardcoded at 128x128.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_fp8_dequant.py
"""
import torch

device = torch.device("xpu")


def test_fp8_dequant_basic():
    """Dequantize FP8 weights and verify against reference."""
    from vllm_kernel_custom import esimd_fp8_dequant

    N = 1024
    K = 1024
    block_n = 128
    block_k = 128

    # Create reference fp16 weight, quantize to fp8, then dequant and compare
    weight_fp16_ref = torch.randn(N, K, dtype=torch.float16, device=device) * 0.5

    # Quantize to FP8 E4M3
    weight_fp8 = weight_fp16_ref.to(torch.float8_e4m3fn)

    # Compute per-block scales: max(abs(block)) / fp8_max
    scale_n = N // block_n
    scale_k = K // block_k
    scale = torch.ones(scale_n, scale_k, dtype=torch.float16, device=device)

    output = torch.zeros(N, K, dtype=torch.float16, device=device)

    esimd_fp8_dequant(weight_fp8, scale, output, N, K, block_n, block_k)

    # Reference: simply cast fp8 back to fp16 (scale=1.0)
    ref = weight_fp8.to(torch.float16)

    torch.testing.assert_close(output.float(), ref.float(), rtol=5e-2, atol=1e-2)
    print(f"[PASS] test_fp8_dequant_basic — N={N}, K={K}")


def test_fp8_dequant_with_scale():
    """Dequantize FP8 with non-trivial scales."""
    from vllm_kernel_custom import esimd_fp8_dequant

    N = 512
    K = 512
    block_n = 128
    block_k = 128

    # Small values that fit well in FP8
    weight_fp16 = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1
    weight_fp8 = weight_fp16.to(torch.float8_e4m3fn)

    scale_n = N // block_n
    scale_k = K // block_k
    scale = torch.rand(scale_n, scale_k, dtype=torch.float16, device=device) * 2.0 + 0.5

    output = torch.zeros(N, K, dtype=torch.float16, device=device)

    esimd_fp8_dequant(weight_fp8, scale, output, N, K, block_n, block_k)

    # Reference: fp8_to_fp16 * block_scale
    ref = weight_fp8.to(torch.float16)
    for bn in range(scale_n):
        for bk in range(scale_k):
            ref[bn * block_n:(bn + 1) * block_n,
                bk * block_k:(bk + 1) * block_k] *= scale[bn, bk]

    torch.testing.assert_close(output.float(), ref.float(), rtol=0.1, atol=5e-2)
    print(f"[PASS] test_fp8_dequant_with_scale — N={N}, K={K}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: FP8 Dequant Tests")
    print("=" * 60)
    test_fp8_dequant_basic()
    test_fp8_dequant_with_scale()
    print("=" * 60)
    print("ALL FP8_DEQUANT TESTS PASSED")
    print("=" * 60)
