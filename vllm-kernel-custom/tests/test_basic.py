"""
Basic ULT for vllm-kernel-custom: covers non-LGRF (esimd_add) and LGRF (esimd_mul_lgrf).

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_basic.py
"""
import torch

device = torch.device("xpu")


def test_esimd_add_fused():
    """Non-LGRF kernel: fused residual add — c = a_bf16 * factor + b_fp16.
    Kernel always loads a as bf16, b as fp16. Output type matches c's dtype."""
    from vllm_kernel_custom import esimd_mul_scale_factor_and_add

    length = 40960
    # a must be bf16 (kernel loads as bf16)
    a = torch.rand((1, length), dtype=torch.bfloat16, device=device)
    # b must be fp16 (kernel loads as fp16)
    b = torch.rand((1, length), dtype=torch.float16, device=device) * 0.1
    # c output as bf16
    c = torch.zeros((1, length), dtype=torch.bfloat16, device=device)

    factor = 1.0
    esimd_mul_scale_factor_and_add(a, b, c, length, factor)

    ref = (a.float() * factor + b.float()).bfloat16()
    torch.testing.assert_close(c.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"[PASS] test_esimd_add_fused (non-LGRF) — c = a_bf16*{factor} + b_fp16, length={length}")


def test_esimd_add_fused_fp16_out():
    """Non-LGRF kernel: fused residual add with fp16 output."""
    from vllm_kernel_custom import esimd_mul_scale_factor_and_add

    length = 40960
    a = torch.rand((1, length), dtype=torch.bfloat16, device=device)
    b = torch.rand((1, length), dtype=torch.float16, device=device) * 0.5
    c = torch.zeros((1, length), dtype=torch.float16, device=device)

    factor = 0.5
    esimd_mul_scale_factor_and_add(a, b, c, length, factor)

    ref = (a.float() * factor + b.float()).half()
    torch.testing.assert_close(c.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"[PASS] test_esimd_add_fused_fp16_out (non-LGRF) — c = a_bf16*{factor} + b_fp16, length={length}")


def test_esimd_mul_lgrf():
    """LGRF kernel: element-wise multiply c = a * b (fp16)."""
    from vllm_kernel_custom import esimd_mul_lgrf

    length = 40960
    a = torch.rand((1, length), dtype=torch.float16, device=device)
    b = torch.rand((1, length), dtype=torch.float16, device=device) * 0.1
    c = torch.zeros((1, length), dtype=torch.float16, device=device)

    esimd_mul_lgrf(a, b, c, 1000, length)

    ref = a * b
    torch.testing.assert_close(c.float(), ref.float(), rtol=1e-3, atol=1e-5)
    print(f"[PASS] test_esimd_mul_lgrf (LGRF) — c = a * b, fp16, length={length}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom Basic ULT")
    print("=" * 60)
    test_esimd_add_fused()
    test_esimd_add_fused_fp16_out()
    test_esimd_mul_lgrf()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
