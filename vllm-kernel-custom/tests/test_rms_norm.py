"""
Test esimd_rms_norm kernel.

RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * weight
With add_residual=1: x = input + residual, then normalize, and residual is updated to x.
flag=0: fp16 output, flag=1: bf16 output (bf16 input).

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_rms_norm.py
"""
import torch

device = torch.device("xpu")


def rms_norm_ref(x, weight, eps):
    """Reference RMSNorm in float32."""
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f / torch.sqrt(variance + eps)
    return (x_normed * weight.float()).to(x.dtype)


def test_rms_norm_no_residual():
    """RMSNorm without residual addition (add_residual=0, flag=0 fp16)."""
    from vllm_kernel_custom import esimd_rms_norm

    hidden_size = 7168
    seq_len = 4
    eps = 1e-6

    weight = torch.rand(hidden_size, dtype=torch.float16, device=device)
    input_t = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device) * 0.1
    residual = torch.zeros(seq_len, hidden_size, dtype=torch.float16, device=device)
    output = torch.zeros(seq_len, hidden_size, dtype=torch.float16, device=device)

    esimd_rms_norm(weight, residual, input_t, output,
                   hidden_size, seq_len, 0, 0, eps)

    ref = rms_norm_ref(input_t, weight, eps)
    torch.testing.assert_close(output.float(), ref.float(), rtol=5e-2, atol=5e-3)
    print(f"[PASS] test_rms_norm_no_residual — hidden_size={hidden_size}, seq_len={seq_len}")


def test_rms_norm_with_residual():
    """RMSNorm with residual addition (add_residual=1, flag=0 fp16).
    x = input + residual, then normalize. residual buffer is updated to x."""
    from vllm_kernel_custom import esimd_rms_norm

    hidden_size = 7168
    seq_len = 2
    eps = 1e-6

    weight = torch.rand(hidden_size, dtype=torch.float16, device=device)
    input_t = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device) * 0.1
    residual = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device) * 0.1
    output = torch.zeros(seq_len, hidden_size, dtype=torch.float16, device=device)

    # Save for reference before kernel modifies residual in-place
    combined = input_t.float() + residual.float()

    esimd_rms_norm(weight, residual, input_t, output,
                   hidden_size, seq_len, 1, 0, eps)

    ref = rms_norm_ref(combined.half(), weight, eps)
    torch.testing.assert_close(output.float(), ref.float(), rtol=5e-2, atol=5e-3)

    # residual should be updated to input + old_residual
    torch.testing.assert_close(residual.float(), combined.half().float(), rtol=1e-3, atol=1e-3)
    print(f"[PASS] test_rms_norm_with_residual — hidden_size={hidden_size}, seq_len={seq_len}")


def test_rms_norm_small_hidden():
    """RMSNorm with smaller hidden_size (must be multiple of 128)."""
    from vllm_kernel_custom import esimd_rms_norm

    hidden_size = 1536
    seq_len = 8
    eps = 1e-5

    weight = torch.rand(hidden_size, dtype=torch.float16, device=device)
    input_t = torch.randn(seq_len, hidden_size, dtype=torch.float16, device=device) * 0.1
    residual = torch.zeros(seq_len, hidden_size, dtype=torch.float16, device=device)
    output = torch.zeros(seq_len, hidden_size, dtype=torch.float16, device=device)

    esimd_rms_norm(weight, residual, input_t, output,
                   hidden_size, seq_len, 0, 0, eps)

    ref = rms_norm_ref(input_t, weight, eps)
    torch.testing.assert_close(output.float(), ref.float(), rtol=5e-2, atol=5e-3)
    print(f"[PASS] test_rms_norm_small_hidden — hidden_size={hidden_size}, seq_len={seq_len}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: RMSNorm Tests")
    print("=" * 60)
    test_rms_norm_no_residual()
    test_rms_norm_with_residual()
    test_rms_norm_small_hidden()
    print("=" * 60)
    print("ALL RMSNORM TESTS PASSED")
    print("=" * 60)
