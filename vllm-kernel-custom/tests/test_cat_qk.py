"""
Test esimd_cat_qk kernel.

Concatenates Q_nope + Q_pe → Q_out and K_nope + K_pe → K_out.
Hardcoded for QK_DIM1=512, QK_DIM2=64. Output is [seq, heads, 576].

Input layouts (from cat.h comments):
  q_nope (q1): [head, seq, 512], q1_head_stride = seq*512
  q_pe   (q2): [seq, head, 64] with custom stride per head = q_stride
  k_nope (k1): [seq, kv_head, 512], k1_head_stride = 512
  k_pe   (k2): [seq, kv_head, 64] with custom stride per head = k_stride

Wrapper params: (q_nope, q_pe, k_nope, k_pe, q_out, k_out,
                 seq_len, q_heads, kv_heads, qk_head_dim, stride)
  qk_head_dim → q_stride (stride per head in q_pe)
  stride → k_stride (stride per head in k_pe)

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_cat_qk.py
"""
import torch

device = torch.device("xpu")


def test_cat_qk_contiguous():
    """Concatenation with contiguous PE strides (stride == 64)."""
    from vllm_kernel_custom import esimd_cat_qk

    seq_len = 4
    q_heads = 16
    kv_heads = 1
    qk_dim1 = 512   # nope dimension
    qk_dim2 = 64    # pe dimension

    # q_nope: [head, seq, 512] — contiguous
    q_nope = torch.randn(q_heads, seq_len, qk_dim1, dtype=torch.float16, device=device) * 0.1

    # q_pe: [seq, head, 64] — contiguous, stride_per_head = 64
    q_pe = torch.randn(seq_len, q_heads, qk_dim2, dtype=torch.float16, device=device) * 0.1

    # k_nope: [seq, kv_head, 512] — contiguous
    k_nope = torch.randn(seq_len, kv_heads, qk_dim1, dtype=torch.float16, device=device) * 0.1

    # k_pe: [seq, kv_head, 64] — contiguous, stride_per_head = 64
    k_pe = torch.randn(seq_len, kv_heads, qk_dim2, dtype=torch.float16, device=device) * 0.1

    q_out = torch.zeros(seq_len, q_heads, qk_dim1 + qk_dim2, dtype=torch.float16, device=device)
    k_out = torch.zeros(seq_len, kv_heads, qk_dim1 + qk_dim2, dtype=torch.float16, device=device)

    # qk_head_dim=q_pe_stride=64, stride=k_pe_stride=64 (contiguous)
    esimd_cat_qk(q_nope, q_pe, k_nope, k_pe,
                 q_out, k_out,
                 seq_len, q_heads, kv_heads, qk_dim2, qk_dim2)

    # Build reference
    q_nope_transposed = q_nope.permute(1, 0, 2)  # [seq, head, 512]
    q_ref = torch.cat([q_nope_transposed, q_pe], dim=-1)  # [seq, head, 576]
    k_ref = torch.cat([k_nope, k_pe], dim=-1)              # [seq, kv_head, 576]

    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=1e-3, atol=1e-5)
    print(f"[PASS] test_cat_qk_contiguous — seq_len={seq_len}, q_heads={q_heads}, kv_heads={kv_heads}")


def test_cat_qk_single_token():
    """Single-token (decode) scenario: seq_len=1."""
    from vllm_kernel_custom import esimd_cat_qk

    seq_len = 1
    q_heads = 128
    kv_heads = 1
    qk_dim1 = 512
    qk_dim2 = 64

    q_nope = torch.randn(q_heads, seq_len, qk_dim1, dtype=torch.float16, device=device) * 0.1
    q_pe = torch.randn(seq_len, q_heads, qk_dim2, dtype=torch.float16, device=device) * 0.1
    k_nope = torch.randn(seq_len, kv_heads, qk_dim1, dtype=torch.float16, device=device) * 0.1
    k_pe = torch.randn(seq_len, kv_heads, qk_dim2, dtype=torch.float16, device=device) * 0.1

    q_out = torch.zeros(seq_len, q_heads, qk_dim1 + qk_dim2, dtype=torch.float16, device=device)
    k_out = torch.zeros(seq_len, kv_heads, qk_dim1 + qk_dim2, dtype=torch.float16, device=device)

    esimd_cat_qk(q_nope, q_pe, k_nope, k_pe,
                 q_out, k_out,
                 seq_len, q_heads, kv_heads, qk_dim2, qk_dim2)

    q_nope_ref = q_nope.permute(1, 0, 2)
    q_ref = torch.cat([q_nope_ref, q_pe], dim=-1)
    k_ref = torch.cat([k_nope, k_pe], dim=-1)

    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=1e-3, atol=1e-5)
    print(f"[PASS] test_cat_qk_single_token — seq_len={seq_len}, q_heads={q_heads}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: Cat QK Tests")
    print("=" * 60)
    test_cat_qk_contiguous()
    test_cat_qk_single_token()
    print("=" * 60)
    print("ALL CAT_QK TESTS PASSED")
    print("=" * 60)
