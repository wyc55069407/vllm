"""
Test esimd_rope kernel (Rotary Position Embedding).

Applies interleaved-pair complex rotation in-place on Q and K.
Cache layout: [max_pos, rope_dim] where first rope_dim/2 = cos, last rope_dim/2 = sin.
Rotation convention (interleaved pairs):
  output[2i]   = input[2i]*cos[i]  - input[2i+1]*sin[i]
  output[2i+1] = input[2i+1]*cos[i] + input[2i]*sin[i]

Q layout: [seq, heads, hd_stride] (contiguous), stride_per_head = hd_stride
K layout: [seq, ..., hd_stride_kv] with custom strides

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_rope.py
"""
import torch
import math

device = torch.device("xpu")


def rope_interleaved_ref(x, cos, sin):
    """Reference interleaved RoPE rotation.
    x: [..., 64]  (interleaved real/imag pairs)
    cos: [32], sin: [32]
    """
    d = x.shape[-1]
    half_d = d // 2
    x_even = x[..., 0::2]  # [32] real parts
    x_odd = x[..., 1::2]   # [32] imag parts
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin
    out = torch.stack([out_even, out_odd], dim=-1)
    return out.reshape(x.shape)


def build_cos_sin_cache(max_pos, rope_dim=64, base=10000.0):
    """Build cos/sin cache matching kernel layout:
    [max_pos, rope_dim] where [pos, :32]=cos, [pos, 32:]=sin."""
    half_d = rope_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_d, dtype=torch.float32) / half_d))
    positions = torch.arange(max_pos, dtype=torch.float32)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [max_pos, 32]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    cache = torch.cat([cos_vals, sin_vals], dim=-1)  # [max_pos, 64]
    return cache.half()


def test_rope_basic():
    """Basic RoPE test with simple positions."""
    from vllm_kernel_custom import esimd_rope

    seq_len = 4
    q_heads = 8
    kv_heads = 1
    rope_dim = 64
    max_pos = 8192

    cos_sin_cache = build_cos_sin_cache(max_pos, rope_dim).to(device)
    positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    positions_k = positions.clone()

    # Q: [seq, heads, rope_dim]
    q = torch.randn(seq_len, q_heads, rope_dim, dtype=torch.float16, device=device) * 0.5
    # K: [seq, kv_heads, rope_dim]
    k = torch.randn(seq_len, kv_heads, rope_dim, dtype=torch.float16, device=device) * 0.5

    q_orig = q.clone()
    k_orig = k.clone()

    q_stride = rope_dim
    k_stride = rope_dim
    k_nope_stride = kv_heads * rope_dim

    esimd_rope(q, k, cos_sin_cache, positions, positions_k,
               q_heads, rope_dim, q_stride,
               kv_heads, rope_dim, k_stride,
               k_nope_stride, seq_len, 0)

    # Reference
    for s in range(seq_len):
        pos = positions[s].item()
        cos_val = cos_sin_cache[pos, :rope_dim // 2].float()
        sin_val = cos_sin_cache[pos, rope_dim // 2:].float()

        for h in range(q_heads):
            q_ref = rope_interleaved_ref(q_orig[s, h].float(), cos_val, sin_val)
            torch.testing.assert_close(q[s, h].float(), q_ref.half().float(), rtol=5e-2, atol=5e-3)

        for h in range(kv_heads):
            k_ref = rope_interleaved_ref(k_orig[s, h].float(), cos_val, sin_val)
            torch.testing.assert_close(k[s, h].float(), k_ref.half().float(), rtol=5e-2, atol=5e-3)

    print(f"[PASS] test_rope_basic — seq_len={seq_len}, q_heads={q_heads}, rope_dim={rope_dim}")


def test_rope_offset_positions():
    """RoPE with non-zero starting positions."""
    from vllm_kernel_custom import esimd_rope

    seq_len = 2
    q_heads = 4
    kv_heads = 1
    rope_dim = 64
    max_pos = 8192

    cos_sin_cache = build_cos_sin_cache(max_pos, rope_dim).to(device)
    positions = torch.tensor([100, 101], dtype=torch.int64, device=device)
    positions_k = positions.clone()

    q = torch.randn(seq_len, q_heads, rope_dim, dtype=torch.float16, device=device) * 0.5
    k = torch.randn(seq_len, kv_heads, rope_dim, dtype=torch.float16, device=device) * 0.5

    q_orig = q.clone()

    esimd_rope(q, k, cos_sin_cache, positions, positions_k,
               q_heads, rope_dim, rope_dim,
               kv_heads, rope_dim, rope_dim,
               kv_heads * rope_dim, seq_len, 0)

    for s in range(seq_len):
        pos = positions[s].item()
        cos_val = cos_sin_cache[pos, :rope_dim // 2].float()
        sin_val = cos_sin_cache[pos, rope_dim // 2:].float()
        for h in range(q_heads):
            q_ref = rope_interleaved_ref(q_orig[s, h].float(), cos_val, sin_val)
            torch.testing.assert_close(q[s, h].float(), q_ref.half().float(), rtol=5e-2, atol=5e-3)

    print(f"[PASS] test_rope_offset_positions — positions=[100,101]")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: RoPE Tests")
    print("=" * 60)
    test_rope_basic()
    test_rope_offset_positions()
    print("=" * 60)
    print("ALL ROPE TESTS PASSED")
    print("=" * 60)
