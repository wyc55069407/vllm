"""
Test esimd_update_kv kernel.

Scatters new KV tokens into a KV cache at specified positions.
  kv_cache[indices[i], :] = new_kv[i, :]

Internal function: esimd_updateKV_indexed(kv_cache, loc, kv_input, kv_hd, kv_stride, update_len)
Wrapper maps: (kv_cache, new_kv, indices, seq_len, head_dim, num_heads) where:
  - seq_len → update_len (number of tokens to update)
  - head_dim → kv_hd (and also kv_stride)
  - num_heads → unused but passed as update_len

Currently hardcoded for 5 threads with last thread handling 64 elements,
meaning kv_hd = 5*128 - 64 = 576 or kv_hd = 640 (5*128).
Actually: 4 threads do 128 elements, last does 64 = 4*128+64 = 576.
Wait — let me re-check: n_threads=5, last does 64. So total = 4*128+64 = 576.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_update_kv.py
"""
import torch

device = torch.device("xpu")


def test_update_kv_basic():
    """Update KV cache at specified indices."""
    from vllm_kernel_custom import esimd_update_kv

    max_seq_len = 1024
    # Kernel uses 5 threads: 4 x 128 + 1 x 64 = 576 elements per token
    head_dim = 576
    update_len = 8

    # KV cache initialized to zeros
    kv_cache = torch.zeros(max_seq_len, head_dim, dtype=torch.float16, device=device)

    # New KV values to insert
    new_kv = torch.randn(update_len, head_dim, dtype=torch.float16, device=device)

    # Indices (int64) — absolute positions in cache
    indices = torch.tensor([10, 50, 100, 200, 300, 500, 700, 900],
                           dtype=torch.int64, device=device)

    # Wrapper: (kv_cache, new_kv, indices, seq_len=update_len, head_dim, num_heads=unused)
    esimd_update_kv(kv_cache, new_kv, indices, update_len, head_dim, 1)

    # Verify: cache at specified indices should match new_kv
    for i in range(update_len):
        idx = indices[i].item()
        torch.testing.assert_close(
            kv_cache[idx].float(), new_kv[i].float(),
            rtol=1e-3, atol=1e-5)

    # Verify: other positions remain zero
    used_indices = set(indices.cpu().tolist())
    for j in [0, 5, 15, 75, 150, 999]:
        if j not in used_indices:
            assert kv_cache[j].abs().sum().item() == 0, \
                f"Position {j} should be zero but has values"

    print(f"[PASS] test_update_kv_basic — max_seq={max_seq_len}, update_len={update_len}, hd={head_dim}")


def test_update_kv_overwrite():
    """Update KV cache: overwrite existing values."""
    from vllm_kernel_custom import esimd_update_kv

    max_seq_len = 512
    head_dim = 576
    update_len = 4

    # Pre-fill cache with ones
    kv_cache = torch.ones(max_seq_len, head_dim, dtype=torch.float16, device=device)

    new_kv = torch.randn(update_len, head_dim, dtype=torch.float16, device=device) * 0.5
    indices = torch.tensor([0, 100, 200, 400], dtype=torch.int64, device=device)

    esimd_update_kv(kv_cache, new_kv, indices, update_len, head_dim, 1)

    # Overwritten positions should match new_kv
    for i in range(update_len):
        idx = indices[i].item()
        torch.testing.assert_close(
            kv_cache[idx].float(), new_kv[i].float(),
            rtol=1e-3, atol=1e-5)

    # Non-overwritten positions should still be ones
    assert kv_cache[50].abs().mean().item() > 0.9, "Non-updated position should remain ~1.0"

    print(f"[PASS] test_update_kv_overwrite — max_seq={max_seq_len}, update_len={update_len}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: Update KV Cache Tests")
    print("=" * 60)
    test_update_kv_basic()
    test_update_kv_overwrite()
    print("=" * 60)
    print("ALL UPDATE_KV TESTS PASSED")
    print("=" * 60)
