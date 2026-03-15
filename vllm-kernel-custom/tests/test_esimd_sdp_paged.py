"""Unit test + performance benchmarks for ESIMD paged SDP attention kernel.

Compares against torch.nn.functional.scaled_dot_product_attention.
Tests both decode (query_len=1) and prefill with causal masking.
Supports block_size=64 and block_size=1024, head_dim=128 and 256.

Performance benchmarks report TFLOPS (prefill) and GB/s (decode).
"""
import torch
import pytest
import math
import time


def create_paged_kv_cache(
    keys, values, block_size, num_kv_heads, head_dim, device,
):
    """Create a paged KV cache from dense key/value tensors.

    Args:
        keys: list of [seq_len_i, num_kv_heads, head_dim] bf16 tensors
        values: list of [seq_len_i, num_kv_heads, head_dim] bf16 tensors
        block_size: KV cache block size

    Returns:
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim] bf16
        block_table: [batch, max_blocks_per_seq] i32
        seq_lens: [batch] i32
    """
    batch = len(keys)
    seq_lens = [k.shape[0] for k in keys]
    max_seq_len = max(seq_lens)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)

    kv_cache = torch.zeros(2, total_blocks, block_size, num_kv_heads, head_dim,
                           dtype=torch.bfloat16, device=device)
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32, device=device)

    block_counter = 0
    for b in range(batch):
        sl = seq_lens[b]
        num_b = (sl + block_size - 1) // block_size
        for bi in range(num_b):
            block_table[b, bi] = block_counter
            start = bi * block_size
            end = min(start + block_size, sl)
            length = end - start
            kv_cache[0, block_counter, :length] = keys[b][start:end]
            kv_cache[1, block_counter, :length] = values[b][start:end]
            block_counter += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return kv_cache, block_table, seq_lens_t


def pytorch_sdp_reference_per_head(query, keys_dense, values_dense, query_lens,
                                    seq_lens_list, num_heads, num_kv_heads,
                                    head_dim, scale, causal=True):
    """PyTorch reference: per-head SDPA with GQA expansion."""
    group_size = num_heads // num_kv_heads
    output_ref = torch.zeros_like(query)
    q_offset = 0

    for b in range(len(keys_dense)):
        ql = query_lens[b]
        sl = seq_lens_list[b]

        k_expanded = keys_dense[b].float().repeat_interleave(group_size, dim=1)
        v_expanded = values_dense[b].float().repeat_interleave(group_size, dim=1)

        for qi in range(ql):
            q_pos = (sl - ql) + qi
            q_vec = query[q_offset + qi].float()

            for h in range(num_heads):
                kv_end = (q_pos + 1) if causal else sl
                k_slice = k_expanded[:kv_end, h, :]
                v_slice = v_expanded[:kv_end, h, :]
                scores = torch.mv(k_slice, q_vec[h]) * scale
                weights = torch.softmax(scores, dim=0)
                output_ref[q_offset + qi, h] = (
                    weights.unsqueeze(1) * v_slice
                ).sum(0).to(torch.bfloat16)

        q_offset += ql

    return output_ref


# ============================================================
# Correctness tests
# ============================================================

@pytest.mark.parametrize("block_size", [64, 1024])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_sdp_paged_correctness(block_size, head_dim, mode):
    """Test ESIMD paged SDP against PyTorch reference."""
    try:
        from vllm_kernel_custom import esimd_sdp_paged
    except ImportError:
        pytest.skip("vllm_kernel_custom not built with SDP paged support")

    device = "xpu"
    num_heads = 16
    num_kv_heads = 4
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    if mode == "decode":
        batch = 4
        seq_lens_list = [32, 64, 48, 16]
        query_lens = [1] * batch
    else:
        batch = 2
        seq_lens_list = [32, 48]
        query_lens = [8, 12]

    keys_dense = []
    values_dense = []
    for sl in seq_lens_list:
        keys_dense.append(
            torch.randn(sl, num_kv_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1)
        values_dense.append(
            torch.randn(sl, num_kv_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1)

    kv_cache, block_table, seq_lens_t = create_paged_kv_cache(
        keys_dense, values_dense, block_size, num_kv_heads, head_dim, device)

    total_q_tokens = sum(query_lens)
    query = torch.randn(total_q_tokens, num_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1

    qsl = [0]
    for ql in query_lens:
        qsl.append(qsl[-1] + ql)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)

    output_esimd = torch.zeros_like(query)

    esimd_sdp_paged(
        query, kv_cache, output_esimd,
        block_table, seq_lens_t, query_start_loc,
        num_heads, num_kv_heads,
        head_dim, block_size,
        max(seq_lens_list), scale, 1,
    )
    torch.xpu.synchronize()

    output_ref = pytorch_sdp_reference_per_head(
        query, keys_dense, values_dense, query_lens,
        seq_lens_list, num_heads, num_kv_heads, head_dim, scale, causal=True)

    output_esimd_f32 = output_esimd.float().cpu()
    output_ref_f32 = output_ref.float().cpu()

    # Relative RMS error
    diff = (output_esimd_f32 - output_ref_f32)
    rms = (diff ** 2).mean().sqrt().item()
    ref_rms = (output_ref_f32 ** 2).mean().sqrt().item()
    rel_rms = rms / max(ref_rms, 1e-8)

    atol = 0.05
    rtol = 0.05
    close = torch.allclose(output_esimd_f32, output_ref_f32, atol=atol, rtol=rtol)

    if not close:
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        print(f"Output diff: max={max_diff:.6f}, mean={mean_diff:.6f}, rel_rms={rel_rms:.6f}")

    assert close, (
        f"SDP paged output mismatch ({mode}, block_size={block_size}, "
        f"head_dim={head_dim}): rel_rms={rel_rms:.6f}"
    )
    assert not torch.isnan(output_esimd).any(), "NaN in ESIMD SDP output"

    print(f"PASS: mode={mode}, block_size={block_size}, head_dim={head_dim}, rel_rms={rel_rms:.6f}")


@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("head_dim", [256])
def test_sdp_paged_noncausal(block_size, head_dim):
    """Test non-causal paged SDP."""
    try:
        from vllm_kernel_custom import esimd_sdp_paged
    except ImportError:
        pytest.skip("vllm_kernel_custom not built")

    device = "xpu"
    num_heads = 16
    num_kv_heads = 4
    scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(123)

    batch = 2
    seq_lens_list = [40, 56]
    query_lens = [6, 10]

    keys_dense, values_dense = [], []
    for sl in seq_lens_list:
        keys_dense.append(torch.randn(sl, num_kv_heads, head_dim,
                                       dtype=torch.bfloat16, device=device) * 0.1)
        values_dense.append(torch.randn(sl, num_kv_heads, head_dim,
                                         dtype=torch.bfloat16, device=device) * 0.1)

    kv_cache, block_table, seq_lens_t = create_paged_kv_cache(
        keys_dense, values_dense, block_size, num_kv_heads, head_dim, device)

    total_q = sum(query_lens)
    query = torch.randn(total_q, num_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1

    qsl = [0]
    for ql in query_lens:
        qsl.append(qsl[-1] + ql)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)

    output_esimd = torch.zeros_like(query)
    esimd_sdp_paged(query, kv_cache, output_esimd, block_table, seq_lens_t,
                    query_start_loc, num_heads, num_kv_heads, head_dim,
                    block_size, max(seq_lens_list), scale, 0)  # causal=0
    torch.xpu.synchronize()

    output_ref = pytorch_sdp_reference_per_head(
        query, keys_dense, values_dense, query_lens,
        seq_lens_list, num_heads, num_kv_heads, head_dim, scale, causal=False)

    atol = 0.05
    rtol = 0.05
    close = torch.allclose(output_esimd.float().cpu(), output_ref.float().cpu(),
                           atol=atol, rtol=rtol)
    assert close, "Non-causal SDP paged output mismatch"
    assert not torch.isnan(output_esimd).any()
    print(f"PASS: noncausal, block_size={block_size}, head_dim={head_dim}")


# ============================================================
# Performance benchmarks
# ============================================================

def _warmup_and_time(fn, warmup=5, iters=20):
    """Run warmup iterations, then time `iters` runs. Returns seconds per call."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


@pytest.mark.parametrize("q_len,kv_len", [
    (512, 512), (1024, 1024), (4096, 4096),
])
def test_sdp_paged_prefill_perf(q_len, kv_len):
    """Benchmark prefill TFLOPS for paged SDP (HD=256)."""
    try:
        from vllm_kernel_custom import esimd_sdp_paged
    except ImportError:
        pytest.skip("vllm_kernel_custom not built")

    device = "xpu"
    head_dim = 256
    num_heads = 16
    num_kv_heads = 4
    block_size = 64
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(0)

    # Single request
    batch = 1
    seq_len = kv_len
    query_len = q_len

    keys_dense = [torch.randn(seq_len, num_kv_heads, head_dim,
                               dtype=torch.bfloat16, device=device) * 0.1]
    values_dense = [torch.randn(seq_len, num_kv_heads, head_dim,
                                 dtype=torch.bfloat16, device=device) * 0.1]

    kv_cache, block_table, seq_lens_t = create_paged_kv_cache(
        keys_dense, values_dense, block_size, num_kv_heads, head_dim, device)

    query = torch.randn(query_len, num_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1

    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    output = torch.zeros_like(query)

    def run():
        esimd_sdp_paged(query, kv_cache, output, block_table, seq_lens_t,
                        query_start_loc, num_heads, num_kv_heads,
                        head_dim, block_size, seq_len, scale, 1)

    sec = _warmup_and_time(run)

    # FLOPS: 2 * q_len * kv_len * head_dim * num_heads (for QK + SxV)
    flops = 2.0 * q_len * kv_len * head_dim * num_heads * 2  # x2 for QK and SV
    tflops = flops / sec / 1e12

    print(f"Prefill perf: q_len={q_len}, kv_len={kv_len}, "
          f"time={sec*1e3:.2f} ms, TFLOPS={tflops:.1f}")


@pytest.mark.parametrize("seq_len", [512, 1024, 4096, 8192])
def test_sdp_paged_decode_perf(seq_len):
    """Benchmark decode GB/s for paged SDP (HD=256)."""
    try:
        from vllm_kernel_custom import esimd_sdp_paged
    except ImportError:
        pytest.skip("vllm_kernel_custom not built")

    device = "xpu"
    head_dim = 256
    num_heads = 16
    num_kv_heads = 4
    block_size = 64
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(0)
    batch = 1

    keys_dense = [torch.randn(seq_len, num_kv_heads, head_dim,
                               dtype=torch.bfloat16, device=device) * 0.1]
    values_dense = [torch.randn(seq_len, num_kv_heads, head_dim,
                                 dtype=torch.bfloat16, device=device) * 0.1]

    kv_cache, block_table, seq_lens_t = create_paged_kv_cache(
        keys_dense, values_dense, block_size, num_kv_heads, head_dim, device)

    # Decode: 1 query token
    query = torch.randn(1, num_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1

    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = torch.zeros_like(query)

    def run():
        esimd_sdp_paged(query, kv_cache, output, block_table, seq_lens_t,
                        query_start_loc, num_heads, num_kv_heads,
                        head_dim, block_size, seq_len, scale, 1)

    sec = _warmup_and_time(run)

    # Bytes: (K + V) * seq_len * head_dim * num_kv_heads * sizeof(bf16)
    bytes_read = 2 * seq_len * head_dim * num_kv_heads * 2  # K+V, bf16=2B
    gb_per_s = bytes_read / sec / 1e9

    print(f"Decode perf: seq_len={seq_len}, "
          f"time={sec*1e6:.1f} us, GB/s={gb_per_s:.1f}")


# ============================================================
# Main entry
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Correctness tests")
    print("=" * 60)
    for mode in ["decode", "prefill"]:
        for bs in [64, 1024]:
            for hd in [256]:
                try:
                    test_sdp_paged_correctness(bs, hd, mode)
                except Exception as e:
                    print(f"FAIL: mode={mode}, bs={bs}, hd={hd}: {e}")

    print()
    print("=" * 60)
    print("Non-causal test")
    print("=" * 60)
    try:
        test_sdp_paged_noncausal(64, 256)
    except Exception as e:
        print(f"FAIL: noncausal: {e}")

    print()
    print("=" * 60)
    print("Prefill performance benchmarks")
    print("=" * 60)
    for q, kv in [(512, 512), (1024, 1024), (4096, 4096)]:
        try:
            test_sdp_paged_prefill_perf(q, kv)
        except Exception as e:
            print(f"FAIL: q={q}, kv={kv}: {e}")

    print()
    print("=" * 60)
    print("Decode performance benchmarks")
    print("=" * 60)
    for sl in [512, 1024, 4096, 8192]:
        try:
            test_sdp_paged_decode_perf(sl)
        except Exception as e:
            print(f"FAIL: seq_len={sl}: {e}")

    print("\nDone.")
