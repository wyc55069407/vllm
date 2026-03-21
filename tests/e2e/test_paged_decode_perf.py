"""
Standalone paged decode SDP performance benchmark.

Measures GB/s for the two-phase paged decode kernel at various seq_len.
Target: >270 GB/s (>60% of 450 GB/s BMG peak).

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_paged_decode_perf.py
"""

import torch
try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    pass
import time

from vllm_kernel_custom import esimd_sdp_paged


def benchmark_paged_decode(
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    warmup: int = 10,
    repeat: int = 100,
):
    """Benchmark paged decode SDP kernel."""
    device = "xpu"

    # Allocate query [batch, num_heads, head_dim]
    query = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)

    # Allocate paged KV cache
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch * num_blocks_per_seq
    kv_cache = torch.randn(
        2, total_blocks, block_size, num_kv_heads, head_dim,
        dtype=dtype, device=device
    )

    # Block table: sequential assignment
    block_table = torch.zeros(batch, num_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(batch):
        for i in range(num_blocks_per_seq):
            block_table[b, i] = b * num_blocks_per_seq + i

    # Sequence lengths
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Query start locations (decode: each request has 1 token)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=device)

    # Output
    output = torch.empty_like(query)

    # Attn scale
    attn_scale = 1.0 / (head_dim ** 0.5)

    # Warmup
    for _ in range(warmup):
        esimd_sdp_paged(
            query, kv_cache, output, block_table, seq_lens, query_start_loc,
            num_heads, num_kv_heads, head_dim, block_size,
            seq_len,  # max_seq_len
            attn_scale,
            0,  # causal=0 for decode
        )
    torch.xpu.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        esimd_sdp_paged(
            query, kv_cache, output, block_table, seq_lens, query_start_loc,
            num_heads, num_kv_heads, head_dim, block_size,
            seq_len,
            attn_scale,
            0,
        )
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_us = elapsed / repeat * 1e6

    # GB/s calculation: KV read = seq_len * num_kv_heads * head_dim * 2 (K+V) * 2 (bytes/elem) * batch
    kv_bytes = batch * seq_len * num_kv_heads * head_dim * 2 * 2
    gbps = kv_bytes / (elapsed / repeat) / 1e9

    return avg_time_us, gbps


def main():
    configs = [
        # (label, num_heads, num_kv_heads, head_dim, block_size, dtype)
        ("HD128 32Q/4KV bf16 (Qwen3-30B)", 32, 4, 128, 128, torch.bfloat16),
        ("HD128 32Q/4KV fp16 (Qwen3-30B)", 32, 4, 128, 128, torch.float16),
        ("HD128 32Q/2KV bf16 (MiniCPM4-8B)", 32, 2, 128, 128, torch.bfloat16),
        ("HD256 16Q/4KV bf16 (Qwen3.5-4B)", 16, 4, 256, 128, torch.bfloat16),
    ]

    seq_lens = [1024, 2048, 4096, 8192, 16384]
    batch = 1

    for label, nh, nkvh, hd, bs, dtype in configs:
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"  GQA ratio: {nh//nkvh}:1, batch={batch}")
        print(f"{'='*60}")
        print(f"{'seq_len':>8} {'time_us':>10} {'GB/s':>10}")
        print(f"{'-'*8:>8} {'-'*10:>10} {'-'*10:>10}")

        for sl in seq_lens:
            result = benchmark_paged_decode(
                batch=batch, seq_len=sl,
                num_heads=nh, num_kv_heads=nkvh,
                head_dim=hd, block_size=bs,
                dtype=dtype,
            )
            if result is None:
                break
            avg_us, gbps = result
            print(f"{sl:>8} {avg_us:>10.1f} {gbps:>10.1f}")

    # Also test correctness vs scalar (small seq_len)
    print(f"\n{'='*60}")
    print("Correctness check: opt decode vs scalar (seq_len=64)")
    print(f"{'='*60}")
    for label, nh, nkvh, hd, bs, dtype in configs[:2]:
        query = torch.randn(1, nh, hd, dtype=dtype, device="xpu")
        num_blocks = 1
        kv_cache = torch.randn(2, num_blocks, bs, nkvh, hd, dtype=dtype, device="xpu")
        block_table = torch.zeros(1, 1, dtype=torch.int32, device="xpu")
        seq_lens_t = torch.tensor([64], dtype=torch.int32, device="xpu")
        qsl = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
        output = torch.empty_like(query)
        attn_scale = 1.0 / (hd ** 0.5)

        esimd_sdp_paged(
            query, kv_cache, output, block_table, seq_lens_t, qsl,
            nh, nkvh, hd, bs, 64, attn_scale, 0,
        )
        torch.xpu.synchronize()

        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        out_abs_mean = output.abs().mean().item()
        print(f"  {label}: NaN={has_nan}, Inf={has_inf}, abs_mean={out_abs_mean:.6f}")


if __name__ == "__main__":
    main()
