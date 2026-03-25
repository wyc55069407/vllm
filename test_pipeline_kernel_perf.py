"""Measure individual kernel timings in the sparse decode/prefill pipeline.

Uses XPU events for timing (GPU-accurate, no CPU sync overhead).
Compares old pipeline (Python K extraction) vs new (paged K pooling).
"""
import os, torch, math, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm_kernel_custom import (
    esimd_infllmv2_k_pooling,
    esimd_infllmv2_k_pooling_paged,
    esimd_infllmv2_pattern_decode,
    esimd_infllmv2_pattern_prefill,
    esimd_infllmv2_force_last_block,
    esimd_infllmv2_mask_convert,
    esimd_sdp_paged_sparse,
)


def time_kernel(fn, warmup=3, iters=10):
    """Time a kernel using XPU events."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.xpu.synchronize()
    return start.elapsed_time(end) / iters


def bench_decode_pipeline(seq_len, batch=1):
    """Benchmark individual kernels in decode pipeline."""
    nh, nkvh, hd = 32, 2, 128
    block_size = 128
    topk = 64
    kernel_size, kernel_stride = 32, 16
    sparse_block = 64
    init_blocks, local_blocks = 2, 4
    scale = 1.0 / math.sqrt(hd)
    device = 'xpu'

    num_pages = (seq_len + block_size - 1) // block_size
    total_pages = num_pages + 4
    num_pooled_blocks = max(1, (seq_len - kernel_size + kernel_stride) // kernel_stride)
    pooling_stride = sparse_block // kernel_stride
    num_pooled = max(1, (num_pooled_blocks + pooling_stride - 1) // pooling_stride)

    # Setup
    kv_cache = torch.randn(2, total_pages, block_size, nkvh, hd,
                           dtype=torch.bfloat16, device=device) * 0.3
    bt = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    bt_padded = torch.zeros(1, total_pages, dtype=torch.int32, device=device)
    bt_padded[0, :num_pages] = bt[0]
    sl = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query = torch.randn(batch, nh, hd, dtype=torch.bfloat16, device=device) * 0.5

    k_pooled = torch.zeros(batch, nkvh, num_pooled_blocks, hd,
                           dtype=torch.float16, device=device)

    # Pre-allocate buffers
    q_fp = query.view(batch, nh, hd).unsqueeze(2).half()
    bs_t = torch.zeros(batch, nh, 1, num_pooled_blocks, dtype=torch.float16, device=device)
    kvbs_t = torch.zeros(batch, nkvh, 1, num_pooled_blocks, dtype=torch.float16, device=device)
    ps_t = torch.zeros(batch, nkvh, 1, num_pooled, dtype=torch.float16, device=device)
    tk_t = torch.zeros(batch, nkvh, 1, topk, dtype=torch.int32, device=device)
    output = torch.zeros(batch, nh, hd, dtype=query.dtype, device=device)
    qsl = torch.tensor([0, batch], dtype=torch.int32, device=device)
    dummy_cnt = torch.zeros(1, dtype=torch.int32, device=device)

    # 1. Paged K pooling
    t_pool = time_kernel(lambda: esimd_infllmv2_k_pooling_paged(
        kv_cache, k_pooled, bt_padded, sl,
        nkvh, hd, block_size, num_pooled_blocks,
        kernel_size, kernel_stride, 0))

    # 2. Pattern detection (decode)
    t_pattern = time_kernel(lambda: esimd_infllmv2_pattern_decode(
        q_fp, k_pooled,
        bs_t, kvbs_t, ps_t, tk_t,
        nh, nkvh, 1, num_pooled_blocks, hd, num_pooled,
        seq_len - 1, 1, init_blocks, local_blocks, topk))

    # 3. Force last block
    sparse_mask = tk_t.squeeze(2).int()
    t_force = time_kernel(lambda: esimd_infllmv2_force_last_block(
        sparse_mask, sl, nkvh, sparse_block))

    # 4. Sparse SDP decode
    t_sdp = time_kernel(lambda: esimd_sdp_paged_sparse(
        query, kv_cache, output, bt_padded, sl, qsl,
        sparse_mask, dummy_cnt,
        nh, nkvh, hd, block_size, seq_len, scale, 1, topk))

    total = t_pool + t_pattern + t_force + t_sdp
    print(f"  seq={seq_len:6d}  pool={t_pool:.3f}ms  pattern={t_pattern:.3f}ms  "
          f"force={t_force:.3f}ms  sdp={t_sdp:.3f}ms  total={total:.3f}ms")
    return total


def bench_prefill_pipeline(seq_len, q_len=None):
    """Benchmark individual kernels in prefill pipeline."""
    if q_len is None:
        q_len = seq_len
    nh, nkvh, hd = 32, 2, 128
    block_size = 128
    topk = 64
    kernel_size, kernel_stride = 32, 16
    sparse_block = 64
    init_blocks, local_blocks = 2, 4
    scale = 1.0 / math.sqrt(hd)
    device = 'xpu'

    num_pages = (seq_len + block_size - 1) // block_size
    total_pages = num_pages + 4
    num_pooled_blocks = max(1, (seq_len - kernel_size + kernel_stride) // kernel_stride)
    pooling_stride = sparse_block // kernel_stride
    num_pooled = max(1, (num_pooled_blocks + pooling_stride - 1) // pooling_stride)
    q_blocks = (q_len + 15) // 16
    total_kv_blocks = (seq_len + sparse_block - 1) // sparse_block

    # Setup
    kv_cache = torch.randn(2, total_pages, block_size, nkvh, hd,
                           dtype=torch.bfloat16, device=device) * 0.3
    bt = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    bt_padded = torch.zeros(1, total_pages, dtype=torch.int32, device=device)
    bt_padded[0, :num_pages] = bt[0]
    sl = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query = torch.randn(q_len, nh, hd, dtype=torch.bfloat16, device=device) * 0.5
    output = torch.zeros(q_len, nh, hd, dtype=query.dtype, device=device)

    k_pooled = torch.zeros(1, nkvh, num_pooled_blocks, hd,
                           dtype=torch.float16, device=device)
    q_fp = query.view(q_len, nh, hd).permute(1, 0, 2).unsqueeze(0).half()
    bs_t = torch.zeros(1, nkvh, q_len, num_pooled_blocks, dtype=torch.float16, device=device)
    ps_t = torch.zeros(1, nkvh, q_len, num_pooled, dtype=torch.float16, device=device)
    tk_t = torch.zeros(1, nkvh, q_len, topk, dtype=torch.int32, device=device)
    mask_out = torch.zeros(nkvh, q_blocks, 1024, dtype=torch.int32, device=device)
    mask_cnt = torch.zeros(nkvh, q_blocks, dtype=torch.int32, device=device)
    qsl = torch.tensor([0, q_len], dtype=torch.int32, device=device)

    history_len = seq_len - q_len

    # 1. Paged K pooling
    t_pool = time_kernel(lambda: esimd_infllmv2_k_pooling_paged(
        kv_cache, k_pooled, bt_padded, sl,
        nkvh, hd, block_size, num_pooled_blocks,
        kernel_size, kernel_stride, 0))

    # 2. Pattern detection (prefill)
    t_pattern = time_kernel(lambda: esimd_infllmv2_pattern_prefill(
        q_fp, k_pooled,
        bs_t, ps_t, tk_t,
        nh, nkvh, q_len, num_pooled_blocks, hd, num_pooled,
        history_len, 1, init_blocks, local_blocks, topk))

    # 3. Mask convert
    mask_orig = tk_t.squeeze(0).int()
    t_mask = time_kernel(lambda: esimd_infllmv2_mask_convert(
        mask_orig, mask_out, mask_cnt,
        q_len, nkvh, total_kv_blocks))

    # 4. Sparse SDP prefill
    t_sdp = time_kernel(lambda: esimd_sdp_paged_sparse(
        query, kv_cache, output, bt_padded, sl, qsl,
        mask_out.int(), mask_cnt.int(),
        nh, nkvh, hd, block_size, seq_len, scale, 0, topk))

    total = t_pool + t_pattern + t_mask + t_sdp
    print(f"  seq={seq_len:6d} q={q_len:5d}  pool={t_pool:.3f}ms  pattern={t_pattern:.3f}ms  "
          f"mask={t_mask:.3f}ms  sdp={t_sdp:.3f}ms  total={total:.3f}ms")
    return total


if __name__ == '__main__':
    print("=" * 80)
    print("  Sparse Decode Pipeline — Per-Kernel Timings (New Paged Approach)")
    print("=" * 80)
    for sl in [8192, 16384, 32768]:
        bench_decode_pipeline(sl)

    print()
    print("=" * 80)
    print("  Sparse Prefill Pipeline — Per-Kernel Timings")
    print("=" * 80)
    for sl, ql in [(16384, 8192), (32768, 8192), (65536, 8192), (131072, 8192)]:
        bench_prefill_pipeline(sl, ql)

    print()
    print("Done.")
