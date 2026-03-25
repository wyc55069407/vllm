"""Compare sparse decode perf: old pipeline (K extraction loop) vs new (paged k_pooling).

This script simulates the decode pipeline outside vLLM to measure
the kernel-level overhead reduction.
"""
import os, time, torch, math
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm_kernel_custom import (
    esimd_infllmv2_k_pooling,
    esimd_infllmv2_k_pooling_paged,
    esimd_infllmv2_pattern_decode,
    esimd_infllmv2_force_last_block,
    esimd_sdp_paged_sparse,
)


def bench_decode_old(kv_cache, block_table, seq_lens, query,
                     nh, nkvh, hd, block_size, scale, topk,
                     kernel_size, kernel_stride, sparse_block,
                     init_blocks, local_blocks,
                     warmup=3, iters=10):
    """Old decode pipeline: Python K extraction + K pooling + pattern + SDP."""
    batch = seq_lens.shape[0]
    device = query.device
    max_sl = int(seq_lens.cpu().max().item())
    num_pooled_blocks = max(1, (max_sl - kernel_size + kernel_stride) // kernel_stride)
    pooling_stride = sparse_block // kernel_stride
    num_pooled = max(1, (num_pooled_blocks + pooling_stride - 1) // pooling_stride)

    def run_once():
        # Step 1: Python K extraction loop
        k_contiguous = torch.zeros(batch, nkvh, max_sl, hd,
                                   dtype=torch.float16, device=device)
        bt_cpu = block_table.cpu()
        seq_lens_cpu = seq_lens.cpu()
        key_cache = kv_cache[0]
        for b in range(batch):
            sl = int(seq_lens_cpu[b].item())
            num_pages = (sl + block_size - 1) // block_size
            for p in range(num_pages):
                phys_page = int(bt_cpu[b, p].item())
                start = p * block_size
                end = min(start + block_size, sl)
                length = end - start
                k_page = key_cache[phys_page, :length]
                k_contiguous[b, :, start:end, :] = k_page.permute(1, 0, 2).half()

        # Step 2: K pooling
        k_pooled = torch.zeros(batch, nkvh, num_pooled_blocks, hd,
                               dtype=torch.float16, device=device)
        esimd_infllmv2_k_pooling(k_contiguous, k_pooled,
                                  nkvh, hd, max_sl, num_pooled_blocks,
                                  kernel_size, kernel_stride)

        # Step 3: Pattern detection
        q_for_pattern = query.view(batch, nh, hd).unsqueeze(2).half()
        block_scores = torch.zeros(batch, nh, 1, num_pooled_blocks,
                                   dtype=torch.float16, device=device)
        kv_block_scores = torch.zeros(batch, nkvh, 1, num_pooled_blocks,
                                      dtype=torch.float16, device=device)
        pooled_scores = torch.zeros(batch, nkvh, 1, num_pooled,
                                    dtype=torch.float16, device=device)
        topk_output = torch.zeros(batch, nkvh, 1, topk,
                                  dtype=torch.int32, device=device)
        esimd_infllmv2_pattern_decode(
            q_for_pattern, k_pooled,
            block_scores, kv_block_scores, pooled_scores, topk_output,
            nh, nkvh, 1, num_pooled_blocks, hd, num_pooled,
            max_sl - 1, 1, init_blocks, local_blocks, topk)

        # Step 4: Force last block (Python loop with .item())
        sparse_mask = topk_output.squeeze(2).int()
        for b in range(batch):
            sl = int(seq_lens_cpu[b].item())
            last_blk = (sl - 1) // sparse_block
            for h in range(nkvh):
                row = sparse_mask[b, h]
                if not (row == last_blk).any().item():
                    sparse_mask[b, h, -1] = last_blk

        # Step 5: Sparse SDP
        output = torch.zeros(batch, nh, hd, dtype=query.dtype, device=device)
        qsl = torch.tensor([0, batch], dtype=torch.int32, device=device)
        dummy_mask_cnt = torch.zeros(1, dtype=torch.int32, device=device)
        esimd_sdp_paged_sparse(
            query, kv_cache, output, block_table, seq_lens, qsl,
            sparse_mask, dummy_mask_cnt,
            nh, nkvh, hd, block_size, max_sl, scale, 1, topk)
        torch.xpu.synchronize()

    for _ in range(warmup):
        run_once()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


def bench_decode_new(kv_cache, block_table, seq_lens, query,
                     nh, nkvh, hd, block_size, scale, topk,
                     kernel_size, kernel_stride, sparse_block,
                     init_blocks, local_blocks,
                     warmup=3, iters=10):
    """New decode pipeline: paged k_pooling + pattern + GPU force_last_block + SDP."""
    batch = seq_lens.shape[0]
    device = query.device
    max_sl = int(seq_lens.max().item())  # one sync here is OK for benchmark
    num_pooled_blocks = max(1, (max_sl - kernel_size + kernel_stride) // kernel_stride)
    pooling_stride = sparse_block // kernel_stride
    num_pooled = max(1, (num_pooled_blocks + pooling_stride - 1) // pooling_stride)

    def run_once():
        # Step 1: Paged K pooling (single GPU kernel)
        k_pooled = torch.zeros(batch, nkvh, num_pooled_blocks, hd,
                               dtype=torch.float16, device=device)
        esimd_infllmv2_k_pooling_paged(
            kv_cache, k_pooled, block_table, seq_lens,
            nkvh, hd, block_size, num_pooled_blocks,
            kernel_size, kernel_stride)

        # Step 2: Pattern detection
        q_for_pattern = query.view(batch, nh, hd).unsqueeze(2).half()
        block_scores = torch.zeros(batch, nh, 1, num_pooled_blocks,
                                   dtype=torch.float16, device=device)
        kv_block_scores = torch.zeros(batch, nkvh, 1, num_pooled_blocks,
                                      dtype=torch.float16, device=device)
        pooled_scores = torch.zeros(batch, nkvh, 1, num_pooled,
                                    dtype=torch.float16, device=device)
        topk_output = torch.zeros(batch, nkvh, 1, topk,
                                  dtype=torch.int32, device=device)
        esimd_infllmv2_pattern_decode(
            q_for_pattern, k_pooled,
            block_scores, kv_block_scores, pooled_scores, topk_output,
            nh, nkvh, 1, num_pooled_blocks, hd, num_pooled,
            max_sl - 1, 1, init_blocks, local_blocks, topk)

        # Step 3: GPU force last block (single tiny kernel)
        sparse_mask = topk_output.squeeze(2).int()
        esimd_infllmv2_force_last_block(sparse_mask, seq_lens, nkvh, sparse_block)

        # Step 4: Sparse SDP
        output = torch.zeros(batch, nh, hd, dtype=query.dtype, device=device)
        qsl = torch.tensor([0, batch], dtype=torch.int32, device=device)
        dummy_mask_cnt = torch.zeros(1, dtype=torch.int32, device=device)
        esimd_sdp_paged_sparse(
            query, kv_cache, output, block_table, seq_lens, qsl,
            sparse_mask, dummy_mask_cnt,
            nh, nkvh, hd, block_size, max_sl, scale, 1, topk)
        torch.xpu.synchronize()

    for _ in range(warmup):
        run_once()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms


if __name__ == '__main__':
    nh, nkvh, hd = 32, 2, 128
    block_size = 128
    scale = 1.0 / math.sqrt(hd)
    topk = 64
    kernel_size, kernel_stride = 32, 16
    sparse_block = 64
    init_blocks, local_blocks = 2, 4

    print("=" * 70)
    print("  Sparse Decode Pipeline: Old (Python K loop) vs New (Paged K Pooling)")
    print("=" * 70)

    for seq_len in [8192, 16384, 32768]:
        batch = 1
        num_pages = (seq_len + block_size - 1) // block_size
        total_pages = num_pages + 4

        kv_cache = torch.randn(2, total_pages, block_size, nkvh, hd,
                               dtype=torch.bfloat16, device='xpu') * 0.3
        block_table = torch.arange(num_pages, dtype=torch.int32, device='xpu').unsqueeze(0)
        bt_padded = torch.zeros(1, total_pages, dtype=torch.int32, device='xpu')
        bt_padded[0, :num_pages] = block_table[0]
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device='xpu')
        query = torch.randn(batch, nh, hd, dtype=torch.bfloat16, device='xpu') * 0.5

        old_ms = bench_decode_old(kv_cache, bt_padded, seq_lens, query,
                                   nh, nkvh, hd, block_size, scale, topk,
                                   kernel_size, kernel_stride, sparse_block,
                                   init_blocks, local_blocks)
        new_ms = bench_decode_new(kv_cache, bt_padded, seq_lens, query,
                                   nh, nkvh, hd, block_size, scale, topk,
                                   kernel_size, kernel_stride, sparse_block,
                                   init_blocks, local_blocks)
        speedup = old_ms / new_ms
        print(f"  seq_len={seq_len:6d}  old={old_ms:7.2f}ms  new={new_ms:7.2f}ms  "
              f"speedup={speedup:.2f}x")

    print()
    print("Done.")
