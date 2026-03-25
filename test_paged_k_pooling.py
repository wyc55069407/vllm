"""Verify paged K pooling kernel matches old contiguous K pooling."""
import torch
import math

from vllm_kernel_custom import (
    esimd_infllmv2_k_pooling,
    esimd_infllmv2_k_pooling_paged,
    esimd_infllmv2_force_last_block,
)

def test_k_pooling_paged():
    """Compare paged vs contiguous K pooling outputs."""
    nkvh, hd = 2, 128
    page_size = 128
    kernel_size, kernel_stride = 32, 16
    batch = 1

    for seq_len in [1024, 4096, 8192, 16384]:
        num_pages = (seq_len + page_size - 1) // page_size
        num_pooled = (seq_len - kernel_size + kernel_stride) // kernel_stride

        # Create paged KV cache: [2, total_pages, page_size, nkvh, hd]
        total_pages = num_pages + 4  # some extra pages
        kv_cache = torch.randn(2, total_pages, page_size, nkvh, hd,
                               dtype=torch.bfloat16, device='xpu') * 0.3

        # Block table: simple identity mapping with offset
        offset = 2  # physical pages start at offset 2
        block_table = torch.arange(offset, offset + num_pages,
                                   dtype=torch.int32, device='xpu').unsqueeze(0)
        # Pad block_table to expected width
        bt_padded = torch.zeros(1, num_pages + 4, dtype=torch.int32, device='xpu')
        bt_padded[0, :num_pages] = block_table[0]

        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device='xpu')

        # Method 1: Paged K pooling (new kernel)
        k_pooled_paged = torch.zeros(batch, nkvh, num_pooled, hd,
                                     dtype=torch.float16, device='xpu')
        esimd_infllmv2_k_pooling_paged(
            kv_cache, k_pooled_paged,
            bt_padded, seq_lens,
            nkvh, hd, page_size, num_pooled,
            kernel_size, kernel_stride)

        # Method 2: Extract contiguous K + old K pooling
        k_contiguous = torch.zeros(batch, nkvh, seq_len, hd,
                                   dtype=torch.float16, device='xpu')
        key_cache = kv_cache[0]  # [total_pages, page_size, nkvh, hd]
        for p in range(num_pages):
            phys_page = offset + p
            start = p * page_size
            end = min(start + page_size, seq_len)
            length = end - start
            k_page = key_cache[phys_page, :length]  # [length, nkvh, hd]
            k_contiguous[0, :, start:end, :] = k_page.permute(1, 0, 2).half()

        k_pooled_old = torch.zeros(batch, nkvh, num_pooled, hd,
                                   dtype=torch.float16, device='xpu')
        esimd_infllmv2_k_pooling(
            k_contiguous, k_pooled_old,
            nkvh, hd, seq_len, num_pooled,
            kernel_size, kernel_stride)

        torch.xpu.synchronize()

        # Compare
        diff = (k_pooled_paged.float() - k_pooled_old.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_rms = (diff ** 2).mean().sqrt().item() / (k_pooled_old.float().abs().mean().item() + 1e-8)

        status = "PASS" if max_diff < 0.05 else "FAIL"
        print(f"  [{status}] seq_len={seq_len:6d}  num_pooled={num_pooled:4d}  "
              f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  rel_rms={rel_rms:.6f}")


def test_force_last_block():
    """Verify GPU force_last_block kernel."""
    nkvh = 2
    topk = 64

    for batch in [1, 4]:
        for seq_len in [4096, 16384, 32768]:
            last_blk = (seq_len - 1) // 64

            # Create mask WITHOUT last block
            sparse_mask = torch.arange(topk, dtype=torch.int32, device='xpu')
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(batch, nkvh, topk).contiguous()

            seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device='xpu')

            esimd_infllmv2_force_last_block(sparse_mask, seq_lens, nkvh, 64)
            torch.xpu.synchronize()

            # Check: last_blk should be in the mask now
            mask_cpu = sparse_mask.cpu()
            for b in range(batch):
                for h in range(nkvh):
                    has_last = (mask_cpu[b, h] == last_blk).any().item()
                    status = "PASS" if has_last else "FAIL"
                    if b == 0 and h == 0:
                        print(f"  [{status}] batch={batch} seq_len={seq_len} last_blk={last_blk} "
                              f"found={has_last}")

            # Test with last block already present
            sparse_mask2 = sparse_mask.clone()
            sparse_mask2[:, :, 0] = last_blk
            orig_mask = sparse_mask2.clone()
            esimd_infllmv2_force_last_block(sparse_mask2, seq_lens, nkvh, 64)
            torch.xpu.synchronize()
            # Should be unchanged
            match = torch.equal(sparse_mask2, orig_mask)
            print(f"  [{'PASS' if match else 'FAIL'}] Already-present test: unchanged={match}")


if __name__ == '__main__':
    print("=== Paged K Pooling Correctness ===")
    test_k_pooling_paged()
    print()
    print("=== Force Last Block Correctness ===")
    test_force_last_block()
    print()
    print("Done.")
