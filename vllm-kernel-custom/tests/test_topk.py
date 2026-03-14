"""
Test esimd_topk_multi_round kernel.

Multi-round top-K selection for DSA (Dynamic Sparse Attention):
  - Input: scores [batch, real_total_count_reserved] fp16
  - Split into groups, each group has total_count_stride elements
  - Round 1: per-group top-2048 selection
  - Round 2: merge all groups, final top-2048 selection
  - Output: final top-2048 indices [batch, topk] uint32

Parameters mapping (from wrapper):
  t0 (i0) = input scores [batch, reserved] fp16
  t1 (i1) = input scores (same, unused as idx in first round)
  t2 (i2) = out_ordered tmp [batch, groups, topk] fp16
  t3 (i3) = out_idx tmp [batch, groups, topk] uint32
  t4 (i4) = final_out_ordered [batch, topk] fp16  (if output_final_out=1)
  t5 (i5) = final_out_idx [batch, topk] uint32
  p0 = total_count_stride (aligned per-group count)
  p1 = groups (RCOUNT: number of groups, 1-10)
  p2 = topk (must be 2048)
  p3 = output_final_out (0 or 1)
  p4 = batch_n

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_topk.py
"""
import torch

device = torch.device("xpu")


def compute_groups(real_total_count):
    """Compute number of groups based on total count (matches kernel logic)."""
    groups = 1
    if real_total_count > 20480:
        groups = 4
    if real_total_count > 30720:
        groups = 6
    if real_total_count > 40960:
        groups = 8
    if real_total_count > 51200:
        groups = 10
    return groups


def test_topk_multi_round_correctness():
    """Verify top-2048 selection returns the correct highest-scoring indices."""
    from vllm_kernel_custom import esimd_topk_multi_round

    batch_n = 2
    real_total_count = 32667
    real_total_count_reserved = 160 * 1024
    topk = 2048

    groups = compute_groups(real_total_count)
    total_count_stride = ((real_total_count // groups) + 4095) // 4096 * 4096
    real_total_count_stride_aligned = groups * total_count_stride

    # Fill reserved buffer with -inf, then fill valid range with random scores
    index_score_rsv = torch.full((batch_n, real_total_count_reserved),
                                  -65504.0, dtype=torch.float16, device=device)
    scores = torch.rand(batch_n, real_total_count, dtype=torch.float16, device=device) * 10 - 5
    index_score_rsv[:, :real_total_count] = scores

    # Temporary buffers
    out_ordered = torch.zeros(batch_n, 10, topk, dtype=torch.float16, device=device)
    out_idx = torch.zeros(batch_n, 10, topk, dtype=torch.uint32, device=device)

    # Final output buffers
    final_out_idx = torch.zeros(batch_n, topk, dtype=torch.uint32, device=device)

    # Unused final_out_ordered (output_final_out=0 means only indices are written)
    final_out_ordered = torch.zeros(batch_n, topk, dtype=torch.float16, device=device)

    # p0=total_count_stride, p1=groups, p2=topk, p3=output_final_out, p4=batch_n
    esimd_topk_multi_round(
        index_score_rsv,         # t0: input scores
        index_score_rsv,         # t1: input (used as idx source in first round)
        out_ordered,             # t2: tmp ordered
        out_idx,                 # t3: tmp idx
        final_out_ordered,       # t4: final ordered (unused when output_final_out=0)
        final_out_idx,           # t5: final indices output
        total_count_stride,      # p0
        groups,                  # p1 (RCOUNT)
        topk,                    # p2
        0,                       # p3 (output_final_out=0, only indices)
        batch_n,                 # p4
    )

    # Reference: per-group top-k then merge
    # Reshape into groups view
    scores_grouped = index_score_rsv[:, :real_total_count_stride_aligned].view(
        batch_n, groups, total_count_stride)

    for b in range(batch_n):
        # Per-group top-2048
        per_group_vals, per_group_idx = scores_grouped[b].topk(topk, dim=-1)

        # Convert per-group local indices to global indices
        global_indices = []
        global_vals = []
        for g in range(groups):
            global_indices.append(per_group_idx[g] + g * total_count_stride)
            global_vals.append(per_group_vals[g])

        merged_vals = torch.cat(global_vals)   # [groups * topk]
        merged_idx = torch.cat(global_indices)  # [groups * topk]

        # Final top-2048 from merged
        _, final_top_pos = merged_vals.topk(topk)
        ref_final_idx = merged_idx[final_top_pos]

        # Get kernel output indices
        kernel_idx = final_out_idx[b].long()

        # Both should select the same set of high-scoring elements
        # (order may differ, so compare the actual score values)
        kernel_scores = index_score_rsv[b, kernel_idx].float()
        ref_scores = index_score_rsv[b, ref_final_idx.long()].float()

        # Sort both by score for comparison
        kernel_sorted = kernel_scores.sort(descending=True).values
        ref_sorted = ref_scores.sort(descending=True).values

        torch.testing.assert_close(kernel_sorted, ref_sorted, rtol=1e-2, atol=1e-2)

    print(f"[PASS] test_topk_multi_round_correctness — batch={batch_n}, "
          f"total_count={real_total_count}, groups={groups}, topk={topk}")


def test_topk_multi_round_single_group():
    """Test with single group (total_count <= 20480, groups=1)."""
    from vllm_kernel_custom import esimd_topk_multi_round

    batch_n = 1
    # With total_count <= 20480, groups=1 and total_count_stride must be one of
    # {4096, 8192, 12288, 16384, 20480}
    real_total_count = 8000
    real_total_count_reserved = 160 * 1024
    topk = 2048

    groups = compute_groups(real_total_count)
    assert groups == 1, f"Expected 1 group, got {groups}"
    total_count_stride = ((real_total_count // groups) + 4095) // 4096 * 4096
    assert total_count_stride == 8192

    index_score_rsv = torch.full((batch_n, real_total_count_reserved),
                                  -65504.0, dtype=torch.float16, device=device)
    scores = torch.rand(batch_n, real_total_count, dtype=torch.float16, device=device) * 10 - 5
    index_score_rsv[:, :real_total_count] = scores

    out_ordered = torch.zeros(batch_n, 10, topk, dtype=torch.float16, device=device)
    out_idx = torch.zeros(batch_n, 10, topk, dtype=torch.uint32, device=device)
    final_out_ordered = torch.zeros(batch_n, topk, dtype=torch.float16, device=device)
    final_out_idx = torch.zeros(batch_n, topk, dtype=torch.uint32, device=device)

    esimd_topk_multi_round(
        index_score_rsv, index_score_rsv,
        out_ordered, out_idx,
        final_out_ordered, final_out_idx,
        total_count_stride, groups, topk, 0, batch_n,
    )

    # Reference: simple top-2048 from total_count_stride elements
    ref_scores = index_score_rsv[0, :total_count_stride]
    _, ref_idx = ref_scores.topk(topk)

    kernel_idx = final_out_idx[0].long()
    kernel_scores = index_score_rsv[0, kernel_idx].float().sort(descending=True).values
    ref_scores_sorted = ref_scores[ref_idx].float().sort(descending=True).values

    torch.testing.assert_close(kernel_scores, ref_scores_sorted, rtol=1e-2, atol=1e-2)

    print(f"[PASS] test_topk_multi_round_single_group — batch={batch_n}, "
          f"total_count={real_total_count}, groups={groups}")


def test_topk_multi_round_large():
    """Test with larger count (groups=8)."""
    from vllm_kernel_custom import esimd_topk_multi_round

    batch_n = 4
    real_total_count = 45000
    real_total_count_reserved = 160 * 1024
    topk = 2048

    groups = compute_groups(real_total_count)
    total_count_stride = ((real_total_count // groups) + 4095) // 4096 * 4096
    real_total_count_stride_aligned = groups * total_count_stride

    index_score_rsv = torch.full((batch_n, real_total_count_reserved),
                                  -65504.0, dtype=torch.float16, device=device)
    scores = torch.rand(batch_n, real_total_count, dtype=torch.float16, device=device) * 10 - 5
    index_score_rsv[:, :real_total_count] = scores

    out_ordered = torch.zeros(batch_n, 10, topk, dtype=torch.float16, device=device)
    out_idx = torch.zeros(batch_n, 10, topk, dtype=torch.uint32, device=device)
    final_out_ordered = torch.zeros(batch_n, topk, dtype=torch.float16, device=device)
    final_out_idx = torch.zeros(batch_n, topk, dtype=torch.uint32, device=device)

    esimd_topk_multi_round(
        index_score_rsv, index_score_rsv,
        out_ordered, out_idx,
        final_out_ordered, final_out_idx,
        total_count_stride, groups, topk, 0, batch_n,
    )

    # Verify: kernel output should contain top-2048 scores
    for b in range(batch_n):
        kernel_idx = final_out_idx[b].long()
        kernel_scores = index_score_rsv[b, kernel_idx].float()

        # The 2048th largest score should be close to or above the min kernel score
        all_scores = index_score_rsv[b, :real_total_count_stride_aligned].float()
        ref_topk_min = all_scores.topk(topk).values[-1]
        kernel_min = kernel_scores.min()

        # Allow small tolerance due to fp16 precision in radix sort
        assert kernel_min >= ref_topk_min - 0.1, \
            f"Batch {b}: kernel min score {kernel_min:.4f} < ref threshold {ref_topk_min:.4f}"

    print(f"[PASS] test_topk_multi_round_large — batch={batch_n}, "
          f"total_count={real_total_count}, groups={groups}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: TopK Multi-Round Tests")
    print("=" * 60)
    test_topk_multi_round_correctness()
    test_topk_multi_round_single_group()
    test_topk_multi_round_large()
    print("=" * 60)
    print("ALL TOPK TESTS PASSED")
    print("=" * 60)
