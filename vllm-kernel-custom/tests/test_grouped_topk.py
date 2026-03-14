"""
Test esimd_grouped_topk kernel.

Grouped TopK for MoE gating (DeepSeek-V3 style):
  - Input: gating logits [batch, 256] fp16
  - Apply sigmoid to get scores
  - Add correction bias [256] fp16 for group selection
  - Select top-4 groups (8 groups of 32 experts each)
  - From those 4 groups, select overall top-8 experts
  - Output: topk_weights [batch, 8] fp32, topk_ids [batch, 8] int32

Wrapper parameter mapping to internal function:
  esimd_grouped_topk(scores, output_vals, output_ids, tmp,
                     batch, experts, topk, groups, group_topk, score_scale)
  Internal: (gating_output, correction_bias, topk_weights, topk_ids,
             nullptr, nullptr,
             topk_in, topk_group_in, num_expert_group_in, input_len, renormalize, scale)

  Tensors: scores→gating_output, output_vals→correction_bias,
           output_ids→topk_weights(fp32), tmp→topk_ids(int32)
  Ints reordered internally: topk→topk_in, group_topk→topk_group_in,
           groups→num_expert_group_in, batch→input_len, experts→renormalize

Currently hardcoded: topk=8, group_topk=4, groups=8, num_experts=256.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm-kernel-custom
    python tests/test_grouped_topk.py
"""
import torch

device = torch.device("xpu")


def grouped_topk_ref(logits, correction_bias, topk=8, num_groups=8, group_topk=4):
    """Reference grouped top-K selection in PyTorch."""
    batch = logits.shape[0]
    num_experts = logits.shape[1]
    experts_per_group = num_experts // num_groups

    scores = torch.sigmoid(logits.float())
    scores_for_choice = scores + correction_bias.float()

    scores_grouped = scores_for_choice.view(batch, num_groups, experts_per_group)

    group_top_vals, _ = scores_grouped.topk(topk, dim=-1)
    group_scores = group_top_vals[:, :, :2].sum(dim=-1)

    _, top_group_indices = group_scores.topk(group_topk, dim=-1)

    group_mask = torch.zeros(batch, num_groups, device=logits.device, dtype=torch.bool)
    group_mask.scatter_(1, top_group_indices, True)

    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group)
    expert_mask = expert_mask.reshape(batch, num_experts)
    scores_masked = scores.clone()
    scores_masked[~expert_mask] = -float('inf')

    topk_weights, topk_ids = scores_masked.topk(topk, dim=-1)
    return topk_weights.float(), topk_ids.int()


def test_grouped_topk_basic():
    """Basic grouped top-K with zero bias."""
    from vllm_kernel_custom import esimd_grouped_topk

    batch = 1
    num_experts = 256
    topk = 8
    num_groups = 8
    group_topk = 4

    logits = torch.randn(batch, num_experts, dtype=torch.float16, device=device)

    # Tensor mapping: scores→gating_output, output_vals→correction_bias,
    #                 output_ids→topk_weights(fp32), tmp→topk_ids(int32)
    correction_bias = torch.zeros(num_experts, dtype=torch.float16, device=device)
    topk_weights_out = torch.zeros(batch, topk, dtype=torch.float32, device=device)
    topk_ids_out = torch.zeros(batch, topk, dtype=torch.int32, device=device)

    # Int params: batch, experts(=renormalize), topk, groups, group_topk
    # experts=1 → renormalize=1 (renormalize weights)
    esimd_grouped_topk(logits, correction_bias, topk_weights_out, topk_ids_out,
                       batch, 1, topk, num_groups, group_topk, 1.0)

    ref_weights, ref_ids = grouped_topk_ref(logits, correction_bias, topk, num_groups, group_topk)

    assert (topk_ids_out >= 0).all() and (topk_ids_out < num_experts).all(), \
        f"Expert IDs out of range: min={topk_ids_out.min()}, max={topk_ids_out.max()}"
    assert (topk_weights_out > 0).all(), f"Weights should be positive, got min={topk_weights_out.min()}"

    # Check same expert set selected (order may differ)
    kernel_ids_set = set(topk_ids_out[0].cpu().tolist())
    ref_ids_set = set(ref_ids[0].cpu().tolist())
    assert kernel_ids_set == ref_ids_set, \
        f"Expert IDs mismatch:\n  kernel: {sorted(kernel_ids_set)}\n  ref:    {sorted(ref_ids_set)}"

    # With renormalize=1, kernel normalizes weights so sum~=1
    w_sum = topk_weights_out[0].sum().item()
    assert 0.8 < w_sum < 1.2, f"Renormalized weights should sum to ~1.0, got {w_sum:.4f}"

    print(f"[PASS] test_grouped_topk_basic — batch={batch}, experts={num_experts}")


def test_grouped_topk_with_bias():
    """Grouped top-K with non-zero correction bias."""
    from vllm_kernel_custom import esimd_grouped_topk

    batch = 4
    num_experts = 256
    topk = 8
    num_groups = 8
    group_topk = 4

    logits = torch.randn(batch, num_experts, dtype=torch.float16, device=device) * 2.0
    correction_bias = torch.randn(num_experts, dtype=torch.float16, device=device) * 0.1
    topk_weights_out = torch.zeros(batch, topk, dtype=torch.float32, device=device)
    topk_ids_out = torch.zeros(batch, topk, dtype=torch.int32, device=device)

    esimd_grouped_topk(logits, correction_bias, topk_weights_out, topk_ids_out,
                       batch, 1, topk, num_groups, group_topk, 1.0)

    assert (topk_ids_out >= 0).all() and (topk_ids_out < num_experts).all(), \
        f"Expert IDs out of range: min={topk_ids_out.min()}, max={topk_ids_out.max()}"
    assert (topk_weights_out > 0).all(), f"Weights should be positive"

    for b in range(batch):
        ids = topk_ids_out[b].cpu().tolist()
        assert len(set(ids)) == topk, f"Batch {b}: duplicate expert IDs: {ids}"

    print(f"[PASS] test_grouped_topk_with_bias — batch={batch}, experts={num_experts}")


def test_grouped_topk_weights_are_sigmoid():
    """Verify output weights match sigmoid of the original logits at selected indices."""
    from vllm_kernel_custom import esimd_grouped_topk

    batch = 2
    num_experts = 256
    topk = 8

    logits = torch.randn(batch, num_experts, dtype=torch.float16, device=device)
    correction_bias = torch.zeros(num_experts, dtype=torch.float16, device=device)
    topk_weights_out = torch.zeros(batch, topk, dtype=torch.float32, device=device)
    topk_ids_out = torch.zeros(batch, topk, dtype=torch.int32, device=device)

    # renormalize=0 to get raw sigmoid weights
    esimd_grouped_topk(logits, correction_bias, topk_weights_out, topk_ids_out,
                       batch, 0, topk, 8, 4, 1.0)

    ref_sigmoid = torch.sigmoid(logits.float())
    for b in range(batch):
        for i in range(topk):
            eid = topk_ids_out[b, i].item()
            expected = ref_sigmoid[b, eid].item()
            actual = topk_weights_out[b, i].item()
            assert abs(actual - expected) < 0.05, \
                f"Batch {b}, expert {eid}: weight={actual:.4f}, sigmoid={expected:.4f}"

    print(f"[PASS] test_grouped_topk_weights_are_sigmoid — batch={batch}")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-kernel-custom: Grouped TopK Tests")
    print("=" * 60)
    test_grouped_topk_basic()
    test_grouped_topk_with_bias()
    test_grouped_topk_weights_are_sigmoid()
    print("=" * 60)
    print("ALL GROUPED TOPK TESTS PASSED")
    print("=" * 60)
