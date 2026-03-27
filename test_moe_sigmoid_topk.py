"""Test fused sigmoid+topk ESIMD kernel vs PyTorch reference.

Correctness: compare topk_ids and topk_weights against torch.sigmoid + torch.topk + renorm.
Performance: measure speedup of fused kernel over 3-op PyTorch baseline.

Usage:
  python test_moe_sigmoid_topk.py                # correctness
  python test_moe_sigmoid_topk.py --perf          # perf benchmark
  python test_moe_sigmoid_topk.py --tokens 16384  # custom token count
"""
import os, sys, time, argparse
import torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def pytorch_sigmoid_topk(logits, bias, topk):
    """Reference: sigmoid → add bias → topk → renormalize."""
    scores = torch.sigmoid(logits.float())
    if bias is not None and bias.numel() > 0:
        scores = scores + bias.unsqueeze(0)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)
    return topk_weights, topk_ids


def test_correctness(M, E, topk, device="xpu"):
    """Compare ESIMD fused kernel against PyTorch reference."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_sigmoid_topk

    torch.manual_seed(42)
    logits = torch.randn(M, E, dtype=torch.float16, device=device)
    bias = torch.randn(E, dtype=torch.float32, device=device) * 0.01

    # Reference (PyTorch)
    ref_weights, ref_ids = pytorch_sigmoid_topk(logits, bias, topk)

    # ESIMD fused kernel
    out_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
    out_ids = torch.empty(M, topk, dtype=torch.int32, device=device)
    esimd_moe_sigmoid_topk(logits, bias, out_weights, out_ids, E, topk)
    torch.xpu.synchronize()

    # Compare: sort both by expert ID for stable comparison
    ref_sorted_ids, ref_order = ref_ids.sort(dim=-1)
    ref_sorted_weights = ref_weights.gather(1, ref_order)

    out_sorted_ids, out_order = out_ids.sort(dim=-1)
    out_sorted_weights = out_weights.gather(1, out_order)

    # Check IDs match
    id_match = (ref_sorted_ids == out_sorted_ids).all().item()

    # Check weights match (after sorting by same ID order)
    weight_diff = (ref_sorted_weights - out_sorted_weights).abs().max().item()
    weight_rms = ((ref_sorted_weights - out_sorted_weights) ** 2).mean().sqrt().item()

    status = "PASS" if id_match and weight_diff < 0.01 else "FAIL"
    print(f"  M={M:5d}: {status}  ids_match={id_match}  "
          f"weight_max_diff={weight_diff:.6f}  weight_rms={weight_rms:.6f}")

    if not id_match and M <= 4:
        print(f"    ref_ids:  {ref_sorted_ids}")
        print(f"    esimd_ids: {out_sorted_ids}")
        print(f"    ref_w:    {ref_sorted_weights}")
        print(f"    esimd_w:  {out_sorted_weights}")

    return status == "PASS"


def test_no_bias(M, E, topk, device="xpu"):
    """Test with no correction bias (empty tensor)."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_sigmoid_topk

    torch.manual_seed(123)
    logits = torch.randn(M, E, dtype=torch.float16, device=device)
    bias_empty = torch.empty(0, dtype=torch.float32, device=device)

    # Reference without bias
    ref_weights, ref_ids = pytorch_sigmoid_topk(logits, None, topk)

    # ESIMD with empty bias
    out_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
    out_ids = torch.empty(M, topk, dtype=torch.int32, device=device)
    esimd_moe_sigmoid_topk(logits, bias_empty, out_weights, out_ids, E, topk)
    torch.xpu.synchronize()

    ref_sorted_ids, ref_order = ref_ids.sort(dim=-1)
    out_sorted_ids, out_order = out_ids.sort(dim=-1)
    id_match = (ref_sorted_ids == out_sorted_ids).all().item()

    ref_sorted_w = ref_weights.gather(1, ref_order)
    out_sorted_w = out_weights.gather(1, out_order)
    wdiff = (ref_sorted_w - out_sorted_w).abs().max().item()

    status = "PASS" if id_match and wdiff < 0.01 else "FAIL"
    print(f"  M={M:5d} (no bias): {status}  ids_match={id_match}  weight_diff={wdiff:.6f}")
    return status == "PASS"


def bench_perf(M, E, topk, device="xpu", warmup=5, runs=20):
    """Benchmark fused ESIMD vs PyTorch 3-op baseline."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_sigmoid_topk

    torch.manual_seed(42)
    logits = torch.randn(M, E, dtype=torch.float16, device=device)
    bias = torch.randn(E, dtype=torch.float32, device=device) * 0.01
    out_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
    out_ids = torch.empty(M, topk, dtype=torch.int32, device=device)

    # Warmup PyTorch
    for _ in range(warmup):
        pytorch_sigmoid_topk(logits, bias, topk)
    torch.xpu.synchronize()

    # Time PyTorch
    t0 = time.perf_counter()
    for _ in range(runs):
        pytorch_sigmoid_topk(logits, bias, topk)
    torch.xpu.synchronize()
    pytorch_ms = (time.perf_counter() - t0) / runs * 1e3

    # Warmup ESIMD
    for _ in range(warmup):
        esimd_moe_sigmoid_topk(logits, bias, out_weights, out_ids, E, topk)
    torch.xpu.synchronize()

    # Time ESIMD
    t0 = time.perf_counter()
    for _ in range(runs):
        esimd_moe_sigmoid_topk(logits, bias, out_weights, out_ids, E, topk)
    torch.xpu.synchronize()
    esimd_ms = (time.perf_counter() - t0) / runs * 1e3

    speedup = pytorch_ms / esimd_ms
    print(f"  M={M:5d}: PyTorch={pytorch_ms:.3f}ms  ESIMD={esimd_ms:.3f}ms  "
          f"speedup={speedup:.1f}x")
    return pytorch_ms, esimd_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true", help="Run perf benchmark")
    parser.add_argument("--tokens", type=int, default=0, help="Custom token count")
    args = parser.parse_args()

    E = 160   # MiniCPM5 experts
    topk = 16 # MiniCPM5 topk

    if args.perf or args.tokens > 0:
        print(f"\nPerf benchmark: E={E}, topk={topk}")
        token_counts = [args.tokens] if args.tokens > 0 else [1, 8, 64, 512, 4096, 16384]
        for M in token_counts:
            bench_perf(M, E, topk)
    else:
        print(f"\nCorrectness test: E={E}, topk={topk}")
        all_pass = True
        for M in [1, 4, 8, 64, 512, 4096, 16384]:
            all_pass &= test_correctness(M, E, topk)
        # Test without bias
        all_pass &= test_no_bias(64, E, topk)
        all_pass &= test_no_bias(4096, E, topk)
        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
