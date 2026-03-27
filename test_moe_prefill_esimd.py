"""ULT for ESIMD MoE prefill fused kernel.
Tests correctness (vs PyTorch reference) and performance.

Usage:
  python test_moe_prefill_esimd.py              # correctness only
  python test_moe_prefill_esimd.py --perf       # perf benchmark
  python test_moe_prefill_esimd.py --all        # both
  python test_moe_prefill_esimd.py --unitrace   # single run for unitrace profiling (16K tokens)

Unitrace profiling:
  UNITRACE=/home/sas/yuchen/vllm_env/pti-gpu/tools/unitrace/build/unitrace
  $UNITRACE -d python test_moe_prefill_esimd.py --unitrace
"""

import argparse
import time
import torch
try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    pass


# ─── PyTorch reference implementation ────────────────────────────────────────

def ref_moe_prefill(x, w13_qweight, w13_scales, w2_qweight, w2_scales,
                    topk_weights, topk_ids, group_size):
    """Pure PyTorch reference for MoE prefill (W4A16 GPTQ INT4 symmetric).

    x:            [M, K]
    w13_qweight:  [E, 2*N, K/2] uint8
    w13_scales:   [E, 2*N, K/GS] fp16
    w2_qweight:   [E, K, N/2] uint8
    w2_scales:    [E, K, N/GS] fp16
    topk_weights: [M, topk] float32
    topk_ids:     [M, topk] int32
    """
    M, K = x.shape
    topk = topk_ids.shape[1]
    N = w13_qweight.shape[1] // 2
    dtype = x.dtype
    device = x.device

    output = torch.zeros(M, K, dtype=dtype, device=device)

    for m in range(M):
        for s in range(topk):
            eid = topk_ids[m, s].item()
            w = topk_weights[m, s].item()

            # Dequantize gate weights [N, K]
            gate_qw = w13_qweight[eid, :N, :]  # [N, K/2] uint8
            gate_scales = w13_scales[eid, :N, :]  # [N, K/GS]

            up_qw = w13_qweight[eid, N:, :]  # [N, K/2] uint8
            up_scales = w13_scales[eid, N:, :]  # [N, K/GS]

            gate_lo = (gate_qw & 0x0F).to(torch.float32)
            gate_hi = ((gate_qw >> 4) & 0x0F).to(torch.float32)
            gate_full = torch.zeros(N, K, dtype=torch.float32, device=device)
            gate_full[:, 0::2] = gate_lo
            gate_full[:, 1::2] = gate_hi

            up_lo = (up_qw & 0x0F).to(torch.float32)
            up_hi = ((up_qw >> 4) & 0x0F).to(torch.float32)
            up_full = torch.zeros(N, K, dtype=torch.float32, device=device)
            up_full[:, 0::2] = up_lo
            up_full[:, 1::2] = up_hi

            gate_scales_f = gate_scales.float()
            up_scales_f = up_scales.float()
            gate_scales_expanded = gate_scales_f.repeat_interleave(group_size, dim=1)
            up_scales_expanded = up_scales_f.repeat_interleave(group_size, dim=1)

            gate_dequant = (gate_full - 8.0) * gate_scales_expanded
            up_dequant = (up_full - 8.0) * up_scales_expanded

            x_f = x[m].float()
            gate_out = gate_dequant @ x_f
            up_out = up_dequant @ x_f

            silu_gate = gate_out * torch.sigmoid(gate_out)
            intermediate_val = silu_gate * up_out

            # Down projection
            down_qw = w2_qweight[eid]
            down_scales = w2_scales[eid]

            down_lo = (down_qw & 0x0F).to(torch.float32)
            down_hi = ((down_qw >> 4) & 0x0F).to(torch.float32)
            down_full = torch.zeros(K, N, dtype=torch.float32, device=device)
            down_full[:, 0::2] = down_lo
            down_full[:, 1::2] = down_hi

            down_scales_f = down_scales.float()
            down_scales_expanded = down_scales_f.repeat_interleave(group_size, dim=1)

            down_dequant = (down_full - 8.0) * down_scales_expanded

            down_out = down_dequant @ intermediate_val

            output[m] += w * down_out.to(dtype)

    return output


# ─── Test helpers ────────────────────────────────────────────────────────────

def make_test_data(M, K, N, E, topk, group_size, dtype, device="xpu"):
    """Generate random test data."""
    x = torch.randn(M, K, dtype=dtype, device=device)

    w13_qweight = torch.randint(0, 256, (E, 2*N, K//2), dtype=torch.uint8, device=device)
    w2_qweight = torch.randint(0, 256, (E, K, N//2), dtype=torch.uint8, device=device)

    w13_scales = torch.randn(E, 2*N, K//group_size, dtype=dtype, device=device) * 0.01
    w2_scales = torch.randn(E, K, N//group_size, dtype=dtype, device=device) * 0.01

    # Transposed scales for oneDNN
    w13_scales_t = w13_scales.permute(0, 2, 1).contiguous()
    w2_scales_t = w2_scales.permute(0, 2, 1).contiguous()

    # Routing: random expert selection
    topk_ids = torch.stack([
        torch.randperm(E, device=device)[:topk] for _ in range(M)
    ]).to(torch.int32)
    topk_weights = torch.rand(M, topk, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    output = torch.zeros(M, K, dtype=dtype, device=device)

    return (x, w13_qweight, w13_scales, w13_scales_t,
            w2_qweight, w2_scales, w2_scales_t,
            topk_weights, topk_ids, output)


def test_correctness(M, K, N, E, topk, group_size, dtype, device="xpu"):
    """Test ESIMD prefill kernel vs PyTorch reference."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_prefill

    data = make_test_data(M, K, N, E, topk, group_size, dtype, device)
    (x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t, tw, ti, output) = data

    # Run reference (only first few tokens for large M to save time)
    ref_M = min(M, 4)
    ref = ref_moe_prefill(x[:ref_M], w13_qw, w13_s, w2_qw, w2_s,
                           tw[:ref_M], ti[:ref_M], group_size)

    # Run ESIMD kernel
    esimd_moe_prefill(x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t,
                       tw, ti, output, group_size)
    torch.xpu.synchronize()

    # Compare only ref_M tokens
    out_cmp = output[:ref_M]
    diff = (out_cmp.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    max_abs_err = diff.max().item()
    rel_rms = (diff ** 2).mean().sqrt().item() / (ref_abs.mean().item() + 1e-8)

    status = "PASS" if rel_rms < 0.05 else "FAIL"
    print(f"  [{status}] M={M}, K={K}, N={N}, E={E}, topk={topk}, GS={group_size}, "
          f"dtype={dtype}: max_abs={max_abs_err:.4f}, rel_rms={rel_rms:.6f} "
          f"(checked {ref_M}/{M} tokens)")
    return status == "PASS"


def compute_moe_traffic(M, K, N, E, topk, group_size):
    """Compute bytes + FLOPs matching C++ reference (moe_prefill_v29.cpp).

    Assumes uniform distribution (all E experts active, ~M*topk/E tokens each).
    Returns (total_flops, total_bytes, weight_bytes, scale_bytes, act_bytes).
    """
    total_seqlen = M * topk
    fused_im = 2 * N
    nblk_h = K // group_size
    nblk_im = N // group_size

    # FLOPs: gate(2*H*IM) + up(2*H*IM) + down(2*IM*H) = 6*H*IM per token-expert pair
    total_flops = 6.0 * total_seqlen * K * N

    # Weight bytes: per active expert, read once regardless of nt
    # guw: [2*IM, H/2] uint8 = 2*N*(K/2)
    # down: [H, IM/2] uint8 = K*(N/2)
    total_weight_bytes = E * (fused_im * (K // 2) + K * (N // 2))

    # Scale bytes: per active expert
    # guw: [2*IM, nblk_h] fp16 = 2*N*nblk_h*2
    # down: [H, nblk_im] fp16 = K*nblk_im*2
    total_scale_bytes = E * (fused_im * nblk_h * 2 + K * nblk_im * 2)

    # Activation bytes:
    # expert_states read:    total_seqlen * H * 2
    # gate_buf write:        total_seqlen * 2*IM * 2
    # gate_buf read (silu):  total_seqlen * 2*IM * 2
    # intermediate write:    total_seqlen * IM * 2
    # intermediate read:     total_seqlen * IM * 2
    # output write:          total_seqlen * H * 2
    total_act_bytes = (total_seqlen * K * 2            # input read
                     + total_seqlen * fused_im * 2     # gate_buf write
                     + total_seqlen * fused_im * 2     # gate_buf read
                     + total_seqlen * N * 2            # intermediate write
                     + total_seqlen * N * 2            # intermediate read
                     + total_seqlen * K * 2)           # output write

    total_bytes = total_weight_bytes + total_scale_bytes + total_act_bytes
    return total_flops, total_bytes, total_weight_bytes, total_scale_bytes, total_act_bytes


def test_perf(M, K, N, E, topk, group_size, dtype, device="xpu",
              warmup=5, iters=20):
    """Benchmark ESIMD MoE prefill with proper bytes/TFLOPS."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_prefill

    data = make_test_data(M, K, N, E, topk, group_size, dtype, device)
    (x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t, tw, ti, output) = data

    # Warmup
    for _ in range(warmup):
        output.zero_()
        esimd_moe_prefill(x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t,
                           tw, ti, output, group_size)
    torch.xpu.synchronize()

    # Timed
    times = []
    for _ in range(iters):
        output.zero_()
        start = time.perf_counter()
        esimd_moe_prefill(x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t,
                           tw, ti, output, group_size)
        torch.xpu.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1e3
    best_ms = min(times) * 1e3
    total_seqlen = M * topk

    total_flops, total_bytes, wt_bytes, sc_bytes, act_bytes = \
        compute_moe_traffic(M, K, N, E, topk, group_size)

    avg_tflops = total_flops / (avg_ms * 1e-3) / 1e12
    avg_gbps = total_bytes / (avg_ms * 1e-3) / 1e9
    best_tflops = total_flops / (best_ms * 1e-3) / 1e12
    best_gbps = total_bytes / (best_ms * 1e-3) / 1e9

    print(f"  M={M} ({total_seqlen} total_seqlen):")
    print(f"    avg: {avg_ms:.2f} ms  {avg_tflops:.2f} TFLOPS  {avg_gbps:.1f} GB/s")
    print(f"    best: {best_ms:.2f} ms  {best_tflops:.2f} TFLOPS  {best_gbps:.1f} GB/s")
    print(f"    traffic: weight={wt_bytes/1e6:.1f}MB  scale={sc_bytes/1e6:.1f}MB  "
          f"act={act_bytes/1e6:.1f}MB  total={total_bytes/1e6:.1f}MB")
    return avg_ms, avg_tflops


def test_unitrace(M, K, N, E, topk, group_size, dtype, device="xpu",
                  warmup=3, iters=5):
    """Single-config run for unitrace profiling. Prints kernel timing summary."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_prefill

    total_seqlen = M * topk
    total_flops, total_bytes, _, _, _ = compute_moe_traffic(M, K, N, E, topk, group_size)

    print(f"Config: M={M}, total_seqlen={total_seqlen}, E={E}, K={K}, N={N}, topk={topk}, GS={group_size}")
    print(f"Expected: FLOPs={total_flops:.3e}  Bytes={total_bytes:.3e}")
    print(f"  (Use unitrace -d to see per-kernel device time)\n")

    data = make_test_data(M, K, N, E, topk, group_size, dtype, device)
    (x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t, tw, ti, output) = data

    # Warmup
    for _ in range(warmup):
        output.zero_()
        esimd_moe_prefill(x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t,
                           tw, ti, output, group_size)
    torch.xpu.synchronize()

    # Measured runs
    for r in range(iters):
        output.zero_()
        start = time.perf_counter()
        esimd_moe_prefill(x, w13_qw, w13_s, w13_s_t, w2_qw, w2_s, w2_s_t,
                           tw, ti, output, group_size)
        torch.xpu.synchronize()
        wall_ms = (time.perf_counter() - start) * 1e3
        tflops = total_flops / (wall_ms * 1e-3) / 1e12
        gbps = total_bytes / (wall_ms * 1e-3) / 1e9
        print(f"  Run {r}: {wall_ms:.2f} ms  {tflops:.2f} TFLOPS  {gbps:.1f} GB/s")

    print(f"\nTo compute from unitrace kernel time:")
    print(f"  TFLOPS = {total_flops:.3e} / (kernel_time_ns * 1e-9) / 1e12")
    print(f"  GB/s   = {total_bytes:.3e} / (kernel_time_ns * 1e-9) / 1e9")


def main():
    parser = argparse.ArgumentParser(description="ESIMD MoE prefill kernel ULT")
    parser.add_argument("--perf", action="store_true", help="Run perf benchmark")
    parser.add_argument("--all", action="store_true", help="Run correctness + perf")
    parser.add_argument("--unitrace", action="store_true",
                        help="Single run for unitrace profiling (16K tokens)")
    parser.add_argument("--tokens", type=int, default=16384,
                        help="Number of tokens for unitrace mode (default: 16384)")
    args = parser.parse_args()

    device = "xpu"

    if args.unitrace:
        print("=" * 70)
        print("ESIMD MoE Prefill — unitrace profiling mode")
        print("=" * 70)
        test_unitrace(args.tokens, K=2048, N=512, E=160, topk=16,
                      group_size=128, dtype=torch.float16, device=device)
        return

    print("=" * 70)
    print("ESIMD MoE Prefill Kernel — Unit Level Test")
    print("=" * 70)

    run_correctness = not args.perf or args.all
    run_perf = args.perf or args.all

    if run_correctness:
        print("\n--- Correctness Tests ---")
        all_pass = True

        # MiniCPM5-like configs (GGEMV requires K>=256, N>=32 for 2D loads)
        test_configs = [
            # (M, K, N, E, topk, GS, dtype)
            (64, 2048, 512, 160, 16, 128, torch.float16),
            (128, 2048, 512, 160, 16, 128, torch.float16),
            (512, 2048, 512, 160, 16, 128, torch.float16),
        ]

        for M, K, N, E, topk, gs, dtype in test_configs:
            passed = test_correctness(M, K, N, E, topk, gs, dtype, device)
            if not passed:
                all_pass = False

        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    if run_perf:
        print("\n--- Performance Benchmark ---")
        print("MiniCPM5: E=160, K=2048, N=512, topk=16, GS=128")
        print()

        for M in [512, 4096, 8192, 16384]:
            test_perf(M, K=2048, N=512, E=160, topk=16,
                      group_size=128, dtype=torch.float16, device=device)
            print()


if __name__ == "__main__":
    main()
