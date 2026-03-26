"""ULT for ESIMD MoE decode fused kernel.
Tests correctness (vs PyTorch reference) and performance (bandwidth).

Usage:
  python test_moe_decode_esimd.py              # correctness only
  python test_moe_decode_esimd.py --perf       # perf benchmark (MiniCPM5 config)
  python test_moe_decode_esimd.py --all        # both
"""

import argparse
import time
import torch
try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    pass  # torch.xpu works without ipex in newer builds

# ─── PyTorch reference implementation ────────────────────────────────────────

def ref_moe_decode(x, w13_qweight, w13_scales, w2_qweight, w2_scales,
                   topk_weights, topk_ids, group_size):
    """Pure PyTorch reference for MoE decode (W4A16 GPTQ INT4 symmetric).

    x:            [M, K]
    w13_qweight:  [E, 2*N, K/2] uint8
    w13_scales:   [E, 2*N, K/GS] bf16/fp16
    w2_qweight:   [E, K, N/2] uint8
    w2_scales:    [E, K, N/GS] bf16/fp16
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

            # Dequantize up weights [N, K]
            up_qw = w13_qweight[eid, N:, :]  # [N, K/2] uint8
            up_scales = w13_scales[eid, N:, :]  # [N, K/GS]

            # Unpack u4: low nibble at even indices, high nibble at odd indices
            gate_lo = (gate_qw & 0x0F).to(torch.float32)
            gate_hi = ((gate_qw >> 4) & 0x0F).to(torch.float32)
            # Interleave: [N, K] where even K positions = lo, odd = hi
            gate_full = torch.zeros(N, K, dtype=torch.float32, device=device)
            gate_full[:, 0::2] = gate_lo
            gate_full[:, 1::2] = gate_hi

            up_lo = (up_qw & 0x0F).to(torch.float32)
            up_hi = ((up_qw >> 4) & 0x0F).to(torch.float32)
            up_full = torch.zeros(N, K, dtype=torch.float32, device=device)
            up_full[:, 0::2] = up_lo
            up_full[:, 1::2] = up_hi

            # Apply scales: each scale covers group_size elements
            gate_scales_f = gate_scales.float()  # [N, K/GS]
            up_scales_f = up_scales.float()
            gate_scales_expanded = gate_scales_f.repeat_interleave(group_size, dim=1)  # [N, K]
            up_scales_expanded = up_scales_f.repeat_interleave(group_size, dim=1)

            gate_dequant = (gate_full - 8.0) * gate_scales_expanded  # [N, K]
            up_dequant = (up_full - 8.0) * up_scales_expanded

            # gate_out = x[m] @ gate_dequant.T -> [N]
            x_f = x[m].float()
            gate_out = gate_dequant @ x_f  # [N]
            up_out = up_dequant @ x_f  # [N]

            # SiLU(gate) * up
            silu_gate = gate_out * torch.sigmoid(gate_out)
            intermediate = silu_gate * up_out  # [N]

            # Down projection: dequant w2[eid] [K, N]
            down_qw = w2_qweight[eid]  # [K, N/2] uint8
            down_scales = w2_scales[eid]  # [K, N/GS]

            down_lo = (down_qw & 0x0F).to(torch.float32)
            down_hi = ((down_qw >> 4) & 0x0F).to(torch.float32)
            down_full = torch.zeros(K, N, dtype=torch.float32, device=device)
            down_full[:, 0::2] = down_lo
            down_full[:, 1::2] = down_hi

            down_scales_f = down_scales.float()
            down_scales_expanded = down_scales_f.repeat_interleave(group_size, dim=1)

            down_dequant = (down_full - 8.0) * down_scales_expanded  # [K, N]

            down_out = down_dequant @ intermediate  # [K]

            output[m] += w * down_out.to(dtype)

    return output


# ─── Test helpers ────────────────────────────────────────────────────────────

def make_test_data(M, K, N, E, topk, group_size, dtype, device="xpu"):
    """Generate random test data matching vLLM MoE weight layout."""
    x = torch.randn(M, K, dtype=dtype, device=device)

    # INT4 packed weights: random u4 values packed 2 per byte
    w13_qweight = torch.randint(0, 256, (E, 2*N, K//2), dtype=torch.uint8, device=device)
    w2_qweight = torch.randint(0, 256, (E, K, N//2), dtype=torch.uint8, device=device)

    # Scales in bf16/fp16
    w13_scales = torch.randn(E, 2*N, K//group_size, dtype=dtype, device=device) * 0.01
    w2_scales = torch.randn(E, K, N//group_size, dtype=dtype, device=device) * 0.01

    # Routing: random expert selection
    topk_ids = torch.stack([
        torch.randperm(E, device=device)[:topk] for _ in range(M)
    ]).to(torch.int32)
    topk_weights = torch.rand(M, topk, dtype=torch.float32, device=device)
    # Normalize weights per token
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    output = torch.zeros(M, K, dtype=dtype, device=device)

    return x, w13_qweight, w13_scales, w2_qweight, w2_scales, topk_weights, topk_ids, output


def test_correctness(M, K, N, E, topk, group_size, dtype, device="xpu"):
    """Test ESIMD kernel vs PyTorch reference."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_decode

    x, w13_qw, w13_s, w2_qw, w2_s, tw, ti, output = make_test_data(
        M, K, N, E, topk, group_size, dtype, device)

    # Run reference
    ref = ref_moe_decode(x, w13_qw, w13_s, w2_qw, w2_s, tw, ti, group_size)

    # Run ESIMD kernel
    esimd_moe_decode(x, w13_qw, w13_s, w2_qw, w2_s, tw, ti, output, group_size)
    torch.xpu.synchronize()

    # Compare
    diff = (output.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    max_abs_err = diff.max().item()
    rel_rms = (diff ** 2).mean().sqrt().item() / (ref_abs.mean().item() + 1e-8)

    status = "PASS" if rel_rms < 0.05 else "FAIL"
    print(f"  [{status}] M={M}, K={K}, N={N}, E={E}, topk={topk}, GS={group_size}, "
          f"dtype={dtype}: max_abs={max_abs_err:.4f}, rel_rms={rel_rms:.6f}")
    return status == "PASS"


def test_perf(M, K, N, E, topk, group_size, dtype, device="xpu",
              warmup=10, iters=100, num_buffers=32):
    """Benchmark ESIMD MoE decode with cache busting (buffer rotation)."""
    from vllm_kernel_custom.esimd_ops import esimd_moe_decode

    # Create multiple input buffers for cache busting
    buffers = []
    for _ in range(num_buffers):
        data = make_test_data(M, K, N, E, topk, group_size, dtype, device)
        buffers.append(data)

    # Warmup
    for i in range(warmup):
        buf = buffers[i % num_buffers]
        esimd_moe_decode(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], group_size)
    torch.xpu.synchronize()

    # Timed iterations
    start = time.perf_counter()
    for i in range(iters):
        buf = buffers[i % num_buffers]
        esimd_moe_decode(buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], group_size)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / iters * 1e6

    # Calculate memory traffic (bytes)
    # Read: x [M*K*2] + w13_qweight [topk*M * 2*N*K/2] + w13_scales [topk*M * 2*N*K/GS*2]
    #      + w2_qweight [topk*M * K*N/2] + w2_scales [topk*M * K*N/GS*2]
    #      + topk_ids [M*topk*4] + topk_weights [M*topk*4]
    # Write: intermediates [M*topk*N*2] + down_out [M*topk*K*2] + output [M*K*2]
    bytes_per_elem = 2  # bf16/fp16
    total_expert_calls = M * topk

    read_bytes = (
        M * K * bytes_per_elem  # x (read once per kernel, but M*topk times for up and M*topk for down)
        + total_expert_calls * (2 * N * K // 2)  # w13_qweight
        + total_expert_calls * (2 * N * (K // group_size) * bytes_per_elem)  # w13_scales
        + total_expert_calls * (K * N // 2)  # w2_qweight
        + total_expert_calls * (K * (N // group_size) * bytes_per_elem)  # w2_scales
        + M * topk * 4  # topk_ids
        + M * topk * 4  # topk_weights
    )
    write_bytes = (
        total_expert_calls * N * bytes_per_elem  # intermediates
        + total_expert_calls * K * bytes_per_elem  # down_out
        + M * K * bytes_per_elem  # output
    )
    total_bytes = read_bytes + write_bytes
    bandwidth_gbs = total_bytes / (avg_us * 1e-6) / 1e9

    print(f"  M={M}, K={K}, N={N}, E={E}, topk={topk}, GS={group_size}, "
          f"dtype={dtype}: {avg_us:.1f} us, "
          f"traffic={total_bytes/1e6:.2f} MB, BW={bandwidth_gbs:.1f} GB/s")
    return avg_us, bandwidth_gbs


def main():
    parser = argparse.ArgumentParser(description="ESIMD MoE decode kernel ULT")
    parser.add_argument("--perf", action="store_true", help="Run perf benchmark")
    parser.add_argument("--all", action="store_true", help="Run correctness + perf")
    args = parser.parse_args()

    device = "xpu"
    print("=" * 70)
    print("ESIMD MoE Decode Kernel — Unit Level Test")
    print("=" * 70)

    run_correctness = not args.perf or args.all
    run_perf = args.perf or args.all

    if run_correctness:
        print("\n--- Correctness Tests ---")
        all_pass = True

        # Sweep configs
        test_configs = [
            # (M, K, N, E, topk, GS, dtype)
            # Small configs for quick validation
            (1, 128, 64,  8,  2,  32, torch.bfloat16),
            (1, 128, 64,  8,  2,  64, torch.bfloat16),
            (1, 128, 128, 8,  2, 128, torch.bfloat16),
            (1, 128, 64,  8,  2,  32, torch.float16),
            (1, 128, 64,  8,  2,  64, torch.float16),

            # Multi-token decode
            (2, 128, 64,  8,  4,  32, torch.bfloat16),
            (4, 128, 64, 16,  4,  64, torch.bfloat16),
            (8, 256, 128, 16,  4, 128, torch.bfloat16),
            (8, 256, 128, 16,  4, 128, torch.float16),

            # MiniCPM5-like config: K=2048, N=512, E=160, topk=16, GS=128
            (1, 2048, 512, 160, 16, 128, torch.bfloat16),
            (1, 2048, 512, 160, 16, 128, torch.float16),

            # Larger multi-token MiniCPM5
            (4, 2048, 512, 160, 16, 128, torch.bfloat16),
            (8, 2048, 512, 160, 16, 128, torch.bfloat16),
        ]

        for M, K, N, E, topk, gs, dtype in test_configs:
            passed = test_correctness(M, K, N, E, topk, gs, dtype, device)
            if not passed:
                all_pass = False

        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    if run_perf:
        print("\n--- Performance Benchmark ---")
        print("MiniCPM5 config: E=160, K=2048, N=512, topk=16, GS=128")
        print(f"Buffer rotation: 32 buffers for cache busting")
        print()

        for M in [1, 2, 4, 8]:
            for dtype in [torch.bfloat16, torch.float16]:
                test_perf(M, K=2048, N=512, E=160, topk=16,
                          group_size=128, dtype=dtype, device=device)


if __name__ == "__main__":
    main()
