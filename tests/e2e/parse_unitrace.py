#!/usr/bin/env python3
"""Parse unitrace Chrome JSON traces and extract decode kernel breakdown.

Usage:
  python tests/e2e/parse_unitrace.py /path/to/python.XXXXX.json [--top 30]
  python tests/e2e/parse_unitrace.py /home/sas/yuchen/vllm_env/Perf_minicpm/1K256/python.41547.json
"""
import json, sys, argparse, re
from collections import defaultdict

# MiniCPM5 model parameters for BW/FLOPS estimation
# 28 layers, GQA 32Q/2KV, HD=128, MoE: E=160 topk=16 K=2048 N=512 GS=128
# Non-MoE: hidden=2048, q_proj=[2048,4096], k_proj=[2048,256], v_proj=[2048,256]
#          o_proj=[4096,2048], shared_expert: up[2048,5120], gate[2048,5120], down[5120,2048]

# Bytes per element
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int4": 0.5, "int8": 1}
PEAK_BW_GBS = 450  # BMG peak memory BW in GB/s


def shorten_kernel_name(name):
    """Shorten long kernel names for display."""
    # ESIMD MoE decode kernels
    if "moe_up_forward" in name:
        return "esimd_moe_decode_up_gate_silu"
    if "moe_down_forward" in name:
        return "esimd_moe_decode_down"
    if "esimd_moe_decode" in name:
        if "up_gate_silu" in name:
            return "esimd_moe_decode_up_gate_silu"
        elif "down" in name:
            return "esimd_moe_decode_down"
        elif "gather" in name:
            return "esimd_moe_decode_gather"
        return "esimd_moe_decode"
    # ESIMD MoE prefill kernels
    if "moe_gather_states" in name:
        return "moe_prefill_gather_states"
    if "moe_onednn_forward" in name:
        return "moe_prefill_onednn"
    if "moe_accumulate" in name:
        return "moe_prefill_accumulate"
    if "moe_fused_sigmoid_topk" in name or "esimd_moe_sigmoid_topk" in name:
        return "esimd_moe_sigmoid_topk"
    if "sdp_paged_decode_opt_phase1" in name:
        return "sdp_paged_decode_phase1"
    if "sdp_paged_decode_opt_phase2" in name:
        return "sdp_paged_decode_phase2"
    if "sdp_paged_prefill" in name:
        return "sdp_paged_prefill"
    if "sdp_paged_sparse" in name:
        return "sdp_paged_sparse"
    # oneDNN GEMM
    if "gemm_kernel" in name:
        # Extract SIMD config
        m = re.search(r'SIMD(\d+)\s*\{([^}]+)\}\s*\{([^}]+)\}', name)
        if m:
            return f"onednn_gemm[SIMD{m.group(1)} {{{m.group(2)}}} {{{m.group(3)}}}]"
        return "onednn_gemm"
    if "gen12lp_gemm" in name or "xe_hp_gemm" in name:
        return "onednn_gemm"
    # PyTorch elementwise ops
    if "VectorizedElementwiseKernel" in name:
        if "MulFunctor" in name:
            return "elementwise_mul"
        if "AddFunctor" in name:
            return "elementwise_add"
        if "CopyScalarFunc" in name:
            return "elementwise_copy"
        if "NegFunctor" in name:
            return "elementwise_neg"
        return "elementwise_other"
    if "ElementwiseGlobalRangeKernel" in name:
        if "MulFunctor" in name:
            return "elementwise_mul"
        if "CopyScalarFunc" in name:
            return "elementwise_copy"
        if "NegFunctor" in name:
            return "elementwise_neg"
        return "elementwise_other"
    # Cat/copy
    if "CatArrayBatchedCopy" in name:
        return "cat_batched_copy"
    if "IndexKernel" in name:
        return "index_select"
    # H2D/D2H copy
    if "zeCommandListAppendMemoryCopy" in name:
        m = re.search(r'\((\w+)\)\[(\d+)\]', name)
        if m:
            return f"memcpy_{m.group(1)}[{m.group(2)}B]"
        return "memcpy"
    # ESIMD SDP
    if "esimd_sdp_paged" in name:
        return "esimd_sdp_paged"
    # RMS norm
    if "fused_add_rms_norm" in name:
        return "fused_add_rms_norm"
    if "rms_norm" in name or "RmsNorm" in name:
        return "rms_norm"
    # SiLU
    if "act_and_mul_kernel" in name or "silu_and_mul" in name:
        return "silu_and_mul"
    # Sigmoid (unfused, from MoE routing when ESIMD topk disabled)
    if "SigmoidFunctor" in name:
        return "sigmoid"
    # Fill (zeros)
    if "FillFunctor" in name:
        return "fill_zeros"
    # Softmax
    if "softmax" in name.lower():
        return "softmax"
    # Reshape and cache
    if "reshape_and_cache" in name:
        return "reshape_and_cache"
    # Reduce
    if "reduce" in name.lower() and "kernel" in name.lower():
        return "reduce_kernel"
    # Triton
    if "triton" in name.lower():
        return f"triton:{name[:60]}"
    # Keep short names as-is
    if len(name) < 80:
        return name
    return name[:75] + "..."


def estimate_decode_bw(kernel_name, seq_len=1024):
    """Estimate bytes transferred for decode (M=1) kernels.
    Returns (read_bytes, write_bytes) or None if unknown."""

    # MoE decode: up_gate_silu (M=1, topk=16, E=160, K=2048, N=512, GS=128)
    if "moe_decode_up_gate_silu" in kernel_name:
        # Read: input[1,2048]*fp16 + 16 experts * (gate[512,2048/2]*int4 + up[512,2048/2]*int4
        #        + gate_scales[512,16]*fp32 + up_scales[512,16]*fp32)
        input_bytes = 1 * 2048 * 2  # fp16
        # W4A16: weight is int4 packed, K=2048, N=512, per expert
        w_bytes_per_expert = 2 * (2048 * 512 // 2)  # gate+up, int4 packed
        s_bytes_per_expert = 2 * (2048 // 128 * 512 * 4)  # gate+up scales fp32
        total_w = 16 * (w_bytes_per_expert + s_bytes_per_expert)
        read_b = input_bytes + total_w
        write_b = 1 * 16 * 512 * 2  # output per expert, fp16
        return read_b, write_b

    if "moe_decode_down" in kernel_name:
        # Read: input[16,512]*fp16 + 16 experts * (down[2048,512/2]*int4 + scales)
        input_bytes = 16 * 512 * 2
        w_bytes_per_expert = 2048 * 512 // 2  # int4
        s_bytes_per_expert = 512 // 128 * 2048 * 4  # scales fp32
        total_w = 16 * (w_bytes_per_expert + s_bytes_per_expert)
        read_b = input_bytes + total_w
        write_b = 16 * 2048 * 2  # fp16
        return read_b, write_b

    if "moe_decode_gather" in kernel_name:
        # Read: scattered results [16, 2048] fp16, write: [1, 2048] fp16
        return 16 * 2048 * 2, 1 * 2048 * 2

    # oneDNN GEMM (decode M=1)
    if "onednn_gemm" in kernel_name or "gemm_kernel" in kernel_name:
        # Can't determine exact dims from kernel name alone
        # Typical QKV: M=1, K=2048, N=4096+256+256=4608 (fused) or separate
        # Will return None - need manual annotation
        return None

    # SDP decode phase1: read Q[1,32,128]*fp16 + K_cache[seq,2,128]*fp16
    if "decode_phase1" in kernel_name:
        # Q: 32 heads * 128 * 2 bytes
        q_bytes = 32 * 128 * 2
        # K cache: seq_len * 2 KV heads * 128 * 2 bytes (read by all 32 Q heads)
        k_bytes = seq_len * 2 * 128 * 2
        return q_bytes + k_bytes, 32 * 4 * 2  # partial scores

    if "decode_phase2" in kernel_name:
        # V cache: seq_len * 2 KV heads * 128 * 2 bytes
        v_bytes = seq_len * 2 * 128 * 2
        return v_bytes + 32 * 4, 32 * 128 * 2  # output

    # RMS norm: read + write hidden_states [1, 2048] fp16 + weight [2048] fp16
    if "rms_norm" in kernel_name:
        return 2048 * 2 + 2048 * 2, 2048 * 2

    # sigmoid_topk: read [1, 160] fp32, write [1, 16] int32 + [1, 16] fp32
    if "sigmoid_topk" in kernel_name:
        return 160 * 4, 16 * 4 + 16 * 4

    return None


def parse_chrome_trace(filepath, top_n=30):
    """Parse Chrome trace JSON (streaming to handle large files)."""
    print(f"Parsing {filepath}...")

    # For large files, use streaming JSON parsing
    kernel_stats = defaultdict(lambda: {"count": 0, "total_us": 0.0,
                                         "min_us": float('inf'),
                                         "max_us": 0.0})

    # Detect decode region: look for repeated patterns after prefill
    all_events = []
    event_count = 0
    decode_events = []

    # Stream parse the JSON array
    import ijson
    try:
        with open(filepath, 'rb') as f:
            for event in ijson.items(f, 'traceEvents.item'):
                if event.get('ph') == 'X' and 'name' in event:
                    name = event['name']
                    dur = event.get('dur', 0)  # microseconds
                    ts = event.get('ts', 0)
                    all_events.append((float(ts), name, float(dur)))
                    event_count += 1
                    if event_count % 500000 == 0:
                        print(f"  Processed {event_count} events...",
                              flush=True)
    except ImportError:
        print("ijson not available, using json (slower, needs more RAM)...")
        with open(filepath) as f:
            data = json.load(f)
        events = data.get('traceEvents', data if isinstance(data, list) else [])
        for event in events:
            if event.get('ph') == 'X' and 'name' in event:
                name = event['name']
                dur = event.get('dur', 0)
                ts = event.get('ts', 0)
                all_events.append((ts, name, dur))
                event_count += 1

    print(f"  Total GPU events: {event_count}")

    # Sort by timestamp
    all_events.sort(key=lambda x: x[0])

    # Find decode region: MoE decode kernels (moe_up_forward, moe_down_forward)
    # only appear during decode. Prefill uses moe_gather_states + moe_onednn.
    # Find the FIRST moe_up_forward or moe_down_forward as decode start.
    # Also look for the ESIMD MoE decode pattern.
    decode_start_idx = len(all_events)  # default: no decode found
    for i, (ts, name, dur) in enumerate(all_events):
        if ("moe_up_forward" in name or "moe_down_forward" in name or
                "esimd_moe_decode" in name):
            decode_start_idx = i
            break

    # If no MoE decode kernel found, try fallback: look for repeated short
    # gemm_kernel patterns (M=1 decode GEMV < 50us)
    if decode_start_idx == len(all_events):
        short_gemm_run = 0
        for i, (ts, name, dur) in enumerate(all_events):
            if "gemm_kernel" in name and dur < 50:
                short_gemm_run += 1
                if short_gemm_run >= 10:  # 10 consecutive short GEMMs
                    decode_start_idx = i - 9
                    break
            else:
                short_gemm_run = 0

    if decode_start_idx < len(all_events):
        decode_events = all_events[decode_start_idx:]
        prefill_events = all_events[:decode_start_idx]
        print(f"  Prefill events: {len(prefill_events)}, "
              f"Decode events: {len(decode_events)}")
        print(f"  Decode starts at event #{decode_start_idx} "
              f"(ts={all_events[decode_start_idx][0]:.0f}us)")
    else:
        decode_events = all_events
        prefill_events = []
        print(f"  No clear prefill/decode boundary, "
              f"analyzing all {len(all_events)} events")

    # Aggregate decode kernels
    for ts, name, dur in decode_events:
        short = shorten_kernel_name(name)
        s = kernel_stats[short]
        s["count"] += 1
        s["total_us"] += dur
        s["min_us"] = min(s["min_us"], dur)
        s["max_us"] = max(s["max_us"], dur)

    # Also aggregate prefill
    prefill_stats = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for ts, name, dur in prefill_events:
        short = shorten_kernel_name(name)
        s = prefill_stats[short]
        s["count"] += 1
        s["total_us"] += dur

    return kernel_stats, prefill_stats, len(decode_events)


def print_report(kernel_stats, prefill_stats, n_decode_events,
                 top_n=30, n_decode_steps=256):
    """Print formatted kernel breakdown report."""
    # Sort by total time
    sorted_kernels = sorted(kernel_stats.items(),
                            key=lambda x: x[1]["total_us"], reverse=True)

    total_decode_us = sum(s["total_us"] for _, s in sorted_kernels)
    total_decode_ms = total_decode_us / 1000

    print(f"\n{'='*100}")
    print(f"  DECODE KERNEL BREAKDOWN (top {top_n})")
    print(f"  Total decode GPU time: {total_decode_ms:.1f} ms "
          f"({n_decode_events} kernel launches)")
    if n_decode_steps > 0:
        print(f"  Estimated decode steps: {n_decode_steps}")
        print(f"  Avg per-step GPU time: {total_decode_ms/n_decode_steps:.2f} ms")
    print(f"{'='*100}")
    print(f"{'Kernel':<55} {'Count':>7} {'Total(ms)':>10} {'%':>6} "
          f"{'Avg(us)':>9} {'Min(us)':>9} {'Max(us)':>9} {'GB/s':>7}")
    print(f"{'-'*100}")

    for name, s in sorted_kernels[:top_n]:
        pct = s["total_us"] / total_decode_us * 100 if total_decode_us > 0 else 0
        avg_us = s["total_us"] / s["count"] if s["count"] > 0 else 0

        # Estimate BW
        bw_str = ""
        bw_est = estimate_decode_bw(name)
        if bw_est and avg_us > 0:
            read_b, write_b = bw_est
            total_bytes = read_b + write_b
            bw_gbs = total_bytes / (avg_us * 1e-6) / 1e9
            bw_pct = bw_gbs / PEAK_BW_GBS * 100
            bw_str = f"{bw_gbs:.0f}({bw_pct:.0f}%)"

        print(f"{name:<55} {s['count']:>7} {s['total_us']/1000:>10.2f} "
              f"{pct:>5.1f}% {avg_us:>9.1f} {s['min_us']:>9.1f} "
              f"{s['max_us']:>9.1f} {bw_str:>7}")

    # Print categorized summary
    print(f"\n{'='*100}")
    print(f"  DECODE TIME BY CATEGORY")
    print(f"{'='*100}")

    categories = {
        "MoE (routed experts)": ["esimd_moe_decode"],
        "MoE routing": ["sigmoid_topk", "sigmoid"],
        "MoE prefill": ["moe_prefill"],
        "Attention (SDP)": ["sdp_paged", "decode_phase", "esimd_sdp"],
        "oneDNN GEMM (QKV/O/shared)": ["onednn_gemm", "gemm_kernel"],
        "RoPE (unfused elemwise)": ["elementwise_mul", "elementwise_neg",
                                    "elementwise_copy", "index_select"],
        "Residual add": ["elementwise_add"],
        "RMS Norm": ["rms_norm"],
        "SiLU+Mul (shared expert)": ["silu_and_mul"],
        "KV cache update": ["reshape_and_cache", "cat_batched", "memcpy"],
        "Fill/zeros": ["fill_zeros"],
        "Other": [],
    }

    cat_times = {}
    assigned = set()
    for cat_name, patterns in categories.items():
        if cat_name == "Other":
            continue
        t = 0
        for kname, s in sorted_kernels:
            if any(p in kname for p in patterns):
                t += s["total_us"]
                assigned.add(kname)
        cat_times[cat_name] = t

    # Other = everything not assigned
    other_t = sum(s["total_us"] for kname, s in sorted_kernels
                  if kname not in assigned)
    cat_times["Other"] = other_t

    for cat_name, t_us in sorted(cat_times.items(),
                                  key=lambda x: x[1], reverse=True):
        pct = t_us / total_decode_us * 100 if total_decode_us > 0 else 0
        per_step = t_us / n_decode_steps / 1000 if n_decode_steps > 0 else 0
        print(f"  {cat_name:<35} {t_us/1000:>8.2f} ms ({pct:>5.1f}%) "
              f"  {per_step:.3f} ms/step")

    # Prefill summary
    if prefill_stats:
        total_prefill_us = sum(s["total_us"]
                               for _, s in prefill_stats.items())
        print(f"\n{'='*100}")
        print(f"  PREFILL KERNEL SUMMARY (top 15)")
        print(f"  Total prefill GPU time: {total_prefill_us/1000:.1f} ms")
        print(f"{'='*100}")
        sorted_prefill = sorted(prefill_stats.items(),
                                key=lambda x: x[1]["total_us"], reverse=True)
        for name, s in sorted_prefill[:15]:
            pct = s["total_us"] / total_prefill_us * 100
            print(f"  {name:<55} {s['count']:>7} "
                  f"{s['total_us']/1000:>10.2f} ms ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Parse unitrace Chrome JSON for kernel breakdown")
    parser.add_argument("trace", help="Path to Chrome trace JSON file")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N kernels (default: 30)")
    parser.add_argument("--decode-steps", type=int, default=256,
                        help="Number of decode steps (default: 256)")
    args = parser.parse_args()

    kernel_stats, prefill_stats, n_events = parse_chrome_trace(
        args.trace, args.top)
    print_report(kernel_stats, prefill_stats, n_events,
                 top_n=args.top, n_decode_steps=args.decode_steps)


if __name__ == "__main__":
    main()
