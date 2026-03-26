"""Measure MoE decode speedup: ESIMD kernel vs baseline.

Short prompt + 512 decode tokens, batch=1, 3 timed runs.
Uses MINICPM5_PROFILE=1 for per-layer attn/MoE breakdown.

Usage:
  # Baseline (no ESIMD MoE):
  python test_minicpm5_moe_speedup.py

  # ESIMD MoE decode:
  MINICPM5_ESIMD_MOE=1 python test_minicpm5_moe_speedup.py
"""
import os, sys, time, json, collections
os.environ["MINICPM5_PROFILE"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
PROFILE_PATH = "/tmp/minicpm5_profile.jsonl"
os.environ["MINICPM5_PROFILE_PATH"] = PROFILE_PATH

NUM_RUNS = 3
MAX_TOKENS = 512


def read_profile_decode_stats(prev_len=0):
    """Read profile, return (avg_decode_step_ms, avg_attn_ms, avg_moe_ms, n_decode_steps)."""
    if not os.path.exists(PROFILE_PATH):
        return None
    with open(PROFILE_PATH) as f:
        all_rows = [json.loads(line) for line in f]
    rows = all_rows[prev_len:]
    if not rows:
        return None

    # Group by step
    by_step = collections.defaultdict(list)
    for step, layer_idx, attn_ms, mlp_ms, total_ms, n_tokens in rows:
        by_step[step].append((layer_idx, attn_ms, mlp_ms, total_ms, n_tokens))

    # Separate prefill vs decode
    decode_steps = {s: v for s, v in by_step.items() if v[0][4] == 1}
    prefill_steps = {s: v for s, v in by_step.items() if v[0][4] > 1}

    result = {"new_rows": len(rows), "total_rows": len(all_rows)}

    for phase, steps_dict in [("prefill", prefill_steps), ("decode", decode_steps)]:
        if not steps_dict:
            continue
        attn_total = mlp_total = all_total = 0.0
        for s, layers in steps_dict.items():
            for _, attn_ms, mlp_ms, total_ms, _ in layers:
                attn_total += attn_ms
                mlp_total += mlp_ms
                all_total += total_ms
        n = len(steps_dict)
        result[phase] = {
            "n_steps": n,
            "avg_step_ms": all_total / n,
            "avg_attn_ms": attn_total / n,
            "avg_moe_ms": mlp_total / n,
            "avg_other_ms": (all_total - attn_total - mlp_total) / n,
        }
    return result


if __name__ == "__main__":
    esimd_moe = os.environ.get("MINICPM5_ESIMD_MOE", "0") == "1"
    mode_label = "ESIMD MoE" if esimd_moe else "Baseline"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    msgs = [{"role": "user", "content": "What is 2+2? Explain in detail."}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n</think>\n\n"

    # Clean profile
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)

    print(f"{'='*70}")
    print(f"  MiniCPM5 Decode Speed — {mode_label}")
    print(f"  Short prompt + {MAX_TOKENS} decode tokens, batch=1, {NUM_RUNS} runs")
    print(f"{'='*70}")

    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.75,
        max_model_len=2048, max_num_seqs=1, block_size=128,
        attention_backend="ESIMD_ATTN",
    )

    # Warmup
    print("\nWarmup...")
    warmup_params = SamplingParams(temperature=0, max_tokens=8)
    _ = llm.generate([prompt], warmup_params)
    time.sleep(0.5)

    # Read warmup profile offset
    prev_len = 0
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH) as f:
            prev_len = sum(1 for _ in f)
    print(f"Warmup done (profile rows: {prev_len})\n")

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS,
                            min_tokens=MAX_TOKENS, ignore_eos=True)
    run_results = []

    for run_idx in range(NUM_RUNS):
        t0 = time.perf_counter()
        out = llm.generate([prompt], params)
        elapsed = time.perf_counter() - t0
        pt = len(out[0].prompt_token_ids)
        gt = len(out[0].outputs[0].token_ids)

        time.sleep(0.3)
        stats = read_profile_decode_stats(prev_len)
        if stats:
            prev_len = stats["total_rows"]

        decode_info = stats.get("decode", {}) if stats else {}
        prefill_info = stats.get("prefill", {}) if stats else {}

        d_step = decode_info.get("avg_step_ms", 0)
        d_attn = decode_info.get("avg_attn_ms", 0)
        d_moe = decode_info.get("avg_moe_ms", 0)
        d_other = decode_info.get("avg_other_ms", 0)
        d_n = decode_info.get("n_steps", 0)
        p_step = prefill_info.get("avg_step_ms", 0)

        run_results.append({
            "elapsed": elapsed, "pt": pt, "gt": gt,
            "d_step": d_step, "d_attn": d_attn, "d_moe": d_moe,
            "d_other": d_other, "d_n": d_n, "p_step": p_step,
        })

        tok_s = gt / elapsed if elapsed > 0 else 0
        print(f"  Run {run_idx+1}: {elapsed:.2f}s, prompt={pt}, gen={gt}, "
              f"{tok_s:.1f} tok/s total")
        if d_n > 0:
            print(f"    decode: {d_step:.2f} ms/step = "
                  f"{d_attn:.2f} attn + {d_moe:.2f} moe + {d_other:.2f} other "
                  f"({d_n} steps)")
        if p_step > 0:
            print(f"    prefill: {p_step:.2f} ms/step")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary — {mode_label}")
    print(f"{'='*70}")
    avg_elapsed = sum(r["elapsed"] for r in run_results) / NUM_RUNS
    avg_d_step = sum(r["d_step"] for r in run_results) / NUM_RUNS
    avg_d_attn = sum(r["d_attn"] for r in run_results) / NUM_RUNS
    avg_d_moe = sum(r["d_moe"] for r in run_results) / NUM_RUNS
    avg_d_other = sum(r["d_other"] for r in run_results) / NUM_RUNS
    avg_gt = sum(r["gt"] for r in run_results) / NUM_RUNS
    avg_tok_s = avg_gt / avg_elapsed if avg_elapsed > 0 else 0

    print(f"  Avg total: {avg_elapsed:.2f}s, {avg_tok_s:.1f} tok/s")
    print(f"  Avg decode step: {avg_d_step:.2f} ms")
    print(f"    Attention: {avg_d_attn:.2f} ms ({avg_d_attn/avg_d_step*100:.1f}%)")
    print(f"    MoE:       {avg_d_moe:.2f} ms ({avg_d_moe/avg_d_step*100:.1f}%)")
    print(f"    Other:     {avg_d_other:.2f} ms ({avg_d_other/avg_d_step*100:.1f}%)")
    print(f"{'='*70}")

    # Save last run output snippet for sanity check
    print(f"\nOutput preview: {out[0].outputs[0].text[:150]}")
    del llm
