"""Measure pure prefill wall time: ESIMD+oneDNN vs Triton.
Uses a simple repeated prompt (no network needed).
1 warmup + 3 timed runs, output 1 token per run.
"""
import os, sys, time, gc

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
NUM_RUNS = 3
MAX_TOKENS = 1


def build_prompt(target_tokens=25000):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Build a long prompt by repeating text
    base = ("The quick brown fox jumps over the lazy dog. "
            "Machine learning models process tokens sequentially. "
            "Neural networks have revolutionized natural language processing. "
            "Attention mechanisms allow models to focus on relevant parts of input. ")
    raw = base * 2000  # way more than needed
    msgs = [{"role": "user", "content": raw}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n</think>\n\n"

    # Trim to target
    tokens = tokenizer.encode(prompt)
    if len(tokens) > target_tokens:
        prompt = tokenizer.decode(tokens[:target_tokens], skip_special_tokens=False)

    ntoks = len(tokenizer.encode(prompt))
    print(f"Prompt: {ntoks} tokens")
    return prompt, ntoks


def run_one(prompt, ntoks, use_esimd_moe, attn_backend):
    import torch
    from vllm import LLM, SamplingParams

    label = "ESIMD+oneDNN" if use_esimd_moe else "Triton"
    os.environ["MINICPM5_ESIMD_MOE"] = "1" if use_esimd_moe else "0"

    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
        max_model_len=32768, max_num_seqs=1, block_size=128,
        attention_backend=attn_backend,
        enable_prefix_caching=False,  # CRITICAL: disable to measure real prefill
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    # Warmup (2 runs to warm JIT / oneDNN caches)
    print(f"\n[{label}] Warmup...", flush=True)
    for w in range(2):
        out = llm.generate([prompt], sp)
        torch.xpu.synchronize()
    pt = len(out[0].prompt_token_ids)
    print(f"[{label}] Actual prompt tokens: {pt}")

    # Timed runs
    times = []
    for r in range(NUM_RUNS):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1e3
        times.append(wall_ms)
        print(f"[{label}] Run {r}: {wall_ms:.1f} ms  ({pt/(wall_ms/1e3):.0f} tok/s)")

    avg_ms = sum(times) / len(times)
    best_ms = min(times)
    print(f"[{label}] Avg: {avg_ms:.1f} ms  Best: {best_ms:.1f} ms")

    del llm
    torch.xpu.empty_cache()
    gc.collect()
    return avg_ms, best_ms, pt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=25000)
    parser.add_argument("--esimd-only", action="store_true")
    parser.add_argument("--triton-only", action="store_true")
    parser.add_argument("--attn", default="ESIMD_ATTN",
                        help="Attention backend (ESIMD_ATTN, INFLLMV2_ESIMD_ATTN, etc.)")
    args = parser.parse_args()

    prompt, ntoks = build_prompt(args.tokens)
    attn = args.attn

    results = {}
    if not args.esimd_only:
        results["triton"] = run_one(prompt, ntoks, False, attn)
    if not args.triton_only:
        results["esimd"] = run_one(prompt, ntoks, True, attn)

    print(f"\n{'='*50}")
    print(f"Summary — {ntoks} target tokens, {attn}")
    print(f"{'='*50}")
    for k, (avg, best, pt) in results.items():
        print(f"  {k:15s}: avg={avg:.1f}ms  best={best:.1f}ms  ({pt/(avg/1e3):.0f} tok/s avg)")
    if "triton" in results and "esimd" in results:
        sa, sb = results["triton"][0] / results["esimd"][0], results["triton"][1] / results["esimd"][1]
        print(f"  Speedup: avg={sa:.2f}x  best={sb:.2f}x")
