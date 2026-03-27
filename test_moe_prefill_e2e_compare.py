"""Compare MoE prefill speed: Triton baseline vs ESIMD+oneDNN.
25K sparse prefill (INFLLMV2_ESIMD_ATTN) to minimize attention %.
Warmup + 3 timed runs, output 1 token per run.

Usage:
  python test_moe_prefill_e2e_compare.py
"""
import os, sys, time, gc, urllib.request, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
PROXY = "http://child-prc.intel.com:913"
NUM_RUNS = 3
MAX_TOKENS = 1  # 1 output token → measure pure prefill


def fetch_wiki(title, target_chars=24000):
    proxy_handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    url = (f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}"
           f"&prop=extracts&explaintext=1&format=json&exlimit=1")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = opener.open(req, timeout=30)
        data = json.loads(resp.read().decode())
        text = next(iter(data["query"]["pages"].values())).get("extract", "")
        if len(text) > target_chars:
            cut = text[:target_chars].rfind(". ")
            text = text[:cut+1] if cut > target_chars * 0.8 else text[:target_chars]
        return text
    except Exception as e:
        print(f"  WARNING: fetch failed for '{title}': {e}")
        return None


def pad_text(text, target_chars):
    if len(text) >= target_chars:
        return text[:target_chars]
    result = text
    while len(result) < target_chars:
        result += f"\n\n[Continued]\n\n" + text
    return result[:target_chars]


def build_long_prompt():
    """Build ~25K token prompt from Wikipedia articles."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    topics = [
        ("Plate_tectonics", "Plate Tectonics"),
        ("History_of_chess", "Chess History"),
        ("Honey_bee", "Honey Bees"),
        ("Roman_concrete", "Roman Engineering"),
        ("History_of_photography", "Photography History"),
    ]
    sections = []
    for i, (wiki, name) in enumerate(topics):
        text = fetch_wiki(wiki, 24000)
        if not text or len(text) < 3000:
            text = f"This section covers {name}. " * 200
        elif len(text) < 14000:
            text = pad_text(text, 24000)
        sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")
        print(f"  [{i+1}] {len(text)} chars: {name}")

    raw = ("Read 5 sections and answer questions about them.\n"
           + "".join(sections)
           + "\n\nBased on ALL sections, give one fact from each section.")
    msgs = [{"role": "user", "content": raw}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n</think>\n\n"

    ntoks = len(tokenizer.encode(prompt))
    print(f"  Prompt: {ntoks} tokens")
    return prompt, ntoks


def run_benchmark(prompt, ntoks, use_esimd_moe, attn_backend):
    """Run warmup + NUM_RUNS timed prefill-only inferences."""
    import torch
    from vllm import LLM, SamplingParams

    label = "ESIMD+oneDNN" if use_esimd_moe else "Triton"
    print(f"\n{'='*60}")
    print(f"  {label} MoE  |  {attn_backend}  |  {ntoks} tokens")
    print(f"{'='*60}")

    if use_esimd_moe:
        os.environ["MINICPM5_ESIMD_MOE"] = "1"
    else:
        os.environ["MINICPM5_ESIMD_MOE"] = "0"

    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
        max_model_len=32768, max_num_seqs=1, block_size=128,
        attention_backend=attn_backend,
    )

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    # Warmup
    print("  Warmup...", flush=True)
    out = llm.generate([prompt], sp)
    pt = len(out[0].prompt_token_ids)
    print(f"  Warmup done, actual prompt tokens: {pt}")

    # Timed runs
    times = []
    for r in range(NUM_RUNS):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1e3
        tok_per_s = pt / (wall_ms / 1e3)
        times.append(wall_ms)
        print(f"  Run {r}: {wall_ms:.0f} ms  ({tok_per_s:.0f} prefill tok/s)")

    avg_ms = sum(times) / len(times)
    best_ms = min(times)
    print(f"  Avg: {avg_ms:.0f} ms  ({pt/(avg_ms/1e3):.0f} tok/s)")
    print(f"  Best: {best_ms:.0f} ms  ({pt/(best_ms/1e3):.0f} tok/s)")

    del llm
    torch.xpu.empty_cache()
    gc.collect()
    return avg_ms, best_ms, pt


if __name__ == "__main__":
    print("Building 25K prompt...")
    prompt, ntoks = build_long_prompt()

    # Use sparse attention to minimize attention overhead (focus on MoE)
    attn_backend = "INFLLMV2_ESIMD_ATTN"

    # 1. Triton baseline
    triton_avg, triton_best, pt = run_benchmark(prompt, ntoks, use_esimd_moe=False,
                                                 attn_backend=attn_backend)

    # 2. ESIMD+oneDNN
    esimd_avg, esimd_best, _ = run_benchmark(prompt, ntoks, use_esimd_moe=True,
                                              attn_backend=attn_backend)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary — {pt} token prefill, {attn_backend}")
    print(f"{'='*60}")
    print(f"  Triton:       avg={triton_avg:.0f}ms  best={triton_best:.0f}ms  "
          f"({pt/(triton_avg/1e3):.0f} tok/s avg)")
    print(f"  ESIMD+oneDNN: avg={esimd_avg:.0f}ms  best={esimd_best:.0f}ms  "
          f"({pt/(esimd_avg/1e3):.0f} tok/s avg)")
    speedup_avg = triton_avg / esimd_avg
    speedup_best = triton_best / esimd_best
    print(f"  Speedup:      avg={speedup_avg:.2f}x  best={speedup_best:.2f}x")
