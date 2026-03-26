"""Profile MiniCPM5 component timing: attention vs MoE, dense vs sparse.
Uses MINICPM5_PROFILE=1 env var to trigger per-layer timing in child process.
Writes profile data to /tmp/minicpm5_profile.jsonl, read after each generate.
"""
import os, sys, time, json, urllib.request, argparse, collections
os.environ["MINICPM5_PROFILE"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
PROXY = "http://child-prc.intel.com:913"
PROFILE_PATH = "/tmp/minicpm5_profile.jsonl"
os.environ["MINICPM5_PROFILE_PATH"] = PROFILE_PATH


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


def read_profile(label, prev_len=0):
    """Read profile data from jsonl file and print summary for new entries only."""
    if not os.path.exists(PROFILE_PATH):
        print(f"  No profile data found")
        return 0

    with open(PROFILE_PATH) as f:
        all_rows = [json.loads(line) for line in f]

    # Only analyze new rows since last read
    rows = all_rows[prev_len:]
    if not rows:
        print(f"  No new profile data")
        return len(all_rows)

    # Group by step: (step, layer_idx, attn_ms, mlp_ms, total_ms, n_tokens)
    by_step = collections.defaultdict(list)
    for step, layer_idx, attn_ms, mlp_ms, total_ms, n_tokens in rows:
        by_step[step].append((layer_idx, attn_ms, mlp_ms, total_ms, n_tokens))

    steps = sorted(by_step.keys())
    prefill_steps = [s for s in steps if by_step[s][0][4] > 1]
    decode_steps = [s for s in steps if by_step[s][0][4] == 1]

    print(f"\n{'='*80}")
    print(f"  {label}: {len(prefill_steps)} prefill, {len(decode_steps)} decode steps")
    print(f"{'='*80}")

    for phase, step_list in [("PREFILL", prefill_steps), ("DECODE", decode_steps)]:
        if not step_list:
            continue

        attn_total = mlp_total = all_total = 0.0
        n_tokens_first = by_step[step_list[0]][0][4]
        attn_per_layer = collections.defaultdict(float)
        mlp_per_layer = collections.defaultdict(float)

        for s in step_list:
            for layer_idx, attn_ms, mlp_ms, total_ms, n_tokens in by_step[s]:
                attn_total += attn_ms
                mlp_total += mlp_ms
                all_total += total_ms
                attn_per_layer[layer_idx] += attn_ms
                mlp_per_layer[layer_idx] += mlp_ms

        n = len(step_list)
        other_total = all_total - attn_total - mlp_total

        print(f"\n  --- {phase} ({n} steps, {n_tokens_first} tok/step) ---")
        print(f"  Total: {all_total:.1f} ms")
        pct_a = attn_total / all_total * 100 if all_total else 0
        pct_m = mlp_total / all_total * 100 if all_total else 0
        pct_o = other_total / all_total * 100 if all_total else 0
        print(f"    Attention: {attn_total:.1f} ms ({pct_a:.1f}%)")
        print(f"    MLP/MoE:   {mlp_total:.1f} ms ({pct_m:.1f}%)")
        print(f"    Other:     {other_total:.1f} ms ({pct_o:.1f}%)")

        per_step = all_total / n
        attn_per = attn_total / n
        mlp_per = mlp_total / n
        other_per = other_total / n
        print(f"  Per step: {per_step:.2f} ms = {attn_per:.2f} attn + "
              f"{mlp_per:.2f} mlp + {other_per:.2f} other")

        for li in [0, 1, 14, 27]:
            ltype = "dense" if li == 0 else "MoE"
            a = attn_per_layer.get(li, 0) / n
            m = mlp_per_layer.get(li, 0) / n
            print(f"    L{li:2d}({ltype:5s}): attn={a:.2f}  mlp={m:.2f} ms")

    print(f"{'='*80}")
    return len(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dense", choices=["dense", "sparse"])
    parser.add_argument("--context", default="both", choices=["short", "long", "both"])
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.mode == "dense":
        backend = "ESIMD_ATTN"
    else:
        backend = "INFLLMV2_ESIMD_ATTN"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Build short prompt
    msgs_short = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    prompt_short = tokenizer.apply_chat_template(msgs_short, tokenize=False, add_generation_prompt=True)
    prompt_short += "<think>\n</think>\n\n"

    # Build long prompt
    prompt_long = None
    if args.context in ("long", "both"):
        topics = [
            ("Plate_tectonics", "Plate Tectonics"),
            ("History_of_chess", "Chess History"),
            ("Honey_bee", "Honey Bees"),
            ("Roman_concrete", "Roman Engineering"),
            ("History_of_photography", "Photography History"),
        ]
        print("Fetching Wikipedia content...")
        sections = []
        for i, (wiki, name) in enumerate(topics):
            text = fetch_wiki(wiki, 24000)
            if not text or len(text) < 3000:
                text = f"This section covers {name}. " * 200
            elif len(text) < 14000:
                text = pad_text(text, 24000)
            sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")
            print(f"  [{i+1}] {len(text)} chars: {name}")
        raw = ("Read 5 sections and answer.\n" + "".join(sections)
               + "\n\nGive one fact from each section.")
        msgs_long = [{"role": "user", "content": raw}]
        prompt_long = tokenizer.apply_chat_template(msgs_long, tokenize=False, add_generation_prompt=True)
        prompt_long += "<think>\n</think>\n\n"

    # Clean profile file
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)

    max_ctx = 32768 if prompt_long else 2048
    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
        max_model_len=max_ctx, max_num_seqs=1, block_size=128,
        attention_backend=backend,
    )

    # Warmup
    params_warmup = SamplingParams(temperature=0, max_tokens=4)
    _ = llm.generate([prompt_short], params_warmup)
    time.sleep(0.5)
    # Note where warmup data ends
    warmup_len = 0
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH) as f:
            warmup_len = sum(1 for _ in f)
    print(f"\nWarmup done ({warmup_len} profile rows)\n")

    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    prev_len = warmup_len

    if args.context in ("short", "both"):
        t0 = time.time()
        out = llm.generate([prompt_short], params)
        elapsed = time.time() - t0
        pt = len(out[0].prompt_token_ids)
        gt = len(out[0].outputs[0].token_ids)
        print(f"SHORT: prompt={pt}, gen={gt}, time={elapsed:.2f}s")
        time.sleep(0.3)
        prev_len = read_profile(f"SHORT CONTEXT — {args.mode} (prompt={pt})", prev_len)

    if args.context in ("long", "both") and prompt_long:
        t0 = time.time()
        out = llm.generate([prompt_long], params)
        elapsed = time.time() - t0
        pt = len(out[0].prompt_token_ids)
        gt = len(out[0].outputs[0].token_ids)
        print(f"\nLONG: prompt={pt}, gen={gt}, time={elapsed:.2f}s")
        time.sleep(0.3)
        prev_len = read_profile(f"LONG CONTEXT — {args.mode} (prompt={pt})", prev_len)

    del llm
