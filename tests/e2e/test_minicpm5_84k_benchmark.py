"""MiniCPM5-16B GPTQ 84K context benchmark: sparse vs dense attention.

Fetches 10 Wikipedia topics (~8.4K tokens each = ~84K total).
1 warmup + 3 timed runs. Reports prefill + decode wall-time, quality score.

Usage:
  python tests/e2e/test_minicpm5_84k_benchmark.py --mode sparse --max-tokens 512
  python tests/e2e/test_minicpm5_84k_benchmark.py --mode dense --max-tokens 512
  python tests/e2e/test_minicpm5_84k_benchmark.py --mode sparse --runs 1 --no-warmup  # for unitrace
"""
import os, sys, time, urllib.request, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
PROXY = "http://child-prc.intel.com:913"

TOPICS = [
    ("Plate_tectonics",        "Plate Tectonics and Continental Drift"),
    ("History_of_chess",       "History of Chess"),
    ("Honey_bee",              "Honey Bee Colony Biology"),
    ("Roman_concrete",         "Ancient Roman Concrete and Engineering"),
    ("History_of_photography", "History of Photography"),
    ("DNA",                    "DNA Structure and Molecular Biology"),
    ("History_of_chocolate",   "History of Chocolate"),
    ("Saturn_(planet)",        "Saturn: The Ringed Planet"),
    ("History_of_writing",     "History of Writing Systems"),
    ("Immune_system",          "The Human Immune System"),
]

KEYWORDS = {
    "Plate Tectonics": ["tectonic", "wegener", "pangaea", "subduction", "earthquake", "himalaya"],
    "Chess":           ["chess", "kasparov", "fischer", "deep blue", "grandmaster", "carlsen"],
    "Honey Bees":      ["bee", "waggle", "honey", "queen", "colony", "pollen", "hive"],
    "Roman Eng":       ["roman", "pantheon", "concrete", "aqueduct", "colosseum"],
    "Photography":     ["photograph", "daguerr", "kodak", "camera", "niépce", "niepce", "talbot"],
    "DNA":             ["dna", "double helix", "nucleotide", "watson", "crick", "gene", "genome"],
    "Chocolate":       ["chocolate", "cacao", "cocoa", "aztec", "maya", "theobrom"],
    "Saturn":          ["saturn", "ring", "titan", "cassini", "gas giant", "enceladus"],
    "Writing":         ["writing", "cuneiform", "hieroglyph", "alphabet", "sumerian", "phoenician"],
    "Immune":          ["immune", "antibod", "lymphocyte", "t cell", "b cell", "antigen", "pathogen"],
}

QUESTION = """

Based on ALL 10 sections above, answer these questions with specific facts:
1. Plate Tectonics: Name one geological event with a date.
2. Chess: Name one chess player and their achievement with a year.
3. Honey Bees: Give one numerical fact about bee biology.
4. Roman Engineering: Name one structure and a dimension.
5. Photography: Name one inventor and their invention with a date.
6. DNA: Name one scientist and their discovery.
7. Chocolate: Name one civilization and how they used chocolate.
8. Saturn: Name one moon or mission with a specific fact.
9. Writing: Name one ancient writing system and where it originated.
10. Immune System: Name one immune cell type and its function.
Keep each answer to 1-2 sentences. Be specific."""


def fetch_wiki(title, target_chars=33600):
    proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    url = (
        f"https://en.wikipedia.org/w/api.php?action=query"
        f"&titles={urllib.request.quote(title)}"
        f"&prop=extracts&explaintext=1&format=json&exlimit=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = opener.open(req, timeout=30)
        data = json.loads(resp.read().decode())
        text = next(iter(data["query"]["pages"].values())).get("extract", "")
        if len(text) > target_chars:
            cut = text[:target_chars].rfind(". ")
            text = text[: cut + 1] if cut > target_chars * 0.8 else text[:target_chars]
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


def build_prompt(tokenizer):
    target_per_topic = 44000  # ~11K tokens each × 10 = ~84K+ (chars/token ratio ~4)
    print(f"Fetching 10 topics (~{target_per_topic} chars each)...")
    sections = []
    for i, (wiki, name) in enumerate(TOPICS):
        text = fetch_wiki(wiki, target_per_topic)
        if not text or len(text) < 3000:
            print(f"  [{i+1}] FAILED fetch, using padded fallback: {name}")
            text = f"This section covers {name}. " * 500
        if len(text) < target_per_topic * 0.6:
            text = pad_text(text, target_per_topic)
            print(f"  [{i+1}] Padded to {len(text)} chars: {name}")
        else:
            print(f"  [{i+1}] Fetched {len(text)} chars: {name}")
        sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")

    raw = (
        "Read the following 10 sections on unrelated topics. "
        "Answer questions about ALL of them.\n"
        + "".join(sections)
        + QUESTION
    )
    msgs = [{"role": "user", "content": raw}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    ntoks = len(tokenizer.encode(prompt))
    print(f"  Prompt: {ntoks:,} tokens ({len(prompt):,} chars)")
    return prompt, ntoks


def quality_check(text):
    tl = text.lower()
    results = {}
    for name, kws in KEYWORDS.items():
        results[name] = any(w in tl for w in kws)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="sparse", choices=["dense", "sparse"]
    )
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max output tokens (includes thinking)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (0.6 recommended for thinking model)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0=off, 1.2=recommended)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Top-k sampling (-1=off)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling (1.0=off)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--signal-ready", action="store_true",
                        help="Print READY signal after model load (for unitrace)")
    args = parser.parse_args()

    if args.mode == "dense":
        backend = "ESIMD_ATTN"
    else:
        backend = "INFLLMV2_ESIMD_ATTN"

    # Enable ESIMD MoE kernels
    os.environ["MINICPM5_ESIMD_MOE"] = "1"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt, ntoks = build_prompt(tokenizer)

    print(f"\nMode: {args.mode}, Backend: {backend}")
    print(f"Max tokens: {args.max_tokens}, Runs: {args.runs}")
    print(f"Loading model...")
    t_load0 = time.perf_counter()

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
        max_model_len=131072,
        max_num_seqs=1,
        block_size=128,
        enable_prefix_caching=False,
        attention_backend=backend,
        max_num_batched_tokens=2048,
    )

    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_load1 = time.perf_counter()
    print(f"Model loaded in {t_load1 - t_load0:.1f}s")

    if args.signal_ready:
        print("READY", flush=True)

    # MiniCPM5 is a thinking model: <think>...</think> then response.
    # Use <|im_end|> (id=130073) as stop token. Budget extra tokens for thinking.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # GPTQ-Int4 thinking model needs temperature>0 to avoid greedy thinking
    # loops. temp=0.6 produces good quality with reasonable think length.
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                        stop_token_ids=[im_end_id],
                        repetition_penalty=args.repetition_penalty,
                        top_k=args.top_k, top_p=args.top_p)
    print(f"Sampling: temp={args.temperature}, rep_pen={args.repetition_penalty}, "
          f"top_k={args.top_k}, top_p={args.top_p}")

    # Warmup
    if not args.no_warmup:
        print("\nWarmup run...", flush=True)
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        t1 = time.perf_counter()
        pt = len(out[0].prompt_token_ids)
        gt = len(out[0].outputs[0].token_ids)
        print(f"  Warmup: {(t1-t0)*1e3:.0f} ms, prompt={pt}, gen={gt}")

        # Show output quality from warmup
        text = out[0].outputs[0].text
        qc = quality_check(text)
        score = sum(qc.values())
        print(f"  Quality: {score}/10")

    # Timed runs
    times = []
    last_text = ""
    for r in range(args.runs):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        t1 = time.perf_counter()

        pt = len(out[0].prompt_token_ids)
        gt = len(out[0].outputs[0].token_ids)
        wall_ms = (t1 - t0) * 1e3

        # Approximate prefill vs decode split
        # prefill ~ wall_ms * pt / (pt + gt)  (rough, assumes uniform tok/s)
        # Better: use total time and report prefill tok/s and decode tok/s
        prefill_frac = pt / (pt + gt)
        est_prefill_ms = wall_ms * prefill_frac
        est_decode_ms = wall_ms * (1 - prefill_frac)

        times.append(wall_ms)
        last_text = out[0].outputs[0].text
        print(
            f"  Run {r}: {wall_ms:.0f} ms total "
            f"(~{est_prefill_ms:.0f} ms prefill @ {pt/(est_prefill_ms/1e3):.0f} tok/s, "
            f"~{est_decode_ms:.0f} ms decode @ {gt/(est_decode_ms/1e3):.1f} tok/s) "
            f"[prompt={pt}, gen={gt}]"
        )

    # Summary
    avg_ms = sum(times) / len(times)
    best_ms = min(times)
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\n{'='*70}")
    print(f"  RESULTS — {args.mode} ({backend})")
    print(f"{'='*70}")
    print(f"  Prompt tokens:   {pt:,}")
    print(f"  Decode tokens:   {gt}")
    print(f"  Avg wall-time:   {avg_ms:.0f} ms (std={std_ms:.0f})")
    print(f"  Best wall-time:  {best_ms:.0f} ms")
    print(f"  Total tok/s:     {(pt+gt)/(avg_ms/1e3):.0f} (avg)")

    # Quality check
    qc = quality_check(last_text)
    print(f"\n  Quality — topics discovered:")
    for k, v in qc.items():
        print(f"    {k}: {'YES' if v else 'NO'}")
    score = sum(qc.values())
    print(f"    Score: {score}/10")

    # Print output
    print(f"\n{'='*70}")
    print(f"  OUTPUT (last run)")
    print(f"{'='*70}")
    print(last_text[:2000])
    if len(last_text) > 2000:
        print(f"  ... ({len(last_text)} chars total)")
    print(f"{'='*70}")

    del llm
