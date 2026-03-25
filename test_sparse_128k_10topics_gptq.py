"""Test sparse prefill+decode at ~100K+ tokens with 10 unrelated topics.
Uses Dynamic NTK RoPE scaling (factor=4) to extend MiniCPM4-8B from 32K to 128K.
Fetches real Wikipedia content for each topic.
"""
import os, urllib.request, json
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B-GPTQ-Int4"
PROXY = "http://child-prc.intel.com:913"

def fetch_wiki(title, target_chars=42000):
    """Fetch Wikipedia article plaintext."""
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
    """Pad text to target length by repeating with markers."""
    if len(text) >= target_chars:
        return text[:target_chars]
    result = text
    rep = 2
    while len(result) < target_chars:
        result += f"\n\n[Continued details]\n\n" + text
        rep += 1
    return result[:target_chars]

# 10 completely unrelated Wikipedia topics
TOPICS = [
    ("Plate_tectonics",           "Plate Tectonics and Continental Drift"),
    ("History_of_chess",          "History of Chess"),
    ("Honey_bee",                 "Honey Bee Colony Biology"),
    ("Roman_concrete",            "Ancient Roman Concrete and Engineering"),
    ("History_of_photography",    "History of Photography"),
    ("DNA",                       "DNA Structure and Molecular Biology"),
    ("History_of_chocolate",      "History of Chocolate"),
    ("Saturn_(planet)",           "Saturn: The Ringed Planet"),
    ("History_of_writing",        "History of Writing Systems"),
    ("Immune_system",             "The Human Immune System"),
]

# Keywords for quality check per topic
KEYWORDS = {
    "Plate Tectonics":  ["tectonic", "wegener", "pangaea", "subduction", "earthquake", "himalaya", "rift", "seafloor"],
    "Chess":            ["chess", "kasparov", "fischer", "steinitz", "deep blue", "botvinnik", "carlsen", "grandmaster"],
    "Honey Bees":       ["bee", "waggle", "honey", "queen bee", "colony", "pollen", "hive", "forager"],
    "Roman Eng":        ["roman", "pantheon", "concrete", "aqueduct", "colosseum", "pont du gard", "opus"],
    "Photography":      ["photograph", "daguerr", "kodak", "camera", "niépce", "niepce", "talbot", "eastman"],
    "DNA":              ["dna", "double helix", "nucleotide", "watson", "crick", "gene", "chromosome", "genome"],
    "Chocolate":        ["chocolate", "cacao", "cocoa", "aztec", "maya", "theobrom", "confection"],
    "Saturn":           ["saturn", "ring", "titan", "cassini", "gas giant", "enceladus", "galileo"],
    "Writing":          ["writing", "cuneiform", "hieroglyph", "alphabet", "sumerian", "phoenician", "script"],
    "Immune":           ["immune", "antibod", "lymphocyte", "t cell", "b cell", "antigen", "pathogen", "vaccine"],
}

QUESTION = """

Based on ALL 10 sections above, answer these questions. Give specific facts with names, numbers, or dates for EACH section:
1. Plate Tectonics: Name one specific earthquake or geological event with a date.
2. Chess: Name one specific chess player and their achievement with a year.
3. Honey Bees: Give one specific numerical fact about bee biology.
4. Roman Engineering: Name one specific structure and a dimension.
5. Photography: Name one inventor and their invention with a date.
6. DNA: Name one scientist and their discovery.
7. Chocolate: Name one historical civilization and how they used chocolate.
8. Saturn: Name one moon or mission with a specific fact.
9. Writing: Name one ancient writing system and where it originated.
10. Immune System: Name one type of immune cell and its function.
Keep each answer to 1-2 sentences. Be specific."""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sparse", choices=["dense", "sparse", "dense_prefill"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--factor", type=float, default=4.0, help="RoPE scaling factor (default 4 for 128K)")
    args = parser.parse_args()

    if args.mode == "dense":
        backend = "ESIMD_ATTN"
    elif args.mode == "sparse":
        backend = "INFLLMV2_ESIMD_ATTN"
    elif args.mode == "dense_prefill":
        backend = "INFLLMV2_ESIMD_ATTN"
        os.environ["INFLLMV2_SPARSE_PREFILL"] = "0"

    max_ctx = int(32768 * args.factor)
    hf_overrides = {
        "max_position_embeddings": max_ctx,
        "rope_scaling": {"rope_type": "dynamic", "factor": args.factor},
    }

    # Target ~10.5K tokens per topic = ~42K chars, total ~105K tokens
    target_per_topic = 42000
    print(f"Fetching 10 topics (~{target_per_topic} chars each)...")

    sections = []
    for i, (wiki, name) in enumerate(TOPICS):
        text = fetch_wiki(wiki, target_per_topic)
        if not text or len(text) < 3000:
            print(f"  [{i+1}] FAILED fetch for {name}, using padded fallback")
            text = f"This section covers {name}. " * 200  # minimal fallback
        else:
            if len(text) < target_per_topic * 0.6:
                text = pad_text(text, target_per_topic)
                print(f"  [{i+1}] Padded to {len(text)} chars: {name}")
            else:
                print(f"  [{i+1}] Fetched {len(text)} chars: {name}")
        sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")

    prompt = "Read the following 10 sections on completely unrelated topics. You will answer questions about ALL of them.\n" + "".join(sections) + QUESTION

    total_chars = len(prompt)
    est_tokens = total_chars // 4
    print(f"\nMode: {args.mode}, RoPE: dynamic (factor={args.factor}), max_ctx={max_ctx}")
    print(f"Prompt: {total_chars:,} chars (~{est_tokens:,} tokens est.)")

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
              max_model_len=max_ctx, max_num_seqs=1, block_size=128,
              attention_backend=backend, hf_overrides=hf_overrides)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    outputs = llm.generate([prompt], params)
    out = outputs[0]
    pt = len(out.prompt_token_ids)
    gt = len(out.outputs[0].token_ids)
    text = out.outputs[0].text

    print(f"Prompt tokens: {pt:,}")
    print(f"Generated tokens: {gt}")
    print(f"\n{'='*70}")
    print(f"  OUTPUT ({args.mode}, factor={args.factor}, {pt:,} tokens)")
    print(f"{'='*70}")
    print(text)
    print(f"{'='*70}")

    tl = text.lower()
    short_names = list(KEYWORDS.keys())
    results = {}
    for name, kws in KEYWORDS.items():
        results[name] = any(w in tl for w in kws)

    print(f"\nQuality — topics discovered:")
    for k, v in results.items():
        print(f"  {k}: {'YES' if v else 'NO'}")
    score = sum(results.values())
    print(f"  Score: {score}/10 topics covered")
    del llm
