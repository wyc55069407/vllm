"""Binary search for the prefill length threshold where output degrades."""
import os, sys, time, json, urllib.request, urllib.parse

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# MINICPM5_ESIMD_MOE is set conditionally in __main__ based on --no-esimd-moe
os.environ.setdefault("MINICPM5_ESIMD_MOE", "1")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
PROXY = "http://child-prc.intel.com:913"

WIKI_TOPICS = [
    "History of the Roman Empire",
    "Quantum mechanics",
    "World War II",
    "Theory of evolution",
    "History of China",
    "Machine learning",
    "Renaissance",
    "Climate change",
    "History of mathematics",
    "Human genome project",
]

QUESTION = (
    "Based on all the articles above, provide a comprehensive summary "
    "covering the key themes across all topics. For each topic, mention "
    "the most important facts and how they connect to each other. "
    "Be detailed and thorough in your response."
)


def fetch_wiki(title, max_chars=80000):
    proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    params = urllib.parse.urlencode({
        'action': 'query', 'titles': title, 'prop': 'extracts',
        'explaintext': 1, 'format': 'json', 'exlimit': 1,
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={'User-Agent': 'MiniCPM5-Test/1.0'})
    resp = opener.open(req, timeout=30)
    data = json.loads(resp.read().decode())
    page = next(iter(data['query']['pages'].values()))
    text = page.get('extract', '')
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def build_wiki_corpus():
    """Fetch all topics once, return combined text."""
    print("Fetching Wikipedia topics...")
    all_text = []
    for topic in WIKI_TOPICS:
        try:
            text = fetch_wiki(topic)
            if text:
                all_text.append(f"\n\n=== {topic} ===\n\n{text}")
                print(f"  {topic}: {len(text)} chars")
        except Exception as e:
            print(f"  {topic}: FAILED ({e})")
    return "\n".join(all_text)


def build_prompt_at_length(tokenizer, corpus, target_tokens):
    """Build a prompt with exactly target_tokens by truncating/repeating corpus."""
    # Repeat if needed
    text = corpus
    tokens = tokenizer.encode(text)
    while len(tokens) < target_tokens:
        text = text + "\n" + text
        tokens = tokenizer.encode(text)
    # Truncate to target
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)

    full_text = text + "\n\nQuestion: " + QUESTION + "\nAnswer:"
    msgs = [{"role": "user", "content": full_text}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ntoks = len(tokenizer.encode(prompt))
    return prompt, ntoks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=str, required=True,
                        help="Comma-separated token lengths to test, e.g. '45000,55000,65000,75000'")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens (short to save time)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Model dtype: bfloat16 or float16")
    parser.add_argument("--no-esimd-moe", action="store_true",
                        help="Disable ESIMD MoE (use Triton MoE)")
    parser.add_argument("--no-esimd-attn", action="store_true",
                        help="Disable ESIMD attention (use default attn)")
    parser.add_argument("--no-onednn", action="store_true",
                        help="Disable oneDNN W4A16 INT4 GEMM (use dequant+F.linear fallback)")
    parser.add_argument("--rep-pen", type=float, default=1.0,
                        help="Repetition penalty (1.0=off, 1.2=HF default)")
    args = parser.parse_args()

    # Override ESIMD flags based on args — must set to "0" not pop,
    # because spawn subprocess re-runs setdefault at module level
    if args.no_esimd_moe:
        os.environ["MINICPM5_ESIMD_MOE"] = "0"
    if args.no_onednn:
        os.environ["DISABLE_ONEDNN_W4A16"] = "1"
    if args.no_esimd_attn:
        attn_backend = None  # use default
    else:
        attn_backend = "ESIMD_ATTN"

    lengths = [int(x.strip()) for x in args.lengths.split(",")]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    corpus = build_wiki_corpus()

    llm_kwargs = dict(
        model=MODEL_PATH, dtype=args.dtype, trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True,
        gpu_memory_utilization=0.85, max_model_len=131072,
        max_num_seqs=1, block_size=128, enable_prefix_caching=False,
    )
    if attn_backend:
        llm_kwargs["attention_backend"] = attn_backend

    print(f"\nLoading vLLM model (dtype={args.dtype}, "
          f"esimd_moe={'ON' if not args.no_esimd_moe else 'OFF'}, "
          f"esimd_attn={'ON' if not args.no_esimd_attn else 'OFF'}, "
          f"onednn={'OFF' if args.no_onednn else 'ON'})...")
    llm = LLM(**llm_kwargs)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sp_kwargs = dict(
        temperature=0, max_tokens=args.max_tokens,
        stop_token_ids=[im_end_id],
    )
    if args.rep_pen != 1.0:
        sp_kwargs["repetition_penalty"] = args.rep_pen
    sp = SamplingParams(**sp_kwargs)
    print(f"Sampling: GREEDY (temp=0), rep_pen={args.rep_pen}, max_tokens={args.max_tokens}\n")

    results = []
    for target in lengths:
        prompt, ntoks = build_prompt_at_length(tokenizer, corpus, target)

        # print("------------------------ prompts are ------------------------")
        # print(prompt)
        # print("---------------------------------------------------------")
        print(f"--- Testing {ntoks} tokens ---")

        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        t1 = time.perf_counter()

        text = out[0].outputs[0].text
        gen_toks = len(out[0].outputs[0].token_ids)

        # Check for repetition: count unique 4-grams in first 200 tokens
        words = text.split()[:500]
        if len(words) >= 4:
            ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
            unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        else:
            unique_ratio = 1.0

        is_degenerate = unique_ratio < 0.5 or gen_toks < 10

        results.append({
            "tokens": ntoks,
            "gen_toks": gen_toks,
            "wall_s": t1-t0,
            "unique_4gram": unique_ratio,
            "degenerate": is_degenerate,
        })

        status = "DEGENERATE" if is_degenerate else "OK"
        print(f"  gen={gen_toks}, unique_4gram={unique_ratio:.3f}, wall={t1-t0:.1f}s → {status}")
        # Show first 300 chars of output
        preview = text[:99999].replace('\n', ' ')
        print(f"  preview: {preview}")
        print()

    print("\n" + "="*70)
    print(f"{'Tokens':>8} | {'Gen':>5} | {'4gram%':>6} | {'Status':>10} | {'Time':>7}")
    print("-"*70)
    for r in results:
        status = "DEGEN" if r["degenerate"] else "OK"
        print(f"{r['tokens']:>8} | {r['gen_toks']:>5} | {r['unique_4gram']:>6.3f} | {status:>10} | {r['wall_s']:>6.1f}s")
    print("="*70)
