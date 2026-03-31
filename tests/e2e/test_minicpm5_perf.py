"""MiniCPM5 GPTQ-Int4 performance benchmark with full kernel control.

Controls: ESIMD MoE, ESIMD Attn (dense/sparse/off), dtype, prefill len, decode len.

Examples:
  # fp16, ESIMD MoE + dense ESIMD attn, 84K prefill, 512 decode
  python tests/e2e/test_minicpm5_perf.py --prefill 84000 --decode 512

  # bf16, Triton MoE + Triton attn, 40K prefill, 256 decode
  python tests/e2e/test_minicpm5_perf.py --prefill 40000 --decode 256 --dtype bf16 --no-esimd-moe --attn triton

  # fp16, ESIMD MoE + sparse ESIMD attn, 84K prefill, 1024 decode
  python tests/e2e/test_minicpm5_perf.py --prefill 84000 --decode 1024 --attn sparse

  # Short test, 3 runs with warmup
  python tests/e2e/test_minicpm5_perf.py --prefill 4000 --decode 128 --runs 3
"""
import os, time, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("MINICPM5_ESIMD_MOE", "1")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "wiki_corpus.json")

QUESTION = (
    "Based on all the articles above, provide a comprehensive summary "
    "covering the key themes across all topics. For each topic, mention "
    "the most important facts and how they connect to each other. "
    "Be detailed and thorough in your response."
)


def build_corpus():
    """Load pre-fetched Wikipedia corpus from wiki_corpus.json."""
    print(f"Loading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    all_text = []
    for topic, text in corpus.items():
        if text:
            all_text.append(f"\n\n=== {topic} ===\n\n{text}")
            print(f"  {topic}: {len(text)} chars")
        else:
            print(f"  {topic}: (empty, skipped)")
    return "\n".join(all_text)


def build_prompt(tokenizer, corpus, target_tokens):
    text = corpus
    tokens = tokenizer.encode(text)
    while len(tokens) < target_tokens:
        text = text + "\n" + text
        tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)

    full_text = text + "\n\nQuestion: " + QUESTION + "\nAnswer:"
    msgs = [{"role": "user", "content": full_text}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    ntoks = len(tokenizer.encode(prompt))
    return prompt, ntoks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MiniCPM5 GPTQ-Int4 perf benchmark")
    parser.add_argument("--prefill", type=int, default=84000,
                        help="Target prefill tokens (default: 84000)")
    parser.add_argument("--decode", type=int, default=512,
                        help="Max decode tokens (default: 512)")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"],
                        help="Model dtype (default: fp16)")
    parser.add_argument("--attn", type=str, default="dense",
                        choices=["dense", "sparse", "triton"],
                        help="Attention backend: dense=ESIMD_ATTN, "
                             "sparse=INFLLMV2_ESIMD_ATTN, triton=default "
                             "(default: dense)")
    parser.add_argument("--no-esimd-moe", action="store_true",
                        help="Disable ESIMD MoE (use Triton MoE)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of timed runs (default: 1)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup run")
    parser.add_argument("--rep-pen", type=float, default=1.0,
                        help="Repetition penalty (1.0=off)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    args = parser.parse_args()

    # Map dtype arg
    dtype_map = {"fp16": "float16", "bf16": "bfloat16"}
    model_dtype = dtype_map[args.dtype]

    # Map attention backend
    attn_map = {
        "dense": "ESIMD_ATTN",
        "sparse": "INFLLMV2_ESIMD_ATTN",
        "triton": None,
    }
    attn_backend = attn_map[args.attn]

    # ESIMD MoE control
    if args.no_esimd_moe:
        os.environ["MINICPM5_ESIMD_MOE"] = "0"

    moe_label = "ESIMD" if not args.no_esimd_moe else "Triton"
    attn_label = args.attn.upper()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    corpus = build_corpus()
    prompt, ntoks = build_prompt(tokenizer, corpus, args.prefill)

    print(f"\n{'='*70}")
    print(f"  Config: dtype={args.dtype}, MoE={moe_label}, Attn={attn_label}")
    print(f"  Prefill={ntoks} tokens, Decode={args.decode} tokens")
    print(f"  Runs={args.runs}, Warmup={'OFF' if args.no_warmup else 'ON'}")
    if args.temperature > 0:
        print(f"  Sampling: temp={args.temperature}, rep_pen={args.rep_pen}")
    else:
        print(f"  Sampling: GREEDY (temp=0), rep_pen={args.rep_pen}")
    print(f"{'='*70}\n")

    print("Loading model...")
    t_load0 = time.perf_counter()

    llm_kwargs = dict(
        model=MODEL_PATH, dtype=model_dtype, trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True,
        gpu_memory_utilization=0.85, max_model_len=131072,
        max_num_seqs=1, block_size=128, enable_prefix_caching=False,
    )
    if attn_backend:
        llm_kwargs["attention_backend"] = attn_backend

    llm = LLM(**llm_kwargs)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_load1 = time.perf_counter()
    print(f"Model loaded in {t_load1-t_load0:.1f}s\n")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sp_kwargs = dict(
        temperature=args.temperature, max_tokens=args.decode,
        stop_token_ids=[im_end_id],
    )
    if args.rep_pen != 1.0:
        sp_kwargs["repetition_penalty"] = args.rep_pen
    sp = SamplingParams(**sp_kwargs)

    # Warmup
    if not args.no_warmup:
        print("Warmup...", flush=True)
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        t1 = time.perf_counter()
        gen_toks = len(out[0].outputs[0].token_ids)
        print(f"  Warmup done: {(t1-t0)*1e3:.0f}ms, gen={gen_toks}\n")

    # Timed runs
    results = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sp)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1e3
        gen_toks = len(out[0].outputs[0].token_ids)
        prompt_toks = len(out[0].prompt_token_ids)
        text = out[0].outputs[0].text

        # Quality: unique 4-gram ratio
        words = text.split()[:500]
        if len(words) >= 4:
            ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
            unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        else:
            unique_ratio = 1.0

        results.append({
            "wall_ms": wall_ms,
            "prompt_toks": prompt_toks,
            "gen_toks": gen_toks,
            "unique_4gram": unique_ratio,
            "text": text,
        })

        tok_per_s = (prompt_toks + gen_toks) / (wall_ms / 1e3)
        print(f"  Run {r}: {wall_ms:.0f}ms, prefill={prompt_toks}, "
              f"gen={gen_toks}, {tok_per_s:.0f} tok/s, "
              f"4gram={unique_ratio:.3f}")

    # Summary
    avg_ms = sum(r["wall_ms"] for r in results) / len(results)
    best_ms = min(r["wall_ms"] for r in results)

    print(f"\n{'='*70}")
    print(f"  RESULTS: dtype={args.dtype} MoE={moe_label} Attn={attn_label}")
    print(f"{'='*70}")
    print(f"  Prefill tokens:  {results[0]['prompt_toks']:,}")
    print(f"  Decode tokens:   {results[-1]['gen_toks']}")
    print(f"  Avg wall-time:   {avg_ms:.0f} ms")
    print(f"  Best wall-time:  {best_ms:.0f} ms")
    avg_total_tok = results[0]['prompt_toks'] + results[-1]['gen_toks']
    print(f"  Avg throughput:  {avg_total_tok/(avg_ms/1e3):.0f} tok/s")

    # Show output preview
    last_text = results[-1]["text"]
    preview = last_text[:1000].replace('\n', ' ')
    print(f"\n  Output preview: {preview}")
    if len(last_text) > 1000:
        print(f"  ... ({len(last_text)} chars total)")
    print(f"{'='*70}")

    del llm
