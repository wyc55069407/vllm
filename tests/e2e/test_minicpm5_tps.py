"""MiniCPM5 Prefill & Decode TPS benchmark.

Measures prefill and decode throughput separately at various context lengths.

Usage:
  # Quick test: single config
  python tests/e2e/test_minicpm5_tps.py --attn sparse --prefill 32000 --decode 64

  # Sweep context lengths (sparse vs dense)
  python tests/e2e/test_minicpm5_tps.py --sweep

  # Custom sweep
  python tests/e2e/test_minicpm5_tps.py --attn sparse --prefill 8000 16000 32000 64000 --decode 64
"""
import os, time, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("MINICPM5_ESIMD_MOE", "1")
os.environ.setdefault("MINICPM5_ESIMD_GEMV", "1")

MODEL = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "wiki_corpus.json")


def build_prompt(tokenizer, corpus_text, target_tokens):
    """Build a prompt of approximately target_tokens length."""
    text = corpus_text
    tokens = tokenizer.encode(text)
    while len(tokens) < target_tokens:
        text = text + "\n" + text
        tokens = tokenizer.encode(text)
    tokens = tokens[:target_tokens]
    text = tokenizer.decode(tokens) + "\nQuestion: Summarize.\nAnswer:"
    msgs = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    ntoks = len(tokenizer.encode(prompt))
    return prompt, ntoks


def measure_tps(llm, prompt, ntoks, decode_tokens):
    """Measure prefill and decode TPS separately.

    Strategy: run twice with same prompt.
      Run 1: max_tokens=1 → measures prefill + 1 decode step
      Run 2: max_tokens=decode_tokens → measures prefill + N decode steps
      Decode time = Run2 - Run1, Decode TPS = (N-1) / decode_time
    """
    import torch

    sp_1 = SamplingParams(temperature=0, max_tokens=1)
    sp_n = SamplingParams(temperature=0, max_tokens=decode_tokens)

    # Run 1: prefill + 1 token
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t0 = time.perf_counter()
    out1 = llm.generate([prompt], sp_1)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_prefill = time.perf_counter() - t0

    # Run 2: prefill + N tokens
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t0 = time.perf_counter()
    out2 = llm.generate([prompt], sp_n)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_full = time.perf_counter() - t0

    gen = len(out2[0].outputs[0].token_ids)
    t_decode = t_full - t_prefill
    decode_toks = gen - 1  # subtract the 1 token from prefill run

    prefill_tps = ntoks / t_prefill
    decode_tps = decode_toks / t_decode if t_decode > 0 else 0
    decode_ms_per_tok = t_decode * 1000 / decode_toks if decode_toks > 0 else 0

    return {
        "prefill_tokens": ntoks,
        "decode_tokens": gen,
        "prefill_ms": t_prefill * 1000,
        "decode_ms": t_decode * 1000,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "decode_ms_per_tok": decode_ms_per_tok,
        "text": out2[0].outputs[0].text,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MiniCPM5 Prefill & Decode TPS benchmark")
    parser.add_argument("--attn", default="sparse",
                        choices=["dense", "sparse"])
    parser.add_argument("--prefill", type=int, nargs="+", default=[32000],
                        help="Prefill token counts (space-separated)")
    parser.add_argument("--decode", type=int, default=64,
                        help="Decode tokens per measurement")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep: sparse+dense × 1K/8K/16K/32K/64K")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"],
                        help="Model dtype (default: fp16)")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import torch

    dtype_map = {"fp16": "float16", "bf16": "bfloat16"}
    model_dtype = dtype_map[args.dtype]

    # Load corpus
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    corpus_text = "\n".join(v for v in corpus.values() if v)

    if args.sweep:
        configs = [
            ("dense", [1000, 8000, 16000, 32000, 64000]),
            ("sparse", [1000, 8000, 16000, 32000, 64000]),
        ]
    else:
        configs = [(args.attn, args.prefill)]

    attn_map = {"dense": "ESIMD_ATTN", "sparse": "INFLLMV2_ESIMD_ATTN"}
    all_results = []

    for attn_name, prefill_list in configs:
        print(f"\n{'='*70}")
        print(f"  Loading model: attn={attn_name}")
        print(f"{'='*70}")

        llm = LLM(
            model=MODEL, dtype=model_dtype, trust_remote_code=True,
            enforce_eager=True, gpu_memory_utilization=0.85,
            max_model_len=131072, max_num_seqs=1, block_size=128,
            enable_prefix_caching=False,
            attention_backend=attn_map[attn_name],
        )
        tokenizer = llm.get_tokenizer()

        for pf in prefill_list:
            prompt, ntoks = build_prompt(tokenizer, corpus_text, pf)
            print(f"\n--- {attn_name} | prefill={ntoks} | decode={args.decode} ---")

            # Warmup
            if not args.no_warmup:
                sp_w = SamplingParams(temperature=0, max_tokens=args.decode)
                llm.generate([prompt], sp_w)
                if hasattr(torch, "xpu"):
                    torch.xpu.synchronize()

            # Measure
            r = measure_tps(llm, prompt, ntoks, args.decode)
            r["attn"] = attn_name
            all_results.append(r)

            print(f"  Prefill: {r['prefill_ms']:.0f}ms "
                  f"({r['prefill_tps']:.0f} tok/s)")
            print(f"  Decode:  {r['decode_ms']:.0f}ms "
                  f"({r['decode_tps']:.1f} tok/s, "
                  f"{r['decode_ms_per_tok']:.1f} ms/tok)")

            # Output quality: unique 4-gram ratio + text preview
            text = r["text"]
            words = text.split()[:500]
            if len(words) >= 4:
                ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
                r["unique_4gram"] = len(set(ngrams)) / len(ngrams)
            else:
                r["unique_4gram"] = 1.0
            print(f"  Quality: 4gram={r['unique_4gram']:.3f}")
            preview = text[:1000].replace('\n', ' ')
            print(f"  Output:  {preview}")
            if len(text) > 1000:
                print(f"           ... ({len(text)} chars total)")

        del llm
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY (decode={args.decode} tokens)")
    print(f"{'='*70}")
    print(f"  {'Attn':<8} {'Prefill':>8} {'Prefill TPS':>12} "
          f"{'Decode TPS':>11} {'ms/tok':>8} {'4gram':>7}")
    print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*11} {'-'*8} {'-'*7}")
    for r in all_results:
        print(f"  {r['attn']:<8} {r['prefill_tokens']:>8,} "
              f"{r['prefill_tps']:>12,.0f} "
              f"{r['decode_tps']:>11.1f} "
              f"{r['decode_ms_per_tok']:>8.1f} "
              f"{r.get('unique_4gram', 0):>7.3f}")
    print(f"{'='*70}")
