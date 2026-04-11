"""Unified MiniCPM performance & quality benchmark.

Swiss-knife script for MiniCPM4-8B / MiniCPM5-16B on Intel XPU.
Supports sparse vs dense attention, separate prefill/decode TPS,
diverse wiki_corpus content, quality metrics, and output preview.

Examples:
  # MiniCPM4-8B, sparse, 16K in / 256 out
  python tests/e2e/test_perf.py --model 8b --attn sparse --prefill 16000 --decode 256

  # MiniCPM5-16B, dense, 32K in / 512 out, bf16
  python tests/e2e/test_perf.py --model 16b --attn dense --prefill 32000 --decode 512 --dtype bf16

  # Sweep: both sparse and dense at multiple context lengths
  python tests/e2e/test_perf.py --model 8b --sweep --prefill 4000 8000 16000 24000

  # Quick comparison: sparse vs dense at single length
  python tests/e2e/test_perf.py --model 8b --compare --prefill 16000

  # MiniCPM5 with Triton MoE (no ESIMD MoE)
  python tests/e2e/test_perf.py --model 16b --no-esimd-moe --prefill 32000

  # Custom: 84K prefill, rep penalty, multiple runs
  python tests/e2e/test_perf.py --model 16b --prefill 84000 --decode 1024 --rep-pen 1.1 --runs 3
"""
import os, time, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODELS = {
    "8b": {
        "path": "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B-GPTQ-Int4",
        "name": "MiniCPM4-8B-GPTQ-Int4",
        "base_max_model_len": 32768,
        "env_moe": "MINICPM4_ESIMD_MOE",
        "env_gemv": "MINICPM4_ESIMD_GEMV",
    },
    "16b": {
        "path": "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4",
        "name": "MiniCPM5-16B-GPTQ-Int4",
        "base_max_model_len": 131072,
        "env_moe": "MINICPM5_ESIMD_MOE",
        "env_gemv": "MINICPM5_ESIMD_GEMV",
    },
}

def get_rope_scaling(model_key, max_prefill, force_rope=False):
    """Return (max_model_len, hf_overrides) for the requested prefill length.

    Both MiniCPM4-8B and MiniCPM5-16B use LongRoPE with trained
    per-dimension long_factor/short_factor arrays.  Do NOT override
    rope_type — that would discard the trained factors and use a
    generic dynamic/NTK scheme instead.

    Just extend max_position_embeddings.  Phi3LongRoPEScaledRotaryEmbedding
    (vLLM) auto-selects long_factor when max_model_len > original_max and
    computes the cos/sin cache up to the new max_position_embeddings.
    """
    cfg = MODELS[model_key]
    base = cfg["base_max_model_len"]

    if max_prefill + 2048 <= base:
        # Plenty of room, no extension needed
        return base, None

    new_max = max_prefill + 2048
    hf_overrides = {
        "max_position_embeddings": new_max,
    }
    return new_max, hf_overrides

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "wiki_corpus.json")

QUESTION = (
    "Based on all the articles above, answer the following questions in detail:\n"
    "1. List the 3 most important people mentioned across all topics, "
    "and explain their contributions with specific dates and numbers.\n"
    "2. What are the key scientific or technical concepts discussed? "
    "Explain each with at least one concrete example from the text.\n"
    "3. Identify any connections or common themes between different topics.\n"
    "4. What are the most surprising or counterintuitive facts mentioned?\n"
    "5. Provide a timeline of the 5 most significant events mentioned, "
    "ordered chronologically with exact dates where available.\n"
    "Be thorough and detailed in each answer."
)


def load_corpus():
    """Load diverse wiki_corpus.json content."""
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    sections = []
    for topic, text in corpus.items():
        if text:
            sections.append(f"\n\n=== {topic} ===\n\n{text}")
    return "".join(sections)


def build_prompt(tokenizer, corpus, target_tokens):
    """Build prompt from diverse content, duplicated/truncated to target length."""
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
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    ntoks = len(tokenizer.encode(prompt))
    return prompt, ntoks


def measure(llm, prompt, ntoks, decode_tokens, sp_kwargs, warmup=True):
    """Measure prefill and decode TPS separately."""
    import torch
    from vllm import SamplingParams

    sp = SamplingParams(**sp_kwargs, max_tokens=decode_tokens)

    # Warmup
    if warmup:
        llm.generate([prompt], sp)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()

    # Prefill-only (1 token)
    sp1 = SamplingParams(**sp_kwargs, max_tokens=1)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t0 = time.perf_counter()
    llm.generate([prompt], sp1)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_prefill = time.perf_counter() - t0

    # Full run (prefill + decode)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_full = time.perf_counter() - t0

    gen = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text
    t_decode = t_full - t_prefill
    decode_toks = max(1, gen - 1)

    prefill_tps = ntoks / t_prefill
    decode_tps = decode_toks / t_decode if t_decode > 0 else 0
    decode_ms = t_decode * 1000 / decode_toks if decode_toks > 0 else 0

    # Quality: unique 4-gram ratio
    words = text.split()[:500]
    if len(words) >= 4:
        ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        u4gram = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    else:
        u4gram = 1.0

    return {
        "prefill_tokens": ntoks,
        "decode_tokens": gen,
        "prefill_ms": t_prefill * 1000,
        "decode_ms_total": t_decode * 1000,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "decode_ms_per_tok": decode_ms,
        "u4gram": u4gram,
        "text": text,
    }


def print_result(r, attn_name):
    """Print a single measurement result."""
    print(f"  Prefill: {r['prefill_tps']:.0f} tok/s ({r['prefill_ms']:.0f}ms)")
    print(f"  Decode:  {r['decode_tps']:.1f} tok/s "
          f"({r['decode_ms_per_tok']:.1f} ms/tok)")
    print(f"  Quality: gen={r['decode_tokens']}, 4gram={r['u4gram']:.3f}")
    preview = r["text"][:800].replace('\n', ' ')
    print(f"  Output:  {preview}")
    if len(r["text"]) > 800:
        print(f"           ... ({len(r['text'])} chars total)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified MiniCPM performance & quality benchmark")
    parser.add_argument("--model", default="8b", choices=["8b", "16b"],
                        help="Model: 8b=MiniCPM4-8B-GPTQ-Int4, "
                             "16b=MiniCPM5-16B-GPTQ-Int4 (default: 8b)")
    parser.add_argument("--attn", default="dense",
                        choices=["dense", "sparse", "triton"],
                        help="Attention backend (default: dense)")
    parser.add_argument("--prefill", type=int, nargs="+", default=[16000],
                        help="Prefill token counts (default: 16000)")
    parser.add_argument("--decode", type=int, default=256,
                        help="Max decode tokens (default: 256)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"],
                        help="Model dtype (default: fp16)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both dense and sparse for comparison")
    parser.add_argument("--sweep", action="store_true",
                        help="Run both dense and sparse at all prefill lengths")
    parser.add_argument("--no-esimd-moe", action="store_true",
                        help="Disable ESIMD MoE (use Triton MoE)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup run")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of timed runs per config (default: 1)")
    parser.add_argument("--rep-pen", type=float, default=1.0,
                        help="Repetition penalty (1.0=off)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--force-rope", action="store_true",
                        help="Force dynamic NTK rope scaling even for 16B model")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]

    # Set env for ESIMD MoE
    if not args.no_esimd_moe:
        os.environ.setdefault(model_cfg["env_moe"], "1")
        if model_cfg["env_gemv"]:
            os.environ.setdefault(model_cfg["env_gemv"], "1")
    else:
        os.environ[model_cfg["env_moe"]] = "0"

    from vllm import LLM, SamplingParams
    import torch

    dtype_map = {"fp16": "float16", "bf16": "bfloat16"}
    model_dtype = dtype_map[args.dtype]

    attn_map = {
        "dense": "ESIMD_ATTN",
        "sparse": "INFLLMV2_ESIMD_ATTN",
        "triton": None,
    }

    # Determine attention configs to run
    if args.sweep or args.compare:
        attn_configs = ["dense", "sparse"]
    else:
        attn_configs = [args.attn]

    # Load corpus once
    corpus = load_corpus()

    # Compute rope scaling based on max prefill requested
    max_prefill = max(args.prefill)
    max_model_len, hf_overrides = get_rope_scaling(args.model, max_prefill,
                                                    args.force_rope)

    moe_label = "ESIMD" if not args.no_esimd_moe else "Triton"
    all_results = []

    for attn_name in attn_configs:
        rope_label = ""
        if hf_overrides and "rope_scaling" in hf_overrides:
            rope_label = (f" | RoPE={hf_overrides['rope_scaling']['rope_type']}"
                          f"(f={hf_overrides['rope_scaling']['factor']:.1f})")

        print(f"\n{'='*70}")
        print(f"  {model_cfg['name']} | attn={attn_name} | "
              f"dtype={args.dtype} | MoE={moe_label}{rope_label}")
        print(f"{'='*70}")

        llm_kwargs = dict(
            model=model_cfg["path"], dtype=model_dtype,
            trust_remote_code=True, enforce_eager=True,
            disable_log_stats=True, gpu_memory_utilization=0.85,
            max_model_len=max_model_len,
            max_num_seqs=1, block_size=128,
            enable_prefix_caching=False,
        )
        if attn_map[attn_name]:
            llm_kwargs["attention_backend"] = attn_map[attn_name]
        if hf_overrides:
            llm_kwargs["hf_overrides"] = hf_overrides

        t_load = time.perf_counter()
        llm = LLM(**llm_kwargs)
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        print(f"  Model loaded in {time.perf_counter()-t_load:.1f}s")

        tokenizer = llm.get_tokenizer()

        # Use <|im_end|> as stop token (matches chat template).
        # ignore_eos=True prevents default eos_token from stopping early.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        sp_kwargs = {"temperature": args.temperature,
                     "stop_token_ids": [im_end_id],
                     "ignore_eos": True}
        if args.rep_pen != 1.0:
            sp_kwargs["repetition_penalty"] = args.rep_pen

        for pf in args.prefill:
            prompt, ntoks = build_prompt(tokenizer, corpus, pf)
            print(f"\n--- {attn_name} | prefill={ntoks} | "
                  f"decode={args.decode} ---")

            for run_i in range(args.runs):
                r = measure(llm, prompt, ntoks, args.decode,
                            sp_kwargs, warmup=not args.no_warmup)
                r["attn"] = attn_name
                r["run"] = run_i
                all_results.append(r)

                if args.runs > 1:
                    print(f"  [Run {run_i}]")
                print_result(r, attn_name)

        del llm
        if hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {model_cfg['name']} | dtype={args.dtype} | "
          f"decode={args.decode}")
    print(f"{'='*70}")
    print(f"  {'Attn':<8} {'Prefill':>8} {'PF TPS':>9} "
          f"{'Dec TPS':>9} {'ms/tok':>8} {'4gram':>7} {'Gen':>5}")
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*5}")
    for r in all_results:
        print(f"  {r['attn']:<8} {r['prefill_tokens']:>8,} "
              f"{r['prefill_tps']:>9,.0f} "
              f"{r['decode_tps']:>9.1f} "
              f"{r['decode_ms_per_tok']:>8.1f} "
              f"{r['u4gram']:>7.3f} "
              f"{r['decode_tokens']:>5}")
    print(f"{'='*70}")

    # Output text for each result
    for r in all_results:
        print(f"\n--- {r['attn']} | prefill={r['prefill_tokens']} | "
              f"gen={r['decode_tokens']} ---")
        preview = r["text"][:1500].replace('\n', ' ')
        print(f"  {preview}")
        if len(r["text"]) > 1500:
            print(f"  ... ({len(r['text'])} chars total)")
