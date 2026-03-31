"""Profile sparse decode overhead: short prefill + few decode steps."""
import os, time
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["MINICPM5_ESIMD_MOE"] = "1"
os.environ["MINICPM5_ESIMD_GEMV"] = "1"
# os.environ["INFLLMV2_HOST_TIMING"] = "1"  # enable for per-section timing (adds sync overhead)

MODEL = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", default="sparse", choices=["dense", "sparse"])
    parser.add_argument("--prefill", type=int, default=32000)
    parser.add_argument("--decode", type=int, default=32)
    args = parser.parse_args()

    attn_map = {"dense": "ESIMD_ATTN", "sparse": "INFLLMV2_ESIMD_ATTN"}

    from vllm import LLM, SamplingParams
    import json, torch

    # Build prompt from corpus
    corpus_path = os.path.join(os.path.dirname(__file__), "wiki_corpus.json")
    with open(corpus_path) as f:
        corpus = json.load(f)
    text = "\n".join(v for v in corpus.values() if v)

    llm = LLM(
        model=MODEL, dtype="float16", trust_remote_code=True,
        enforce_eager=True, gpu_memory_utilization=0.85,
        max_model_len=131072, max_num_seqs=1, block_size=128,
        enable_prefix_caching=False,
        attention_backend=attn_map[args.attn],
    )
    tokenizer = llm.get_tokenizer()

    # Build prompt of target length
    tokens = tokenizer.encode(text)
    while len(tokens) < args.prefill:
        text = text + "\n" + text
        tokens = tokenizer.encode(text)
    tokens = tokens[:args.prefill]
    text = tokenizer.decode(tokens) + "\nQuestion: Summarize.\nAnswer:"
    msgs = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    ntoks = len(tokenizer.encode(prompt))
    print(f"Input: {ntoks} tokens, decode: {args.decode}, attn: {args.attn}")

    # Force full decode (no early stop) for fair timing
    sp = SamplingParams(temperature=0, max_tokens=args.decode)

    # Warmup (includes prefill + decode)
    print("Warmup...")
    out = llm.generate([prompt], sp)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    gen = len(out[0].outputs[0].token_ids)
    print(f"Warmup done, gen={gen}")

    # Timed run: Use two calls to separate prefill from decode
    # Run 1: prefill only (1 token decode)
    sp_1 = SamplingParams(temperature=0, max_tokens=1)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t0 = time.perf_counter()
    out1 = llm.generate([prompt], sp_1)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t_prefill = time.perf_counter() - t0
    print(f"Prefill: {t_prefill*1000:.0f}ms ({ntoks} tokens, "
          f"{ntoks/t_prefill:.0f} tok/s)")

    # Run 2: full run (prefill + decode)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp)
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    t1 = time.perf_counter()
    gen = len(out[0].outputs[0].token_ids)
    wall = t1 - t0
    t_decode = wall - t_prefill
    print(f"Total: {gen} tokens, wall={wall:.3f}s")
    print(f"Decode: {gen} tokens in {t_decode*1000:.0f}ms = "
          f"{gen/t_decode:.1f} tok/s ({t_decode*1000/gen:.1f} ms/tok)")
    print(f"\n{out[0].outputs[0].text[:300]}")
