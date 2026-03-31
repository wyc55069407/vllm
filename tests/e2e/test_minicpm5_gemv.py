"""MiniCPM5 E2E test with all decode optimizations enabled:
- ESIMD MoE (decode + prefill)
- ESIMD GEMV for QKV/O projections
- ESIMD fused shared expert
- ESIMD neox-style RoPE
"""
import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["MINICPM5_ESIMD_MOE"] = "1"
os.environ["MINICPM5_ESIMD_GEMV"] = "1"

MODEL = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"

if __name__ == "__main__":
    import time
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, gpu_memory_utilization=0.85,
        max_model_len=131072, max_num_seqs=1, block_size=128,
        enable_prefix_caching=False, attention_backend="ESIMD_ATTN",
    )

    prompt = (
        "Explain the key differences between classical mechanics and "
        "quantum mechanics. Cover the historical development, main "
        "principles, mathematical frameworks, and practical applications "
        "of each. Be detailed and thorough."
    )
    msgs = [{"role": "user", "content": prompt}]
    tokenizer = llm.get_tokenizer()
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    ntoks = len(tokenizer.encode(text))
    print(f"Input: {ntoks} tokens")

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(temperature=0, max_tokens=256,
                        stop_token_ids=[im_end])

    t0 = time.perf_counter()
    out = llm.generate([text], sp)
    t1 = time.perf_counter()

    gen_text = out[0].outputs[0].text
    gen_toks = len(out[0].outputs[0].token_ids)
    wall = t1 - t0
    print(f"Output: {gen_toks} tokens, wall={wall:.1f}s")
    if gen_toks > 1:
        print(f"Decode throughput: {gen_toks/wall:.1f} tok/s")
    print(f"\n{gen_text[:500]}")
