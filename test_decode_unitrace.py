"""Decode-focused test for unitrace profiling.

Small prefill (~8K tokens) then many decode tokens (~100).
Run with unitrace to capture per-kernel decode hotspots.

Usage:
  # Direct run (check correctness):
  python test_decode_unitrace.py

  # With unitrace (profile decode kernels):
  unitrace --device-timing -v python test_decode_unitrace.py 2>&1 | tee unitrace_decode.log
"""
import os, time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=84000,
                        help="Approx context length in chars (~tokens/4)")
    parser.add_argument("--mode", default="sparse", choices=["dense", "sparse"])
    args = parser.parse_args()

    backend = "INFLLMV2_ESIMD_ATTN" if args.mode == "sparse" else "ESIMD_ATTN"

    # Build a long context prompt
    filler = "The quick brown fox jumps over the lazy dog. " * 100
    repeat = args.context_len // len(filler) + 1
    long_text = (filler * repeat)[:args.context_len]
    prompt = long_text + "\n\nSummarize the above text in 3 sentences."

    print(f"Mode: {args.mode}, context ~{len(prompt)//4} tokens, max_tokens={args.max_tokens}")

    from vllm import LLM, SamplingParams

    max_ctx = 131072
    hf_overrides = {
        "max_position_embeddings": max_ctx,
        "rope_scaling": {"rope_type": "dynamic", "factor": 4.0},
    }

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
              max_model_len=max_ctx, max_num_seqs=1, block_size=128,
              attention_backend=backend, hf_overrides=hf_overrides)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    t1 = time.perf_counter()

    out = outputs[0]
    pt = len(out.prompt_token_ids)
    gt = len(out.outputs[0].token_ids)
    total_time = t1 - t0

    # Estimate decode time (total - prefill estimate)
    # Rough: prefill at ~500 tok/s for sparse at 84K
    prefill_est = pt / 500.0
    decode_est = total_time - prefill_est
    decode_tps = gt / decode_est if decode_est > 0 else 0

    print(f"\nPrompt tokens: {pt:,}")
    print(f"Generated tokens: {gt}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Estimated decode: {decode_est:.2f}s ({decode_tps:.1f} tok/s)")
    print(f"\nOutput: {out.outputs[0].text[:200]}...")

    del llm
