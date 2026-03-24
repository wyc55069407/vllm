"""Test sparse prefill+decode at ~28K tokens.
Uses the HP+Travel 25K haystack (unique content) and asks for summarization.
Compares sparse vs dense output quality.
"""
import os, sys
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B"
HP_TRAVEL_FILE = "/home/sas/yuchen/vllm_env/Harry_potter_travel_25K_prompt.txt"

QUESTION = """Based on the text above, please answer the following:
1. Summarize the Harry Potter story in 2-3 sentences, mentioning specific character names.
2. Summarize the first travel itinerary (European Capitals) in 2-3 sentences, mentioning specific cities.
3. Summarize the second travel itinerary (Iberian Peninsula) in 2-3 sentences, mentioning specific cities.
4. What are 3 specific differences between the two travel itineraries?"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sparse", choices=["dense", "sparse", "dense_prefill"])
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "dense":
        backend = "ESIMD_ATTN"
    elif args.mode == "sparse":
        backend = "INFLLMV2_ESIMD_ATTN"
    elif args.mode == "dense_prefill":
        backend = "INFLLMV2_ESIMD_ATTN"
        os.environ["INFLLMV2_SPARSE_PREFILL"] = "0"

    with open(HP_TRAVEL_FILE, "r") as f:
        haystack = f.read()

    # Replace the original instruction with our question at the end
    # Strip original instruction (first 5 lines)
    lines = haystack.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if "<<First book>>" in line:
            body_start = i
            break
    body = "\n".join(lines[body_start:])

    prompt = "You are a helpful AI assistant. Read the following text carefully and answer the questions at the end.\n\n" + body + "\n\n" + QUESTION

    print(f"Mode: {args.mode}, backend: {backend}")

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
              max_model_len=32768, max_num_seqs=4, block_size=128,
              attention_backend=backend)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    outputs = llm.generate([prompt], params)
    out = outputs[0]
    pt = len(out.prompt_token_ids)
    gt = len(out.outputs[0].token_ids)
    text = out.outputs[0].text

    print(f"Prompt tokens: {pt}")
    print(f"Generated tokens: {gt}")
    print(f"\n{'='*70}")
    print(f"  OUTPUT ({args.mode})")
    print(f"{'='*70}")
    print(text)
    print(f"{'='*70}")

    # Quality check
    tl = text.lower()
    checks = {
        "HP characters": any(w in tl for w in ["harry", "dumbledore", "dursley", "hogwarts", "voldemort"]),
        "European cities": any(w in tl for w in ["paris", "london", "rome"]),
        "Iberian cities": any(w in tl for w in ["barcelona", "madrid", "lisbon", "porto", "seville"]),
        "Differences": "differ" in tl or "while" in tl or "contrast" in tl or "first" in tl,
    }
    print(f"\nQuality check:")
    for k, v in checks.items():
        print(f"  {k}: {'YES' if v else 'NO'}")
    n = sum(checks.values())
    print(f"  Score: {n}/4")

    del llm
