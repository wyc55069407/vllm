"""Profile MiniCPM5 multi-batch decode at short context with ESIMD MoE.

Tests concurrent decode with 1-8 sequences using dense ESIMD_ATTN.

Usage:
  MINICPM5_ESIMD_MOE=1 python test_minicpm5_profile_short.py --num-seqs 1
  MINICPM5_ESIMD_MOE=1 python test_minicpm5_profile_short.py --num-seqs 4
  MINICPM5_ESIMD_MOE=1 python test_minicpm5_profile_short.py --num-seqs 8
"""
import os, sys, argparse
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("MINICPM5_ESIMD_MOE", "1")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"

PROMPTS = [
    "What is 2+2? Answer briefly.",
    "Name the three primary colors.",
    "What is the capital of France?",
    "How many continents are there?",
    "What is the chemical symbol for water?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in km/s?",
    "Name the largest planet in our solar system.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seqs", type=int, default=1,
                        help="Number of concurrent sequences (1-8)")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    assert 1 <= args.num_seqs <= 8, "num-seqs must be 1..8"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"Multi-batch decode: num_seqs={args.num_seqs}, max_tokens={args.max_tokens}")
    print(f"MINICPM5_ESIMD_MOE={os.environ.get('MINICPM5_ESIMD_MOE', '0')}")

    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.75,
        max_model_len=2048, max_num_seqs=args.num_seqs, block_size=128,
        attention_backend="ESIMD_ATTN",
    )

    # Build prompts with chat template
    prompts = []
    for i in range(args.num_seqs):
        msgs = [{"role": "user", "content": PROMPTS[i]}]
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p += "<think>\n</think>\n\n"
        prompts.append(p)

    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, params)

    for i, out in enumerate(outputs):
        pt = len(out.prompt_token_ids)
        gt = len(out.outputs[0].token_ids)
        text = out.outputs[0].text[:120].replace('\n', ' ')
        print(f"  [{i}] prompt={pt}, gen={gt}: {text}")

    del llm
