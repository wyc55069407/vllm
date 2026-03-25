"""Quick test for MiniCPM5 MoE model."""
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"

if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=1,
        block_size=128,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    params = SamplingParams(temperature=0, max_tokens=512, stop=["</think>"])

    # Test with chat template
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"Formatted prompt: {repr(prompt[:100])}")

    outputs = llm.generate([prompt], params)
    for out in outputs:
        pt = len(out.prompt_token_ids)
        gt = len(out.outputs[0].token_ids)
        text = out.outputs[0].text
        print(f"Prompt tokens: {pt}, Generated tokens: {gt}")
        print(f"Output: {text}")

    # Also test raw prompt
    outputs2 = llm.generate(["The capital of France is"], params)
    for out in outputs2:
        text = out.outputs[0].text
        print(f"\nRaw prompt output: {text}")

    del llm
