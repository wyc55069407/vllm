"""Test MiniCPM5 MoE model with longer prompts and outputs."""
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

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

    params = SamplingParams(temperature=0, max_tokens=1024)

    # Test 1: Simple math with chat template
    msgs1 = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    p1 = tokenizer.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=True)

    # Test 2: Longer context - summarize a passage
    passage = """The Apollo 11 mission was the spaceflight that first landed humans on the Moon.
Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew that
landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. Armstrong became the
first person to step onto the lunar surface six hours and 39 minutes later on July 21 at
02:56 UTC; Aldrin joined him 19 minutes later. They spent about two and a quarter hours
together exploring the site they had named Tranquility Base upon landing. Armstrong and Aldrin
collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth as pilot Michael
Collins flew the Command Module Columbia in lunar orbit, and were on the Moon's surface for
21 hours, 36 minutes before lifting off to rejoin Columbia."""
    msgs2 = [{"role": "user", "content": f"Summarize this passage in 3 bullet points:\n\n{passage}"}]
    p2 = tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)

    # Test 3: Code generation
    msgs3 = [{"role": "user", "content": "Write a Python function to compute the nth Fibonacci number using dynamic programming. Include a docstring."}]
    p3 = tokenizer.apply_chat_template(msgs3, tokenize=False, add_generation_prompt=True)

    # Test 4: Multi-turn conversation
    msgs4 = [
        {"role": "user", "content": "What are the three laws of thermodynamics?"},
        {"role": "assistant", "content": "<think>\nThe user is asking about the three laws of thermodynamics.\n</think>\n\nThe three laws of thermodynamics are:\n1. Energy cannot be created or destroyed, only transformed.\n2. Entropy of an isolated system always increases.\n3. As temperature approaches absolute zero, entropy approaches a minimum."},
        {"role": "user", "content": "Can you explain the second law in more detail with a real-world example?"},
    ]
    p4 = tokenizer.apply_chat_template(msgs4, tokenize=False, add_generation_prompt=True)

    prompts = [p1, p2, p3, p4]
    labels = ["Simple math", "Summarization", "Code generation", "Multi-turn"]

    for label, prompt in zip(labels, prompts):
        tids = tokenizer.encode(prompt)
        print(f"\n{'='*70}")
        print(f"  TEST: {label} ({len(tids)} prompt tokens)")
        print(f"{'='*70}")
        outputs = llm.generate([prompt], params)
        out = outputs[0]
        pt = len(out.prompt_token_ids)
        gt = len(out.outputs[0].token_ids)
        text = out.outputs[0].text
        print(f"Prompt tokens: {pt}, Generated tokens: {gt}")
        print(f"Output:\n{text}")

    del llm
