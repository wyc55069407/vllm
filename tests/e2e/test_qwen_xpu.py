"""
MiniCPM4-0.5B offline inference on Intel XPU via vLLM.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_minicpm4_xpu.py
"""

from vllm import LLM, SamplingParams

# MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-0.5B"
# MODEL_PATH = "/home/sas/yuchen/vllm_env/models/Qwen3.5-35B-A3B-GPTQ-Int4"
MODEL_PATH = "/home/sas/yuchen/vllm_env/models/Qwen3.5-4b"
# MODEL_PATH = "/home/sas/yuchen/vllm_env/models/Qwen3-30B-A3B-GPTQ-Int4"

def main():
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        trust_remote_code=True,
        # enable_chunked_prefill=False,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.8,
        # block_size=128,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

    prompts = [
        "Shanghai is",
        "The capital of France is",
        "Explain quantum computing in one sentence:",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt:    {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("---")


if __name__ == "__main__":
    main()
