"""
Qwen3-30B-A3B-GPTQ-Int4 offline inference on Intel XPU via vLLM.

MoE model (128 experts, 8 active) with GPTQ 4-bit quantization.
Uses PyTorch dequant fallback for GPTQ on XPU.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_qwen3_30b_gptq_xpu.py
"""

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/Qwen3-30B-A3B-GPTQ-Int4"


def main():
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.30,
        max_model_len=64,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=20)

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
