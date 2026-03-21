"""
MiniCPM4-0.5B offline inference on Intel XPU via vLLM.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_minicpm4_xpu.py
"""

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-0.5B"


def main():
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)

    prompts = [
        "Hello, my name is",
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
