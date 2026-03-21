"""
MiniCPM4-8B offline inference on Intel XPU via vLLM
with ESIMD_ATTN attention backend (HD=128 optimized paged decode + DPAS prefill).

Model config: 32 Q heads, 2 KV heads (GQA 16:1), HD=128.
Tests both bf16 and fp16 dtypes.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_minicpm4_8b_esimd_xpu.py
"""

from vllm import LLM, SamplingParams


MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B"

PROMPTS = [
    "Shanghai is",
    "The capital of France is",
    "Explain quantum computing in one sentence:",
]


def run_inference(dtype_str: str):
    print(f"\n{'='*60}")
    print(f"MiniCPM4-8B with ESIMD_ATTN, dtype={dtype_str}")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL_PATH,
        dtype=dtype_str,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.90,
        max_model_len=64,
        block_size=128,
        attention_backend="ESIMD_ATTN",
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=20)
    outputs = llm.generate(PROMPTS, sampling_params)

    for output in outputs:
        print(f"Prompt:    {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("---")

    # Cleanup
    del llm


def main():
    run_inference("bfloat16")
    run_inference("float16")


if __name__ == "__main__":
    main()
