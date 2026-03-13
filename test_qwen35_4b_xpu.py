"""
Qwen3.5-4B offline inference on Intel XPU via vLLM.

This is a hybrid model (GDN linear attention + full attention).
Uses aggressive memory params to fit within 24GB BMG VRAM.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    # With PyTorch reference decode (for debugging):
    VLLM_GDN_PYTORCH_DECODE=1 VLLM_GDN_DEBUG=1 python test_qwen35_4b_xpu.py
    # Without (original Triton kernels):
    python test_qwen35_4b_xpu.py
"""

import os

# Enable PyTorch reference for GDN decode to debug Triton kernel issues
os.environ.setdefault("VLLM_GDN_PYTORCH_DECODE", "0")
os.environ.setdefault("VLLM_GDN_DEBUG", "0")

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/Qwen3.5-4b"


def main():
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        # Use Triton attention (supports arbitrary block_size)
        # instead of FA2 which requires block_size=64.
        # Hybrid model aligns block_size to ~528 for float32 state.
        attention_backend="TRITON_ATTN",
        # Disable chunked prefill: chunk_gated_delta_rule Triton kernel
        # produces NaN in final state on XPU when processing continuation
        # chunks (2nd+ chunks with initial state).
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=20)

    prompts = [
        "Shanghai is",
        "The capital of France is",
        "1 + 1 =",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt:    {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("---")


if __name__ == "__main__":
    main()
