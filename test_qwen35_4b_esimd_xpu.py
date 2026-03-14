"""
Qwen3.5-4B ESIMD end-to-end test on Intel XPU via vLLM.

Tests the full ESIMD kernel stack: ESIMD paged SDP for attention layers
and ESIMD GDN for linear attention layers. No Triton kernels used.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_qwen35_4b_esimd_xpu.py
"""

import os

# Enable ESIMD GDN kernel (replaces Triton chunk_gated_delta_rule)
os.environ["VLLM_GDN_ESIMD"] = "1"
os.environ.setdefault("VLLM_GDN_DEBUG", "0")

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/Qwen3.5-4b"


def main():
    print("=" * 60)
    print("Qwen3.5-4B ESIMD End-to-End Test")
    print("  GDN kernel: ESIMD (VLLM_GDN_ESIMD=1)")
    print("  Attention:  ESIMD_ATTN")
    print("=" * 60)

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        # ESIMD attention backend (no Triton ATTN)
        attention_backend="ESIMD_ATTN",
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=100)

    prompts = [
        "Shanghai is",
        "The capital of France is",
        "1 + 1 =",
    ]

    print("\n--- Generating (greedy, max_tokens=30) ---")
    outputs = llm.generate(prompts, sampling_params)

    all_ok = True
    for output in outputs:
        text = output.outputs[0].text
        print(f"Prompt:    {output.prompt}")
        print(f"Generated: {text}")

        # Basic sanity: output should not be empty or all special tokens
        if len(text.strip()) == 0:
            print("  ** WARNING: empty output!")
            all_ok = False

        # Check for NaN-like garbage (repeated identical tokens often signal NaN)
        tokens = text.split()
        if len(tokens) > 5 and len(set(tokens)) == 1:
            print("  ** WARNING: all identical tokens (possible NaN corruption)")
            all_ok = False

        print("---")

    # Run a second batch to test decode path (state reuse)
    print("\n--- Second batch (tests decode state continuity) ---")
    outputs2 = llm.generate(["Tell me a joke about programming:"], sampling_params)
    for output in outputs2:
        text = output.outputs[0].text
        print(f"Prompt:    {output.prompt}")
        print(f"Generated: {text}")
        if len(text.strip()) == 0:
            print("  ** WARNING: empty output!")
            all_ok = False
        print("---")

    if all_ok:
        print("\nPASS: All outputs look reasonable.")
    else:
        print("\nWARNING: Some outputs may have issues. Check above.")


if __name__ == "__main__":
    main()
