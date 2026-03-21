"""E2E test: MiniCPM4-8B-GPTQ-Int4 with ESIMD_ATTN and default attention.
HD=128, 32Q/2KV heads. Attention NOT excluded from GPTQ quantization.
Tests both bf16 and fp16 dtypes.

Usage:
    source ~/intel/oneapi/setvars.sh --force
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate vllm_xpu
    python test_minicpm4_8b_gptq.py
"""
import gc, torch, sys
from vllm import LLM, SamplingParams

MODEL = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B-GPTQ-Int4"
PROMPTS = [
    "What is the capital of France?",
    "1 + 1 =",
    "The quick brown fox",
]

def run_test(dtype_str, attn_backend, block_size=None):
    label = f"dtype={dtype_str}  attn={attn_backend}"
    print(f"\n{'='*60}")
    print(f"MiniCPM4-8B-GPTQ-Int4  {label}")
    print(f"{'='*60}")
    kwargs = dict(
        model=MODEL,
        dtype=dtype_str,
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    if attn_backend == "ESIMD_ATTN":
        kwargs["attention_backend"] = "ESIMD_ATTN"
        kwargs["block_size"] = block_size or 128
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm.generate(PROMPTS, sp)
    ok = True
    for out in outputs:
        text = out.outputs[0].text
        tids = list(out.outputs[0].token_ids[:10])
        print(f"\nPrompt: {out.prompt}")
        print(f"Output: {text.strip()}")
        print(f"Tokens: {tids}")
        stripped = text.strip()
        if len(stripped) < 2:
            print("  ** WARNING: very short output **")
            ok = False
        words = stripped.split()
        if len(words) > 5 and len(set(words)) == 1:
            print("  ** WARNING: all identical tokens (garbage?) **")
            ok = False
    del llm
    gc.collect()
    torch.xpu.empty_cache()
    return ok

if __name__ == "__main__":
    configs = [
        ("bfloat16", "ESIMD_ATTN"),
        ("float16",  "ESIMD_ATTN"),
        ("bfloat16", "default"),
        ("float16",  "default"),
    ]
    results = {}
    for dtype_str, attn in configs:
        key = f"{dtype_str}+{attn}"
        try:
            results[key] = run_test(dtype_str, attn)
        except Exception as e:
            print(f"\nERROR with {key}: {e}")
            import traceback; traceback.print_exc()
            results[key] = False
        gc.collect()
        torch.xpu.empty_cache()

    print(f"\n{'='*60}")
    print("Summary:")
    for key, ok in results.items():
        print(f"  {key}: {'PASS' if ok else 'FAIL'}")
