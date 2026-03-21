"""
2-round conversation test to observe prefill vs extend vs decode attention.
Patches flash_attn_varlen_func to log whether block_table (paged) is used.

Usage:
    conda activate vllm_xpu
    source ~/intel/oneapi/setvars.sh --force
    cd ~/yuchen/vllm_env/vllm
    python test_minicpm4_2round.py
"""

import vllm_xpu_kernels.flash_attn_interface as _fa_mod

_orig_func = _fa_mod.flash_attn_varlen_func
_call_count = 0


def _traced_fa(**kwargs):
    global _call_count
    _call_count += 1
    q = kwargs.get("q")
    block_table = kwargs.get("block_table")
    cu_seqlens_q = kwargs.get("cu_seqlens_q")
    cu_seqlens_k = kwargs.get("cu_seqlens_k")
    seqused_k = kwargs.get("seqused_k")
    max_seqlen_q = kwargs.get("max_seqlen_q")
    max_seqlen_k = kwargs.get("max_seqlen_k")

    paged = block_table is not None
    mode = "PAGED" if paged else "NON-PAGED"
    q_shape = tuple(q.shape) if q is not None else "?"
    bt_shape = tuple(block_table.shape) if paged else "None"
    cq = cu_seqlens_q.tolist() if cu_seqlens_q is not None else "None"
    ck = cu_seqlens_k.tolist() if cu_seqlens_k is not None else "None"
    sk = seqused_k.tolist() if seqused_k is not None else "None"

    print(
        f"  [FA2 #{_call_count:3d}] {mode:>9s} | "
        f"q={str(q_shape):>20s} | "
        f"max_q={max_seqlen_q:>5} max_k={max_seqlen_k:>5} | "
        f"block_table={str(bt_shape):>12s} | "
        f"cu_q={cq} cu_k={ck} seqused_k={sk}",
        flush=True,
    )
    return _orig_func(**kwargs)


# Patch at module level so child process picks it up
_fa_mod.flash_attn_varlen_func = _traced_fa


def main():
    global _call_count
    from vllm import LLM, SamplingParams

    MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-0.5B"

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    # --- Round 1: initial prefill + decode ---
    print("\n" + "=" * 80, flush=True)
    print("ROUND 1: Initial prompt (prefill + decode)", flush=True)
    print("=" * 80, flush=True)
    _call_count = 0
    conversation = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    out1 = llm.chat(conversation, sampling_params)
    reply1 = out1[0].outputs[0].text
    print(f"\nReply: {reply1}", flush=True)

    # --- Round 2: extend context (2nd prefill with KV cache) + decode ---
    print("\n" + "=" * 80, flush=True)
    print("ROUND 2: Follow-up (extend/2nd prefill + decode)", flush=True)
    print("=" * 80, flush=True)
    _call_count = 0
    conversation.append({"role": "assistant", "content": reply1})
    conversation.append({"role": "user", "content": "And what about Germany?"})
    out2 = llm.chat(conversation, sampling_params)
    reply2 = out2[0].outputs[0].text
    print(f"\nReply: {reply2}", flush=True)


if __name__ == "__main__":
    main()
