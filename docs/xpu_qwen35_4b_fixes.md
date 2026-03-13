# Qwen3.5-4B on Intel XPU (BMG) — Fixes and Usage

## Overview

Qwen3.5-4B is a hybrid model combining GDN (Gated Delta Net) linear attention (20 layers) and full attention (8 layers, every 4th). Running it on Intel XPU (Battlemage, 24GB VRAM) required several fixes to vLLM.

## Fixes Applied

### 1. XPU Data Pointer int64 Overflow (`vllm/v1/worker/utils.py`)

**Problem**: XPU device data pointers can exceed int64 range, causing overflow when stored as Python int.

**Fix**: Use numpy uint64 reinterpretation for safe pointer arithmetic.

### 2. FLA Triton Kernel Device Context (`vllm/model_executor/layers/fla/ops/utils.py`)

**Problem**: The FLA (Flash Linear Attention) `input_guard` decorator only handled CUDA device context, failing on XPU.

**Fix**: Added XPU device context handling (`torch.xpu.device(...)`) alongside CUDA.

### 3. Non-contiguous K/V in Hybrid Models (`vllm/_xpu_ops.py`)

**Problem**: XPU FA2 kernel requires contiguous K/V tensors. Hybrid models produce non-contiguous K/V from layout reordering (`_update_hybrid_attention_mamba_layout`).

**Fix**: Added `.contiguous()` calls for K and V in `flash_attn_varlen_func`.

### 4. Hybrid Block Size vs FA2 (`vllm/platforms/xpu.py`)

**Problem**: Hybrid KV cache uses block_size=528 (aligned for float32 GDN state). XPU FA2 only supports block_size=64. These are mutually exclusive — block_size is system-wide.

**Fix**: Force `TRITON_ATTN` backend at runtime. The Triton attention backend supports arbitrary block sizes (any multiple of 16). Updated `xpu.py` to allow block_size >= 64 for hybrid models.

### 5. GDN Prefill NaN in Final State (`vllm/model_executor/models/qwen3_next.py`)

**Problem**: The `chunk_gated_delta_rule` FLA Triton kernel produces NaN in the final recurrent state on XPU, while the chunked output computation is correct. This caused all subsequent decode steps to produce garbage ("!!!!" tokens), since decode reads the corrupted state.

- 19 out of 20 GDN layers exhibit this bug (layer 0 is fine)
- Affects both chunked and non-chunked prefill on XPU
- Root cause: likely a Triton XPU codegen issue in the state accumulation path

**Fix**: After `chunk_gated_delta_rule` runs, check for NaN in `last_recurrent_state`. If found, recompute the final state using a PyTorch sequential reference (`_fused_sigmoid_gating_delta_rule_update_pytorch_ref`). This preserves the fast output from the chunk kernel while ensuring correct state for decode.

Additionally added `_causal_conv1d_update_pytorch_ref` as a fallback for the conv1d decode kernel (toggled via `VLLM_GDN_PYTORCH_DECODE=1` env var for debugging).

### 6. GPTQ Dequantization Fallback (`vllm/model_executor/layers/quantization/gptq.py`)

**Problem**: GPTQ CUDA kernels (`gptq_gemm`, `gptq_shuffle`) are not available on XPU.

**Fix**: Added PyTorch-native dequantization fallback that unpacks INT4/INT8 weights and applies group-wise scaling on XPU.

## Why FA2 Backend Cannot Work with Hybrid Models

The XPU FA2 kernel requires `block_size=64` for paged attention. Hybrid models require `block_size=528` to store GDN recurrent state in the unified page. These are mutually exclusive because block_size is a system-wide parameter in vLLM's KV cache allocator.

The Triton attention backend supports arbitrary block sizes, making it the correct choice for hybrid models on XPU.

## How to Run

```bash
# Activate environment
conda activate vllm_xpu
source ~/intel/oneapi/setvars.sh --force

cd ~/yuchen/vllm_env/vllm

# Basic run (recommended)
python test_qwen35_4b_xpu.py

# With debug logging for GDN kernels
VLLM_GDN_DEBUG=1 python test_qwen35_4b_xpu.py

# With PyTorch reference decode (for debugging Triton kernel issues)
VLLM_GDN_PYTORCH_DECODE=1 VLLM_GDN_DEBUG=1 python test_qwen35_4b_xpu.py
```

### Key Parameters

```python
LLM(
    model="/path/to/Qwen3.5-4b",
    dtype="bfloat16",
    enforce_eager=True,              # No torch.compile on XPU yet
    gpu_memory_utilization=0.5,      # Fit within 24GB BMG VRAM
    max_model_len=512,               # Limit context for memory
    attention_backend="TRITON_ATTN", # Required: FA2 incompatible with hybrid block_size
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_GDN_PYTORCH_DECODE` | `0` | Use PyTorch reference for GDN decode kernels (conv1d + recurrent update) |
| `VLLM_GDN_DEBUG` | `0` | Enable per-layer debug logging for GDN state and outputs |

## Files Modified

| File | Change |
|------|--------|
| `vllm/_xpu_ops.py` | K/V contiguity for hybrid models |
| `vllm/model_executor/layers/fla/ops/utils.py` | XPU device context in input_guard |
| `vllm/model_executor/layers/quantization/gptq.py` | PyTorch GPTQ dequant fallback for XPU |
| `vllm/model_executor/models/qwen3_next.py` | NaN state repair + PyTorch reference kernels |
| `vllm/platforms/xpu.py` | Allow block_size >= 64 for hybrid models |
| `vllm/v1/worker/utils.py` | int64 overflow fix for XPU data pointers |
| `test_qwen35_4b_xpu.py` | Test script for Qwen3.5-4B on XPU |
