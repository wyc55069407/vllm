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

### 7. Hybrid Block Size Power-of-2 Rounding (`vllm/model_executor/models/config.py`)

**Problem**: The computed hybrid block_size (528) is not a power of 2. While Triton ATTN accepts it, power-of-2 sizes are preferred for memory alignment and kernel performance.

**Fix**: On XPU, when the hybrid block_size exceeds 64 (the FA2 maximum), round up to the next power of 2 (528 → 1024). This uses Triton ATTN and pads mamba pages by ~95% (2.1MB → 4.1MB), which is acceptable given the small model size.

| block_size | Backend | Mamba padding | Notes |
|-----------|---------|---------------|-------|
| 64 | FA2 | N/A (doesn't fit) | **Incompatible**: mamba state 2.1MB > page 256KB |
| 528 | Triton | 0.8% | Original: minimal waste, non-power-of-2 |
| 1024 | Triton | 95.4% | **Current**: power-of-2, ~2MB extra per mamba page |

## Why FA2 Backend Cannot Work with Hybrid Models

The XPU FA2 kernel requires `block_size=64` for paged attention. FA2 also restricts hybrid models to `[16, 32, 64]` to avoid a NaN propagation bug (flash-attention#1974). Hybrid models like Qwen3.5-4B require `block_size>=528` to fit the 2.1MB GDN recurrent state into one unified page.

These constraints are fundamentally incompatible:
- FA2 max block_size for hybrid: 64 → page size: 64 × 4096 = 256KB
- Mamba state minimum: 2,096KB
- No block_size in FA2's supported range can contain the mamba state

The Triton attention backend supports arbitrary block sizes, making it the correct choice for hybrid models on XPU. With the power-of-2 fix, block_size=1024 provides clean alignment.

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

## Hybrid Page Block Size Derivation (block_size=528)

The block_size=528 comes from aligning the GDN recurrent state to fit within the unified paged KV cache. Both full attention layers and GDN linear attention layers share the same physical page — each page holds 528 KV token slots for attention AND one GDN state for linear attention.

### Per-token KV size (full attention layers)

```
head_dim          = 256    (explicit in config, NOT hidden_size/num_heads)
num_kv_heads      = 4      (GQA)
KV_per_token      = 2(K+V) × 4 × 256 × 2(bf16) = 4096 bytes
```

### GDN state size (linear attention layers)

```
conv_dim          = linear_key_head_dim(128) × linear_num_key_heads(16) × 2
                  + linear_value_head_dim(128) × linear_num_value_heads(32)
                  = 4096 + 4096 = 8192

conv_state_shape  = (conv_kernel_size-1, conv_dim) = (3, 8192)
conv_state_bytes  = 3 × 8192 × 2(bf16) = 49,152 bytes (48 KB)

ssm_state_shape   = (num_v_heads, head_v_dim, head_k_dim) = (32, 128, 128)
ssm_state_bytes   = 32 × 128 × 128 × 4(float32) = 2,097,152 bytes (2048 KB)
                    ^^^^ float32 because model config mamba_ssm_dtype=float32
                    (recurrent state accumulates products, needs higher precision)

mamba_page_total  = 48 + 2048 = 2096 KB
```

### Block size calculation

```
kernel_alignment  = 16     (Triton attention kernel requirement)
block_size        = 16 × ⌈2096 KB / (16 × 4 KB)⌉
                  = 16 × ⌈32.75⌉
                  = 16 × 33
                  = 528 tokens

attn_page_size    = 528 × 4096 = 2,162,688 bytes (2112 KB)
mamba_page_padded = 2112 KB (padded from 2096 KB, +0.76%)
```

The ssm_state in float32 dominates: 2048 KB out of 2096 KB total (97.7%).

## Files Modified

| File | Change |
|------|--------|
| `vllm/_xpu_ops.py` | K/V contiguity for hybrid models |
| `vllm/model_executor/layers/fla/ops/utils.py` | XPU device context in input_guard |
| `vllm/model_executor/layers/quantization/gptq.py` | PyTorch GPTQ dequant fallback + oneDNN W4A16 |
| `vllm/model_executor/models/config.py` | Power-of-2 hybrid block_size rounding on XPU |
| `vllm/model_executor/models/qwen3_next.py` | NaN state repair + PyTorch reference kernels |
| `vllm/platforms/xpu.py` | Allow block_size >= 64 for hybrid models |
| `vllm/v1/worker/utils.py` | int64 overflow fix for XPU data pointers |
| `test_qwen35_4b_xpu.py` | Test script for Qwen3.5-4B on XPU |
