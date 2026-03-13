# Qwen3-30B-A3B-GPTQ-Int4 on Intel XPU (BMG) — Fixes and Learnings

## Overview

Qwen3-30B-A3B is a Mixture-of-Experts (MoE) model with 128 experts (8 active per token), 48 layers, and GPTQ 4-bit quantization. Running it on Intel BMG (24GB VRAM) required understanding the GPTQ/MoE dispatch chain and fitting the 16GB quantized model into memory.

### Model Specs
- Architecture: `Qwen3MoeForCausalLM`
- Hidden size: 2048, head_dim: 128, 32 attention heads, 4 KV heads
- 128 experts, 8 active per token, MoE intermediate size: 768
- GPTQ 4-bit, group_size=128, symmetric, no desc_act
- Attention: Standard full attention (no GDN/linear attention) → FA2 compatible

## Fixes Applied

### 1. Memory Management (OOM)

**Problem**: Model weights are ~16GB (GPTQ-Int4 packed). With 24GB total VRAM, vLLM's default KV cache profiler over-allocates, causing `UR_RESULT_ERROR_DEVICE_LOST` (XPU OOM).

**Fix**: Set `gpu_memory_utilization=0.30` and `max_model_len=64`. The profiler reports available memory before model weights are fully resident, so conservative settings are needed.

**Math**: 22.7GB total - 15.6GB model = ~7GB free. 48 layers × 136MB/layer KV cache = 6.5GB at 0.30 utilization.

### 2. GPTQ Non-MoE Linear Layers (from Qwen3.5-4B fixes)

**Problem**: GPTQ CUDA kernels (`ops.gptq_gemm`) unavailable on XPU. The `int4_gemm_w4a16` op in `_xpu_C` is also not present in the current `vllm_xpu_kernels` package.

**Fix**: PyTorch dequantization fallback in `GPTQLinearMethod.apply()` — unpacks INT4 weights, applies group-wise scaling, then uses `F.linear` (→ oneMKL GEMM). This was already implemented for Qwen3.5-4B.

### 3. MoE W4A16 Expert GEMM

**Status**: Works out of the box via Triton kernel.

**Dispatch chain**:
1. `GPTQConfig.get_quant_method()` detects `FusedMoE` layer → delegates to `MoeWNA16Config`
2. `MoeWNA16Method.apply()` calls `fused_experts()` with `quant_config=int4_w4a16_moe_quant_config`
3. `fused_experts_impl()` → `dispatch_fused_moe_kernel()` with `use_int4_w4a16=True`
4. Routes to `invoke_fused_moe_wna16_triton_kernel()` → `fused_moe_kernel_gptq_awq` Triton kernel
5. This Triton kernel handles int4 weight dequantization + GEMM inline and **works on XPU via triton-xpu**

### 4. Attention Backend

**Status**: FA2 works directly. No hybrid block_size issue (unlike Qwen3.5-4B).

This model uses standard full attention across all 48 layers with `block_size=64`, which is compatible with the XPU FA2 kernel.

## Kernel and Backend Analysis

### Current XPU Kernel Support

| Component | Kernel Used | Performance Level |
|-----------|-------------|-------------------|
| **Non-MoE Linear (GPTQ)** | PyTorch dequant + `F.linear` (oneMKL) | Functional, not optimal |
| **MoE Expert GEMM (W4A16)** | Triton `fused_moe_kernel_gptq_awq` | Functional, default config |
| **Attention** | XPU FA2 (`flash_attn_varlen_func`) | Optimized |
| **RMSNorm, Embedding, etc.** | PyTorch native | Standard |

### `vllm_xpu_kernels` Package Gaps

The installed `vllm_xpu_kernels` only provides `flash_attn_varlen_func`. Missing ops:
- `int4_gemm_w4a16` — Would accelerate GPTQ non-MoE linear layers
- `fp8_gemm_w8a16` — Would enable FP8 quantization path
- No dedicated W4A16 MoE kernel (Triton fallback works)

### oneDNN / oneMKL W4A16 Support

- **GEMM (W4A16)**: `F.linear` dispatches to oneMKL for the FP16 matmul after dequantization. oneMKL handles the compute efficiently but the dequant step is separate overhead.
- **MoE (W4A8)**: Not directly supported. The Triton MoE kernel handles W4A16 (weight int4, activation FP16). True W4A8 would require a custom kernel.
- oneDNN primitive-level INT4 GEMM support exists in newer versions but is not exposed through PyTorch's XPU backend yet.

### Performance Optimization Path (Future)

1. **Short term**: The current Triton MoE kernel + PyTorch dequant path works correctly
2. **Medium term**: If `vllm_xpu_kernels` adds `int4_gemm_w4a16`, route GPTQ linear layers through `XPUwNa16LinearKernel` for fused dequant+GEMM
3. **Long term**: Implement ESIMD/SYCL W4A16 MoE kernel in `vllm-kernel-custom` using oneDNN matmul primitives with int4 source type (available in oneDNN 3.7+)

## How to Run

```bash
conda activate vllm_xpu
source ~/intel/oneapi/setvars.sh --force
cd ~/yuchen/vllm_env/vllm

python test_qwen3_30b_gptq_xpu.py
```

### Key Parameters

```python
LLM(
    model="/path/to/Qwen3-30B-A3B-GPTQ-Int4",
    dtype="float16",             # Model quantized in FP16
    enforce_eager=True,          # No torch.compile on XPU yet
    gpu_memory_utilization=0.30, # Critical: 16GB model in 24GB VRAM
    max_model_len=64,            # Minimal context to fit in remaining memory
)
```

Notes:
- No `attention_backend` override needed — FA2 works natively
- No chunked prefill issues — standard attention, no GDN/linear attention
- Loading takes ~4 minutes (16GB safetensors over IO)

## Files Modified (Reused from Qwen3.5-4B)

All code fixes for this model were already implemented in the `xpu-qwen35-4b-fixes` branch:

| File | Relevant Fix |
|------|-------------|
| `vllm/model_executor/layers/quantization/gptq.py` | PyTorch GPTQ dequant fallback for non-MoE linear layers |
| `vllm/v1/worker/utils.py` | int64 overflow fix for XPU data pointers |
| `vllm/_xpu_ops.py` | K/V contiguity (not needed for this model but harmless) |

New file:
| File | Purpose |
|------|---------|
| `test_qwen3_30b_gptq_xpu.py` | Test script for Qwen3-30B-A3B-GPTQ-Int4 on XPU |
