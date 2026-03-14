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

### 2. GPTQ Non-MoE Linear Layers — oneDNN W4A16 INT4 GEMM

**Problem**: GPTQ CUDA kernels (`ops.gptq_gemm`) unavailable on XPU. The `int4_gemm_w4a16` op in `_xpu_C` is also not present in the current `vllm_xpu_kernels` package. The initial PyTorch dequant+`F.linear` fallback works but has overhead from separate dequantization.

**Fix (v2 — oneDNN native)**: Integrated `vllm-kernel-custom`'s `onednn_w4a16_int4` kernel for fused INT4 dequant+GEMM. This uses oneDNN's matmul primitive with `u4` weight type, group-wise K-scaled per-N scales, and optional zero points.

**Weight format conversion** (`_convert_to_onednn_u4` in `gptq.py`):
1. Unpack GPTQ `qweight` [K/8, N] int32 → [K, N] uint4 values
2. Transpose to [N, K] and repack as oneDNN u4: [N, K/2] uint8 (2 values per byte, lower nibble first)
3. Unpack `qzeros` [num_groups, N/8] int32 → [num_groups, N] uint4 values
4. For GPTQ v1: add 1 to zero points (v1 offset convention)
5. Repack zero points as oneDNN u4: [num_groups, N/2] uint8
6. Convert scales [num_groups, N] fp16 → fp32

**oneDNN primitive setup** (in `vllm-kernel-custom/csrc/xpu/onednn_fp8.sycl`):
- Weight desc: `{K, N} u4 format_tag::ba` (physical [N,K])
- Scales: `set_scales(DNNL_ARG_WEIGHTS, mask=3, groups={group_size, 1}, f32)` — K-grouped + per-N
- Zero points: `set_zero_points(DNNL_ARG_WEIGHTS, mask=3, groups={group_size, 1}, u4)`
- Supports FP16, BF16, and FP32 activations

**Limitations**: Only works when `desc_act=false` (sequential K-groups). Falls back to dequant+`F.linear` when `desc_act=true` or `vllm-kernel-custom` is not installed.

**Previous fix (v1)**: PyTorch dequantization fallback — unpacks INT4 weights, applies group-wise scaling, then uses `F.linear` (→ oneMKL). Still available as fallback.

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
| **Non-MoE Linear (GPTQ)** | oneDNN W4A16 INT4 GEMM (`vllm-kernel-custom`) | Native fused dequant+GEMM |
| **MoE Expert GEMM (W4A16)** | Triton `fused_moe_kernel_gptq_awq` | Functional, default config |
| **Attention** | XPU FA2 (`flash_attn_varlen_func`) | Optimized |
| **RMSNorm, Embedding, etc.** | PyTorch native | Standard |

### oneDNN W4A16 INT4 GEMM Details

The `onednn_w4a16_int4` kernel in `vllm-kernel-custom` provides native INT4 GEMM:
- Uses oneDNN matmul primitive with `u4` data type (available in oneDNN 3.7+ / oneAPI 2025.3)
- Supports group-wise quantization: K-grouped + per-N scales via `set_scales(mask=3, groups={group_size, 1})`
- Supports optional zero points via `set_zero_points(mask=3, groups={group_size, 1}, u4)`
- Weight stored in physical [N,K] layout, 2 u4 values per byte
- Dispatches FP16, BF16, FP32 activations

### `vllm-kernel-custom` Dependency

The oneDNN W4A16 path requires `vllm-kernel-custom` to be installed. If not available, falls back to PyTorch dequant+`F.linear`. Build instructions:
```bash
cd ~/yuchen/vllm_env/vllm-kernel-custom
source ~/intel/oneapi/setvars.sh --force
export TORCH_XPU_ARCH_LIST=bmg-g21
python setup_sycl.py clean && python setup_sycl.py install
```

### Performance Optimization Path (Future)

1. **Done**: oneDNN W4A16 INT4 GEMM for non-MoE GPTQ linear layers (fused dequant+GEMM)
2. **Done**: Triton MoE W4A16 kernel works for expert GEMM
3. **Future**: oneDNN-based W4A16 MoE kernel in `vllm-kernel-custom` for potentially better MoE performance

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

## Files Modified

### From Qwen3.5-4B fixes (reused):

| File | Relevant Fix |
|------|-------------|
| `vllm/model_executor/layers/quantization/gptq.py` | PyTorch GPTQ dequant fallback + oneDNN W4A16 INT4 GEMM |
| `vllm/v1/worker/utils.py` | int64 overflow fix for XPU data pointers |
| `vllm/_xpu_ops.py` | K/V contiguity (not needed for this model but harmless) |

### oneDNN W4A16 integration (new):

| File | Change |
|------|--------|
| `vllm/model_executor/layers/quantization/gptq.py` | Added `_convert_to_onednn_u4()` for GPTQ→oneDNN weight conversion; updated `apply()` to call `onednn_w4a16_int4` kernel |

### External dependency (`vllm-kernel-custom`):

| File | Change |
|------|--------|
| `csrc/xpu/onednn_fp8.sycl` | Added `onednn_w4a16_int4_impl` template + `onednn_w4a16_int4` wrapper |
| `include/vllm_kernel_ops.h` | Added `onednn_w4a16_int4` declaration |
| `csrc/xpu/torch_extension_sycl.cc` | Registered `onednn_w4a16_int4` torch op |
| `python/vllm_kernel_custom/esimd_ops.py` | Added `onednn_w4a16_int4` Python wrapper |
| `python/vllm_kernel_custom/__init__.py` | Added `onednn_w4a16_int4` to exports |

### New files:

| File | Purpose |
|------|---------|
| `test_qwen3_30b_gptq_xpu.py` | Test script for Qwen3-30B-A3B-GPTQ-Int4 on XPU |
