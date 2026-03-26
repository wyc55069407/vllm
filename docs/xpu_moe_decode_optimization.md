# ESIMD MoE Decode Kernel Optimization (MiniCPM5)

## Overview

Fused ESIMD MoE decode kernel for W4A16 GPTQ INT4 symmetric quantization on Intel Xe2 (BMG).
Replaces the generic Triton-based FusedMoE during decode (M ≤ 8 tokens) with a hand-tuned
3-phase ESIMD pipeline: **up_gate_silu → down → gather**.

Target model: MiniCPM5 MoE (E=160, K=2048, N=512, topk=16, GS=128).

## Performance Results

### Decode Step Speedup (batch=1, short context)

| Component  | Baseline (Triton MoE) | ESIMD MoE | Speedup |
|------------|----------------------|-----------|---------|
| **Total decode step** | **52.0 ms** | **29.7 ms** | **1.75x** |
| Attention  | 17.3 ms              | 16.2 ms   | ~same   |
| **MoE**    | **31.9 ms**          | **10.7 ms** | **2.97x** |
| Other      | 2.8 ms               | 2.7 ms    | ~same   |

MoE share of decode time: 61.3% → 36.2%.

### Standalone Kernel Bandwidth (pipeline throughput, no host overhead)

| M (batch) | Time (μs) | BW (GB/s) | % of 450 peak |
|-----------|-----------|-----------|----------------|
| 1         | 88        | 296       | 65.7%          |
| 2         | 160       | 326       | 72.4%          |
| 4         | 292       | 357       | 79.3%          |
| 8         | 564       | 370       | 82.1%          |

## Architecture

### 3-Phase Pipeline

1. **up_gate_silu** (`moe_up_gate_silu_rows`): Fused gate + up projection + SiLU activation
   - ROWS_UP=4 rows/thread, sharing input vector load across rows
   - Range: `(M, topk, N/ROWS_UP)` — 1 thread per WG

2. **down** (`moe_down_rows`): Down projection
   - ROWS_DN=8 rows/thread, sharing intermediate vector load
   - Range: `(M, topk, K/ROWS_DN)` — 1 thread per WG

3. **gather** (`moe_gather_kernel`): Weighted reduction across experts
   - Range: `(M, K)` — 1 thread per output element

### Key Optimizations

- **dequant_dot fusion**: INT4 dequant + dot product in one pass without materializing
  full VL-sized weight vector. Saves ~2KB register pressure per call.
  Processes GS-sized blocks with strided `select` on input.

- **ROWS batching**: Each thread processes multiple output rows sharing a single input
  vector load. Reduces L2 input traffic by ROWS× and thread count by ROWS×.

- **SIMD nibble interleaving**: `lo = p & 0x0F; hi = (p >> 4) & 0x0F` with strided
  input select `(base+0, stride=2)` and `(base+1, stride=2)` for even/odd K positions.

- **VL=512**: Processes 512 elements per inner loop iteration (4 groups of GS=128),
  providing 4 iterations for up kernel (K=2048) and 1 iteration for down (N=512).

### INT4 Symmetric Dequantization

```
weight_fp = (uint4_value - 8) * scale
```

Weight layout (matching vLLM GPTQ MoE format):
- `w13_qweight`: `[E, 2*N, K/2]` uint8 (2 u4 values per byte)
- `w13_scales`:  `[E, 2*N, K/GS]` bf16/fp16
- `w2_qweight`:  `[E, K, N/2]` uint8
- `w2_scales`:   `[E, K, N/GS]` bf16/fp16

## Integration

### Enable

```bash
MINICPM5_ESIMD_MOE=1 python your_script.py
```

### How It Works

In `MiniCPM5MoEMoE.forward()`, when `num_tokens ≤ 8` (decode):

1. **Routing** (pure PyTorch): gate → sigmoid → e_score_correction_bias → topk(16) → renormalize
2. **ESIMD kernel**: `esimd_moe_decode(x, w13_qw, w13_s, w2_qw, w2_s, topk_w, topk_ids, out, gs)`
3. **Shared expert**: `MiniCPM5MoEMLP(hidden_states)` called separately
4. **Scaling**: `routed_scaling_factor` applied to routed output
5. **Combine**: `output = routed_output + shared_output`

Prefill (num_tokens > 8) continues to use the standard SharedFusedMoE path.

### Requirements

- `vllm-kernel-custom` package with `common_ops` extension (standard GRF, no doubleGRF)
- Intel Xe2 BMG GPU
- bf16 or fp16 inference dtype
- GPTQ INT4 symmetric quantization (group_size 32/64/128)

## Files

| File | Description |
|------|-------------|
| `vllm-kernel-custom/csrc/xpu/esimd_kernels/moe_decode.h` | ESIMD kernel implementations |
| `vllm-kernel-custom/csrc/xpu/moe_decode.sycl` | SYCL dispatch + PyTorch binding |
| `vllm-kernel-custom/python/vllm_kernel_custom/esimd_ops.py` | Python wrapper |
| `vllm/model_executor/models/minicpm5_moe.py` | Model integration (`_esimd_forward`) |

## Optimization Attempts (Down Kernel)

The down kernel achieves 68-80% BW efficiency (vs 82% for up) due to having only 1 inner
loop iteration (N=512=VL_D) which limits load/compute overlap. Attempted optimizations:

| Attempt | Result |
|---------|--------|
| VL_D=256 (2 iterations) | 17-19% worse (smaller loads less efficient) |
| ROWS_DN=4 | Same M=1, 10% worse M=8 |
| ROWS_DN=16 | Register spill (128B private mem) |
| Fused down+gather (SLM) | 14% worse M=1 |
| Multi-thread WG (WG=4/8) | 8-51% worse (WG overhead > L1 sharing benefit) |

Conclusion: ROWS_UP=4, ROWS_DN=8, VL=512, 1 thread/WG is optimal for MiniCPM5 config.
