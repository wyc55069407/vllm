# SDP Paged Kernel HD=256 Optimization on Intel Xe2 (BMG)

This document captures all performance optimization learnings for the paged Flash Attention
SDP prefill kernel (`sdp_paged_prefill_dpas`) with head_dim=256, targeting Intel Xe2
Battlemage (BMG) discrete GPU using SYCL ESIMD.

## Hardware Target

- **GPU**: Intel Battlemage (BMG) — Xe2 architecture, 20 XE cores
- **Peak BF16 XMX**: 96 TFLOPS (DPAS bf16×bf16→fp32)
- **DRAM bandwidth**: 520 GB/s
- **GRF**: doubleGRF mode (256 registers per thread)
- **Compiler**: Intel oneAPI DPC++ (icpx), `-O2 -ffast-math -doubleGRF`

## Kernel Overview

- **Function**: `sdp_paged_prefill_dpas<bool CAUSAL>()` in `sdp_paged.h`
- **Workgroup**: 32 threads (1 per hardware thread in an XE core)
- **Tile**: 128 Q tokens × full KV length per workgroup
- **Data type**: BF16 I/O, FP32 accumulation
- **KV cache layout**: `[2, num_blocks, block_size, num_kv_heads, head_dim]` — stride-aware
- **Block size**: 1024 tokens per KV cache block
- **GQA**: Supports grouped-query attention (e.g., 16 Q heads / 4 KV heads)

## Performance Summary

### Standalone Benchmark (16Q/4KV, HD=256, block_size=1024, bf16)

**Noncausal:**

| Size | Non-paged ref | Paged (optimized) | Gap |
|------|--------------|-------------------|-----|
| 1K   | 48.8 TFLOPS  | 44.4 TFLOPS       | -9.0% |
| 2K   | 54.8 TFLOPS  | 52.4 TFLOPS       | -4.4% |
| 4K   | 57.8 TFLOPS  | 53.1 TFLOPS       | -8.1% |
| 8K   | 59.7 TFLOPS  | 51.7 TFLOPS       | -13.4% |

**Causal (q_len == kv_len):**

| Size | TFLOPS | Time vs Noncausal |
|------|--------|-------------------|
| 1K   | 34.1   | 0.252ms vs 0.388ms (1.54× faster) |
| 2K   | 43.3   | 0.795ms vs 1.314ms (1.65× faster) |
| 4K   | 49.8   | 2.765ms vs 5.182ms (1.87× faster) |
| 8K   | 51.2   | 10.754ms vs 21.288ms (1.98× faster) |

Causal time approaches 2× speedup at large sizes (triangular mask skips ~half the work).

### vLLM End-to-End (Qwen3.5-4B, ESIMD attention backend)

| Size  | ESIMD paged | FA2 paged | ESIMD vs FA2 |
|-------|-------------|-----------|--------------|
| 1024  | 36.2T       | 27.7T     | **+31%**     |
| 4096  | 47.2T       | 38.5T     | **+23%**     |
| 8192  | 46.4T       | 41.5T     | **+12%**     |

ESIMD paged beats Triton FA2 paged at all sizes (12-48%).

## Optimization Journey (v1 → v9, 2× speedup)

### What Worked (in order of impact)

#### 1. VS Loop Unrolling + Hoisted V Page Translation
Replaced `for` loop over 8 KV blocks with 8 manual `VS_LOAD_AND_DPAS` macro invocations.
V surface descriptor configured once per outer iteration (block_size=1024 >= KV_CHUNK=128).
**Impact**: ~15% improvement. Macros generate better ISA than `#pragma unroll for`.

#### 2. K[0] Full Prefetch in Prologue
Before the main QK sequential loop, prefetch all 16 D-blocks of K[0]. Without this, only
D-block 0 is warm; blocks 1-15 load cold from DRAM.
**Impact**: ~10% at large sizes.

#### 3. V[0] Prefetch in Prologue
Interleave V[0] prefetches during the QK[0] D-block computation loop (matching the
non-paged reference kernel pattern).
**Impact**: ~5% at large sizes.

#### 4. V Prefetch Timing Fix
Changed from prefetching V for the CURRENT KV iteration to the NEXT iteration during the
QK[k+1] overlap region. This gives one full iteration of lead time (softmax + scatter +
barrier) as prefetch window.
**Impact**: ~5%.

#### 5. K Prefetch During VS Accumulation
Insert 2 K prefetches per VS KV-block iteration × 8 blocks = 16 = all D-blocks of
K[outerIter+2]. Hides K load latency two iterations ahead.
**Impact**: ~3%.

#### 6. `template<bool CAUSAL>` + `if constexpr`
Compile-time branch elimination. Noncausal path avoids 32 registers for causal bound
vectors. Two separate binary instantiations.
**Impact**: ~3% for noncausal.

#### 7. Per-Iteration Block Table Lookup
Replaced cached `blk_table_cache[16]` array (16 registers wasted) with single
`block_table_row[logical_block]` load per outer iteration. With block_size=1024 and
KV_CHUNK=128, block table changes only every 8th inner iteration.
**Impact**: 16 regs freed, ~1%.

#### 8. D-Loop Peeling (KEY technique)
Split the D-dimension loop into: main loop `d=0..PF_HD_BLKS-2` with unconditional next-K
prefetch/load, plus a separate last D-block without. This eliminates the `if (d < HD_BLKS-1)`
branch from the hot loop.
**Impact**: +6% at 4K vs branched loop. Proven consistently better.

#### 9. Half-Rate V Prefetch (KEY discovery)
Prefetch V data every OTHER D-block instead of every D-block:
```cpp
if ((d & 1) == 0) {
    lsc_prefetch_2d<uint32_t, 8, 16, 1, ...>(payloadVpf);
}
```
Full-rate V prefetch overwhelms memory subsystem at small/medium sequence lengths. Half-rate
reduces memory pressure while keeping V warm enough for subsequent VS accumulation.

**V Prefetch Rate Comparison:**

| Size | No V prefetch | Half-rate | Full-rate |
|------|--------------|-----------|-----------|
| 1K   | 47.4T        | **44.4T** | 43.6T     |
| 2K   | **56.3T**    | 52.4T     | 51.5T     |
| 4K   | 49.5T        | **53.1T** | 51.5T     |
| 8K   | 46.0T        | **51.7T** | 49.1T     |

No-prefetch wins at 2K but collapses at 4K+. Full-rate loses everywhere vs half-rate.
Half-rate is the best compromise across all sizes.

### What Didn't Work

| Attempt | Result | Reason |
|---------|--------|--------|
| K prefetch before barrier B wait | -2% worse | Moving K prefetch earlier disrupts the proven VS+K_prefetch interleaving |
| VS `#pragma unroll for` loop | -6% worse | Compiler generates inferior ISA compared to manual macro expansion |
| `constexpr block_size` | 0% change | Compiler already constant-folds the runtime value |
| `-O3` vs `-O2` | 0% change | No benefit, `-O2` is sufficient |
| K/V allocation padding (cache aliasing) | 0% change | No L3 aliasing issue detected |
| FP16 K prefetch format | -5% worse | Format conversion overhead exceeds prefetch benefit |

## Remaining Paging Overhead (4-13% vs non-paged)

The gap between paged and non-paged kernels comes from:

1. **Block table lookup + address computation**: ~1-2% per outer KV iteration. Each
   iteration must load a block table entry, compute physical address from logical position.
2. **Larger binary size**: Paged kernel is 341KB vs 280KB for non-paged reference.
   Possible instruction cache pressure at scale.
3. **Inherent paged indirection**: Cannot be eliminated. Minimized by using large
   block_size (1024) so each block table lookup amortizes over many tokens.

## Architecture Design Decisions

### S^T (Transposed Scores) Layout
Scores matrix stored transposed in SLM: `S[kv_pos][q_tile]` instead of `S[q_tile][kv_pos]`.
This enables scatter-free `lsc_slm_scatter` writes from QK DPAS output and coalesced reads
for VS DPAS input. Eliminates the expensive SLM transpose step that costs ~5% in naive
implementations.

### Named Barrier Pipelining
Split-barrier pattern separates `nbarrier_arrive()` from `nbarrier_wait()`:
- After SLM scatter of QK scores: `arrive` on barrier B
- While barrier B propagates: compute softmax, setup V surface descriptors
- Before VS accumulation reads SLM: `wait` on barrier B

This hides barrier latency behind useful compute (~100+ cycles).

### 2D Surface Load/Store for K and V
Uses `lsc_load_2d` / `lsc_prefetch_2d` with `config_2d_mem_access` surface descriptors.
Enables hardware 2D blocking — each thread loads its portion of the K/V tile without
manual offset arithmetic. Critical for DPAS operand preparation.

### Interleaved KV Cache Layout Handling (Critical Bug Fix)
vLLM allocates KV cache as physical `[num_blocks, 2, block_size, nkvh, hd]` but presents
the logical shape `[2, num_blocks, block_size, nkvh, hd]` via `permute`. This means:
- `stride(0) = block_size * nkvh * hd` (K→V offset within same physical block)
- `stride(1) = 2 * block_size * nkvh * hd` (block stride spans K+V)
- Physical rows per block = `stride(1)/stride(2) = 2 * block_size`, NOT `block_size`

The kernel must use `phys_block_shift = __builtin_ctz(stride(1)/stride(2))` for 2D surface
Y coordinate computation, NOT `block_size_shift`. Using `block_size_shift` gives wrong
addresses for interleaved layouts and produces garbage output on multi-token prefill.

**Symptom**: 1-token decode works (uses scalar path, not 2D surface), but multi-token
prefill produces "!!!" garbage. Standalone tests with contiguous KV cache pass because
`phys_block_shift == block_size_shift` for contiguous layout.

### Causal Mask Implementation
`template<bool CAUSAL>` with `if constexpr` for zero-cost abstraction:
- Causal path: computes `causal_bound` per Q-row, uses `merge()` to mask scores to `-inf`
- Noncausal path: no mask computation, no extra registers
- Early-exit: entire KV chunks beyond causal boundary are skipped

## Build Instructions

```bash
# Standalone benchmark
source ~/intel/oneapi/setvars.sh --force
cd vllm-kernel-custom
icpx tests/bench_paged_standalone.cpp -o bench_paged_standalone \
  -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" \
  -I csrc/xpu/esimd_kernels -I csrc -ffast-math -std=c++17 -O2 \
  2>&1 | grep -i spill   # Must show NO spill

# PyTorch extension (vllm-kernel-custom)
source ~/intel/oneapi/setvars.sh --force
export TORCH_XPU_ARCH_LIST=bmg-g21
python setup_sycl.py clean && python setup_sycl.py install
```

**Critical**: doubleGRF flag (`-doubleGRF` or `-options -doubleGRF`) must be present in
BOTH compile AND device-link (`-fsycl-link`) steps. Missing it from dlink causes 9×
slowdown due to 10KB register spill.

## File Locations

- Kernel: `vllm-kernel-custom/csrc/xpu/esimd_kernels/sdp_paged.h`
- ESIMD attention backend: `vllm/v1/attention/backends/esimd_attn.py`
- Standalone benchmark: `vllm-kernel-custom/tests/bench_paged_standalone.cpp`
- Reference (non-paged) benchmark: `vllm-kernel-custom/tests/bench_ref_standalone.cpp`
- Non-paged reference kernel: `SDP_noncausal_ref_256d/rev256_onednn_v2_88tflops_s_scatter.h`
