# XPU Paged SDP HD=128 Optimization Notes

ESIMD paged SDP attention kernel for HD=128 on Intel Xe2 (Battlemage/BMG).
Targets Qwen3-30B-A3B configuration: 32 query heads, 4 KV heads, GQA ratio 8.

## Architecture

- **Workgroup**: 16 threads (subgroups), doubleGRF (512 bytes/thread)
- **Q tile**: 256 Q rows per workgroup (16 threads x 16 rows each)
- **KV chunk**: 64 KV positions per iteration
- **No D-loop**: HD=128 fits in registers without head-dimension tiling
- **Online softmax**: single-pass with running max/sum correction

## Kernel Design: `sdp_paged_prefill_dpas_128<CAUSAL, IS_BF16>`

### Data Flow

1. **Q Load**: 2D surface load (lsc_load_2d), VNNI-packed, stored in `bf16QState`
2. **K Load**: 2D surface load, paged via block_table + phys_block_shift
3. **QK DPAS**: `dpas<8,8,float,float,half_t,half_t>` — bf16 or fp16 depending on IS_BF16
4. **Softmax**: Online softmax with causal masking, scores narrowed to half precision
5. **V Load**: Prefetched into tempBuffer, bf16→fp16 conversion when IS_BF16=true (skipped for fp16)
6. **SxV DPAS**: `dpas<8,8,fp16,fp16,fp16,fp16>` — always fp16 (V is fp16 in both modes)
7. **Output**: Accumulated in fp16, final softmax normalization, scatter to output

### bf16 vs fp16 Strategy

When IS_BF16=true (bf16 data):
- QK DPAS uses bf16 operands (native bf16 dot product)
- V data is bf16 → converted to fp16 for SxV DPAS (interleaved with QK DPAS to hide latency)
- Output narrowed from fp32→bf16

When IS_BF16=false (fp16 data):
- QK DPAS uses fp16 operands
- V data is already fp16 → no conversion needed
- Output narrowed from fp32→fp16

### Paged KV Cache Access

- Block table: `block_table[batch_idx, block_idx]` → physical block number
- 2D surface descriptors: stride-aware, handles interleaved per-block layout
- `phys_block_shift = ctz(stride(1)/stride(2))` for correct Y coordinate mapping
- KV cache shape: `(2, num_blocks, block_size, num_kv_heads, head_size)` logical

### Causal Masking

- Pre-computed boundaries per Q row based on `query_start_loc` and `seq_lens`
- Vectorized merge: scores beyond causal boundary set to -inf before softmax
- No extra cost for non-causal mode (boundary check skipped)

## Performance Results

Tested on Intel Battlemage (BMG) 24GB, 32Q/4KV heads.

### HD=128 Prefill (TFLOPS)

| SeqLen | ESIMD bf16 | ESIMD fp16 | torch SDPA bf16 | Speedup |
|--------|-----------|-----------|-----------------|---------|
| 1K NC  | 59.0      | 57.9      | 50.6            | 1.17x   |
| 2K NC  | 71.1      | 69.9      | 54.8            | 1.30x   |
| 4K NC  | 81.2      | 80.4      | 57.0            | 1.42x   |
| 1K C   | 41.4      | 39.1      | 35.8            | 1.16x   |
| 2K C   | 55.9      | 52.5      | 43.8            | 1.28x   |
| 4K C   | 66.8      | 63.2      | 49.1            | 1.36x   |

NC = noncausal, C = causal. fp16 within 5% of bf16.

### HD=256 Prefill (TFLOPS)

| SeqLen | ESIMD bf16 | ESIMD fp16 | torch SDPA bf16 |
|--------|-----------|-----------|-----------------|
| 1K NC  | 45.9      | 45.9      | 52.0            |
| 2K NC  | 53.8      | 53.9      | 60.3            |
| 4K NC  | 57.3      | 57.3      | 62.5            |
| 4K C   | 51.4      | 51.5      | 56.5            |

HD=256 fp16 identical to bf16.

## Requirements

- `block_size >= 64` (KV chunk size). Recommended: `block_size=128`
- ESIMD ATTN backend: `attention_backend="ESIMD_ATTN"`
- doubleGRF build (lgrf extension)
- Supported dtypes: bf16, fp16
- Supported head sizes: 128, 256

## Files

- `vllm-kernel-custom/csrc/xpu/esimd_kernels/sdp_paged.h` — All kernel implementations
- `vllm-kernel-custom/csrc/xpu/uni_esimd_kernel_lgrf.sycl` — Dispatch and launch
- `vllm/v1/attention/backends/esimd_attn.py` — Python backend integration
