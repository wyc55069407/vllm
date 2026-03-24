# InfLLMv2 Sparse Attention on Intel BMG XPU

## Overview

InfLLMv2 block-sparse attention for MiniCPM4-8B on Intel Battlemage (BMG) discrete GPU,
integrated into vLLM V1 with ESIMD kernels.

**Key benefit**: Enables long-context inference (32K+) with sub-linear attention cost
by selecting only the top-K most relevant KV blocks per query position.

## Architecture

```
Prefill (seq_len > 8192):
  K_all → k_pooling → pooled_k [nkvh, num_blocks, hd]
  Q_chunk + pooled_k → qk_gemm → max_pooling → topk → topk_idx [nkvh, q_len, 64]
  topk_idx → mask_convert → union_mask [nkvh, q_len/16, 1024]
  Q_chunk + paged_kv + union_mask → sparse paged prefill SDP → output

Decode (seq_len > 8192):
  K_new → paged KV cache + incremental compression buffer
  Q + pooled_k → qk_gemm_decode → softmax → pooling → max_pool → topk
  Q + paged_kv + topk_mask → sparse paged decode SDP → output
```

## Configuration

- **Model**: MiniCPM4-8B (nh=32, nkvh=2, hd=128, GQA ratio=16)
- **Backend**: `INFLLMV2_ESIMD_ATTN`
- **Block size**: 128 (vLLM page size)
- **Sparse block**: 64 tokens
- **TopK**: 64 blocks selected per KV head
- **Dense threshold**: seq_len <= 8192 uses dense attention
- **Init blocks**: 2 (always attend to first 128 tokens)
- **Local blocks**: 4 (always attend to last 256 tokens)

## ESIMD Kernels

### Sparse Prefill — Fast GQA-Grouped (`sdp_paged_sparse_gqa.h`)

- **Dispatch**: 2D `{q_blocks, nkvh*16}` — each WG handles 1 q_block x 1 KV head x 16 Q heads
- **DPAS Usage 2 (Swapped)**: Q→b_tile (VNNI), K→a_tile — 16x less memory traffic than per-Q-head
- **Output layout**: Transposed (`acc[n*16+m] = C[m,n]`), written via `lsc_scatter` SOA
- **Normalization**: Per-lane vector divisor (not scalar) due to transposed DPAS layout
- **Memory**: Reads each KV block once for all 16 Q heads in the GQA group

### Sparse Decode — Two-Phase (`sdp_paged_sparse.h`)

- **Phase 1**: Mask-directed paged blocks, HD=128, QHT=4, SP_BLK=64, CHUNK=256
- **Phase 2**: Reduction across chunks (reuses dense decode phase2)
- **Last-block fix**: Force-inserts current token's sparse block into topk mask

### Pattern Detection (`infllmv2_pattern.h`)

- `k_pooling<2, 128, 32, 16>` — sliding window mean pooling
- `qk_gemm<32, 2, 128, 32, 16>` — prefill block scoring
- `qk_gemm_decode<32, 2, 128, 16>` — decode block scoring
- `softmax_decode<32, 256>` — decode softmax
- `pooling_decode<32, 2>` — GQA head averaging
- `qk_max_pooling<2, 64, 16>` — block-level max pooling
- `topk_indices_gpu<2, 64>` — top-K selection

### Mask Convert (`mask_convert.h`)

Per-token topk indices → per-q-block (16 positions) union mask. Pure index arithmetic.

## Usage

```bash
# Sparse attention (default)
python test_minicpm4_8b_hp_travel_25k.py --sparse

# Dense baseline
python test_minicpm4_8b_hp_travel_25k.py --dense

# Environment controls
INFLLMV2_SPARSE_PREFILL=0  # disable sparse prefill (use dense)
INFLLMV2_SPARSE_DECODE=0   # disable sparse decode
INFLLMV2_DEBUG_TRACE=1     # enable debug tracing
```

## Key Bug Fixes

1. **Normalization bug (GQA kernel)**: DPAS Usage 2 transposed layout means each 32-element
   output chunk contains interleaved data for all 16 heads. Must use per-lane vector divisor
   (`softMaxDivisor[lane]`), not scalar (`softMaxDivisor[head_offset]`).
   Symptom: correct at small data scale (~0.1), diverges at real model scale (~2.6).

2. **Last-block decode bug**: topk operates on pooled blocks (0..num_pooled-1), but the
   current decode token's sparse block can exceed num_pooled. Fix: force-insert
   `(seq_len-1)//64` into topk mask after selection.

3. **KV cache stride**: vLLM physical layout `[num_blocks, 2, bs, nkvh, hd]` vs logical
   `[2, num_blocks, bs, nkvh, hd]`. 2D surface Y coords must use physical block shift.

## Test Scripts

- `test_sparse_sdp_ult.py` — Standalone ULT: fast GQA kernel vs old kernel vs PyTorch GPU reference
- `test_minicpm4_8b_hp_sparse.py` — E2E 15K sparse/dense comparison
- `test_minicpm4_8b_hp_travel_25k.py` — E2E 25K two-topic quality test

## Known Limitations

- **K extraction bottleneck**: Python loop copies paged KV cache to contiguous buffer for pooling.
  Need: direct paged K pooling kernel.
- **Single-batch prefill**: Sparse prefill currently supports batch=1 only.
- **Fixed model config**: Hardcoded for nkvh=2, hd=128, GQA ratio=16.
