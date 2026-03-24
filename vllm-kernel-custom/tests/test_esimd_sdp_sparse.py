"""Unit test + performance benchmarks for ESIMD sparse paged SDP attention.

Tests both:
  - Sparse prefill (fast GQA-grouped kernel): correctness vs PyTorch reference + TFLOPS
  - Sparse decode (two-phase): correctness vs PyTorch reference + GB/s

Config: MiniCPM4-8B — nh=32, nkvh=2, hd=128, block_size=128, sparse_block=64, topk=64
"""
import torch
import math
import time
import argparse


# ---------- helpers ----------
def create_paged_kv_cache(seq_len, nkvh, hd, block_size, device, dtype, data_scale=1.0):
    """Create paged KV cache for a single sequence."""
    num_pages = (seq_len + block_size - 1) // block_size + 1  # +1 for safety
    kv_cache = (torch.randn(2, num_pages, block_size, nkvh, hd, dtype=dtype, device=device)
                * data_scale)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    return kv_cache, block_table, seq_lens


def build_sparse_mask_prefill(q_len, seq_len, nkvh, n_sparse_blocks, sparse_block=64):
    """Build sparse mask for prefill: each 16-token q_block selects same n_sparse_blocks.
    Selects first n_sparse_blocks (covers init + local-ish blocks).
    """
    q_blocks = (q_len + 15) // 16
    total_kv_blocks = (seq_len + sparse_block - 1) // sparse_block

    mask = torch.zeros(nkvh, q_blocks, 1024, dtype=torch.int32, device='xpu')
    mask_cnt = torch.zeros(nkvh, q_blocks, dtype=torch.int32, device='xpu')

    for kvh in range(nkvh):
        for qb in range(q_blocks):
            # Select n_sparse_blocks evenly spaced, plus local
            q_abs_block = (qb * 16 + (seq_len - q_len)) // sparse_block
            n = min(n_sparse_blocks, total_kv_blocks)
            blocks = set()
            # Init blocks
            blocks.update(range(min(2, total_kv_blocks)))
            # Local blocks (around current q position)
            for lb in range(max(0, q_abs_block - 4), min(total_kv_blocks, q_abs_block + 1)):
                blocks.add(lb)
            # Fill remaining from evenly spaced
            step = max(1, total_kv_blocks // (n - len(blocks) + 1))
            for i in range(0, total_kv_blocks, step):
                blocks.add(i)
                if len(blocks) >= n:
                    break
            # If still short, add sequential
            for i in range(total_kv_blocks):
                if len(blocks) >= n:
                    break
                blocks.add(i)

            sorted_blocks = sorted(blocks)[:n]
            mask_cnt[kvh, qb] = len(sorted_blocks)
            for i, b in enumerate(sorted_blocks):
                mask[kvh, qb, i] = b

    return mask, mask_cnt


def build_sparse_mask_decode(batch, nkvh, seq_lens_list, n_sparse_blocks=64, sparse_block=64):
    """Build sparse mask for decode: [batch, nkvh, 64] block indices."""
    mask = torch.zeros(batch, nkvh, n_sparse_blocks, dtype=torch.int32, device='xpu')
    for b in range(batch):
        sl = seq_lens_list[b]
        total_kv_blocks = (sl + sparse_block - 1) // sparse_block
        for kvh in range(nkvh):
            n = min(n_sparse_blocks, total_kv_blocks)
            # Select evenly spaced blocks
            blocks = set()
            blocks.update(range(min(2, total_kv_blocks)))  # init
            cur_blk = (sl - 1) // sparse_block
            blocks.update(range(max(0, cur_blk - 4), min(total_kv_blocks, cur_blk + 1)))  # local
            step = max(1, total_kv_blocks // (n + 1))
            for i in range(0, total_kv_blocks, step):
                blocks.add(i)
                if len(blocks) >= n:
                    break
            for i in range(total_kv_blocks):
                if len(blocks) >= n:
                    break
                blocks.add(i)
            sorted_blocks = sorted(blocks)[:n]
            # Force-include last block (the decode token's block)
            if cur_blk not in sorted_blocks and len(sorted_blocks) == n:
                sorted_blocks[-1] = cur_blk
                sorted_blocks.sort()
            for i, blk in enumerate(sorted_blocks):
                mask[b, kvh, i] = blk
    return mask


# ---------- PyTorch GPU reference ----------
def ref_sparse_prefill(query, kv_cache, block_table, seq_lens,
                       sparse_mask, sparse_mask_cnt,
                       nh, nkvh, hd, bs, scale, sparse_block=64):
    """PyTorch reference for sparse prefill."""
    q_len = query.shape[0]
    q_blocks = (q_len + 15) // 16
    group_size = nh // nkvh
    seq_len = seq_lens[0].item()
    history_len = seq_len - q_len

    q = query.float()
    k_cache, v_cache = kv_cache[0], kv_cache[1]
    bt = block_table[0]

    output = torch.zeros(q_len, nh, hd, dtype=torch.float32, device=query.device)

    for kvh in range(nkvh):
        for qb in range(q_blocks):
            q_start = qb * 16
            q_end = min(q_start + 16, q_len)
            n_blks = sparse_mask_cnt[kvh, qb].item()
            if n_blks == 0:
                continue

            # Gather KV tokens from sparse blocks
            tok_list = []
            for bi in range(n_blks):
                kv_blk = sparse_mask[kvh, qb, bi].item()
                for t in range(sparse_block):
                    tok = kv_blk * sparse_block + t
                    if tok < seq_len:
                        tok_list.append(tok)

            if not tok_list:
                continue
            nt = len(tok_list)
            K = torch.zeros(nt, hd, dtype=torch.float32, device=query.device)
            V = torch.zeros(nt, hd, dtype=torch.float32, device=query.device)
            for idx, tok in enumerate(tok_list):
                pg = tok // bs
                off = tok % bs
                pp = bt[pg].item()
                K[idx] = k_cache[pp, off, kvh].float()
                V[idx] = v_cache[pp, off, kvh].float()

            for qh_off in range(group_size):
                qh = kvh * group_size + qh_off
                q_slice = q[q_start:q_end, qh]
                scores = q_slice @ K.t() * scale

                # Causal mask
                for qi_local in range(q_end - q_start):
                    causal_bound = history_len + q_start + qi_local
                    for idx, tok in enumerate(tok_list):
                        if tok > causal_bound:
                            scores[qi_local, idx] = float('-inf')

                scores = scores - scores.max(dim=-1, keepdim=True).values
                scores = torch.exp(scores)
                scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-12)
                output[q_start:q_end, qh] = scores @ V

    return output


def ref_sparse_decode(query, kv_cache, block_table, seq_lens,
                      sparse_mask, nh, nkvh, hd, bs, scale,
                      n_sparse_blocks=64, sparse_block=64):
    """PyTorch reference for sparse decode (batch, q_len=1 per seq)."""
    batch = query.shape[0]
    group_size = nh // nkvh
    k_cache, v_cache = kv_cache[0], kv_cache[1]

    output = torch.zeros(batch, nh, hd, dtype=torch.float32, device=query.device)

    for b in range(batch):
        q = query[b].float()  # [nh, hd]
        sl = seq_lens[b].item()
        bt = block_table[b]

        for kvh in range(nkvh):
            tok_list = []
            for bi in range(n_sparse_blocks):
                kv_blk = sparse_mask[b, kvh, bi].item()
                for t in range(sparse_block):
                    tok = kv_blk * sparse_block + t
                    if tok < sl:
                        tok_list.append(tok)

            if not tok_list:
                continue
            nt = len(tok_list)
            K = torch.zeros(nt, hd, dtype=torch.float32, device=query.device)
            V = torch.zeros(nt, hd, dtype=torch.float32, device=query.device)
            for idx, tok in enumerate(tok_list):
                pg = tok // bs
                off = tok % bs
                pp = bt[pg].item()
                K[idx] = k_cache[pp, off, kvh].float()
                V[idx] = v_cache[pp, off, kvh].float()

            for qh_off in range(group_size):
                qh = kvh * group_size + qh_off
                scores = (q[qh] @ K.t()) * scale  # [nt]
                scores = scores - scores.max()
                scores = torch.exp(scores)
                scores = scores / (scores.sum() + 1e-12)
                output[b, qh] = scores @ V

    return output


# ---------- kernel wrappers ----------
def run_sparse_prefill(query, kv_cache, block_table, seq_lens,
                       sparse_mask, sparse_mask_cnt,
                       nh, nkvh, hd, bs, max_sl, scale, topk):
    from vllm_kernel_custom import esimd_sdp_paged_sparse
    q_len = query.shape[0]
    qsl = torch.tensor([0, q_len], dtype=torch.int32, device='xpu')
    out = torch.zeros(q_len, nh, hd, dtype=query.dtype, device='xpu')
    max_cnt = int(sparse_mask_cnt.max().item())
    esimd_sdp_paged_sparse(query, kv_cache, out, block_table, seq_lens, qsl,
                           sparse_mask.int(), sparse_mask_cnt.int(),
                           nh, nkvh, hd, bs, max_sl, scale, 0, max_cnt)
    torch.xpu.synchronize()
    return out


def run_sparse_decode(query, kv_cache, block_table, seq_lens,
                      sparse_mask, nh, nkvh, hd, bs, max_sl, scale, nsb=64):
    from vllm_kernel_custom import esimd_sdp_paged_sparse
    batch = query.shape[0]
    qsl = torch.arange(batch + 1, dtype=torch.int32, device='xpu')
    empty_cnt = torch.empty(0, dtype=torch.int32, device='xpu')
    out = torch.zeros(batch, nh, hd, dtype=query.dtype, device='xpu')
    esimd_sdp_paged_sparse(query, kv_cache, out, block_table, seq_lens, qsl,
                           sparse_mask.int(), empty_cnt,
                           nh, nkvh, hd, bs, max_sl, scale, 1, nsb)
    torch.xpu.synchronize()
    return out


# ---------- tests ----------
def test_sparse_prefill_correctness(q_len, seq_len, n_sparse_blocks, dtype=torch.bfloat16):
    """Test sparse prefill kernel vs PyTorch reference."""
    nh, nkvh, hd, bs = 32, 2, 128, 128
    scale = 1.0 / math.sqrt(hd)

    kv_cache, bt, sl = create_paged_kv_cache(seq_len, nkvh, hd, bs, 'xpu', dtype, data_scale=0.5)
    q = (torch.randn(q_len, nh, hd, dtype=dtype, device='xpu') * 0.5)
    mask, mask_cnt = build_sparse_mask_prefill(q_len, seq_len, nkvh, n_sparse_blocks)

    ref = ref_sparse_prefill(q, kv_cache, bt, sl, mask, mask_cnt, nh, nkvh, hd, bs, scale)
    out = run_sparse_prefill(q, kv_cache, bt, sl, mask, mask_cnt, nh, nkvh, hd, bs, seq_len, scale, n_sparse_blocks)

    diff = (ref - out.float()).abs()
    max_diff = diff.max().item()
    rms = (diff ** 2).mean().sqrt().item()
    nan_cnt = torch.isnan(out).sum().item()
    ref_rms = (ref ** 2).mean().sqrt().item()
    rel_rms = rms / (ref_rms + 1e-12)

    status = "PASS" if (max_diff < 0.5 and nan_cnt == 0 and rel_rms < 0.01) else "FAIL"
    print(f"  [{status}] prefill q={q_len} kv={seq_len} sparse_blks={n_sparse_blocks} "
          f"max_diff={max_diff:.4f} rel_rms={rel_rms:.6f} nan={nan_cnt}")
    return status == "PASS"


def test_sparse_decode_correctness(batch, seq_len, dtype=torch.bfloat16):
    """Test sparse decode kernel vs PyTorch reference."""
    nh, nkvh, hd, bs = 32, 2, 128, 128
    nsb = 64
    scale = 1.0 / math.sqrt(hd)

    # Create shared KV cache large enough for all sequences
    max_pages = (seq_len + bs - 1) // bs + 1
    total_pages = max_pages * batch + 10
    kv_cache = (torch.randn(2, total_pages, bs, nkvh, hd, dtype=dtype, device='xpu') * 0.5)

    # Each batch element gets different pages
    bt = torch.zeros(batch, max_pages, dtype=torch.int32, device='xpu')
    sl_list = []
    for b in range(batch):
        s = seq_len - b * 100  # vary seq_len slightly
        if s < 256:
            s = 256
        sl_list.append(s)
        for p in range(max_pages):
            bt[b, p] = b * max_pages + p

    sl = torch.tensor(sl_list, dtype=torch.int32, device='xpu')
    q = (torch.randn(batch, nh, hd, dtype=dtype, device='xpu') * 0.5)
    mask = build_sparse_mask_decode(batch, nkvh, sl_list, nsb)

    ref = ref_sparse_decode(q, kv_cache, bt, sl, mask, nh, nkvh, hd, bs, scale, nsb)
    out = run_sparse_decode(q, kv_cache, bt, sl, mask, nh, nkvh, hd, bs, max(sl_list), scale, nsb)

    diff = (ref - out.float()).abs()
    max_diff = diff.max().item()
    rms = (diff ** 2).mean().sqrt().item()
    nan_cnt = torch.isnan(out).sum().item()
    ref_rms = (ref ** 2).mean().sqrt().item()
    rel_rms = rms / (ref_rms + 1e-12)

    status = "PASS" if (max_diff < 0.5 and nan_cnt == 0 and rel_rms < 0.01) else "FAIL"
    print(f"  [{status}] decode batch={batch} kv={seq_len} nsb={nsb} "
          f"max_diff={max_diff:.4f} rel_rms={rel_rms:.6f} nan={nan_cnt}")
    return status == "PASS"


def bench_sparse_prefill(q_len, seq_len, n_sparse_blocks, dtype=torch.bfloat16, warmup=5, iters=20):
    """Benchmark sparse prefill TFLOPS."""
    nh, nkvh, hd, bs = 32, 2, 128, 128
    scale = 1.0 / math.sqrt(hd)
    sparse_block = 64

    kv_cache, bt, sl = create_paged_kv_cache(seq_len, nkvh, hd, bs, 'xpu', dtype, data_scale=0.5)
    q = (torch.randn(q_len, nh, hd, dtype=dtype, device='xpu') * 0.5)
    mask, mask_cnt = build_sparse_mask_prefill(q_len, seq_len, nkvh, n_sparse_blocks)

    # Warmup
    for _ in range(warmup):
        run_sparse_prefill(q, kv_cache, bt, sl, mask, mask_cnt, nh, nkvh, hd, bs, seq_len, scale, n_sparse_blocks)

    # Timed
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_sparse_prefill(q, kv_cache, bt, sl, mask, mask_cnt, nh, nkvh, hd, bs, seq_len, scale, n_sparse_blocks)
    torch.xpu.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    kv_attended = n_sparse_blocks * sparse_block
    flops = q_len * nh * kv_attended * hd * 4  # QK + SV, multiply-add=2
    tflops = flops / elapsed / 1e12

    print(f"  prefill q={q_len:5d} kv={seq_len:6d} blks={n_sparse_blocks:3d} "
          f"attended={kv_attended:5d}  {elapsed*1000:7.2f} ms  {tflops:5.1f} TFLOPS")
    return tflops


BMG_PEAK_BW_GBS = 450.0  # BMG HBM bandwidth

def bench_sparse_decode(batch, seq_len, dtype=torch.bfloat16, warmup=10, iters=50):
    """Benchmark sparse decode GB/s and BW%."""
    nh, nkvh, hd, bs = 32, 2, 128, 128
    nsb = 64
    scale = 1.0 / math.sqrt(hd)

    max_pages = (seq_len + bs - 1) // bs + 1
    total_pages = max_pages * batch + 10
    kv_cache = (torch.randn(2, total_pages, bs, nkvh, hd, dtype=dtype, device='xpu') * 0.5)

    bt = torch.zeros(batch, max_pages, dtype=torch.int32, device='xpu')
    sl_list = [seq_len] * batch
    for b in range(batch):
        for p in range(max_pages):
            bt[b, p] = b * max_pages + p
    sl = torch.tensor(sl_list, dtype=torch.int32, device='xpu')
    q = (torch.randn(batch, nh, hd, dtype=dtype, device='xpu') * 0.5)
    mask = build_sparse_mask_decode(batch, nkvh, sl_list, nsb)

    for _ in range(warmup):
        run_sparse_decode(q, kv_cache, bt, sl, mask, nh, nkvh, hd, bs, seq_len, scale, nsb)

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_sparse_decode(q, kv_cache, bt, sl, mask, nh, nkvh, hd, bs, seq_len, scale, nsb)
    torch.xpu.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    # Memory: read Q + read K/V for nsb*64 tokens + write output
    kv_bytes = batch * nkvh * nsb * 64 * hd * 2 * 2  # K+V, 2 bytes each
    q_bytes = batch * nh * hd * 2
    o_bytes = batch * nh * hd * 2
    total_bytes = kv_bytes + q_bytes + o_bytes
    gbps = total_bytes / elapsed / 1e9
    bw_pct = gbps / BMG_PEAK_BW_GBS * 100

    print(f"  decode batch={batch:2d} kv={seq_len:6d} nsb={nsb:3d}  "
          f"{elapsed*1000:7.3f} ms  {gbps:6.1f} GB/s  {bw_pct:5.1f}% BW")
    return gbps


# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sparse SDP correctness + perf test")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"

    all_pass = True

    if not args.skip_correctness:
        print(f"\n{'='*60}")
        print(f"  Sparse Prefill Correctness ({dtype_name})")
        print(f"{'='*60}")
        for q_len, seq_len, nsb in [
            (16, 1024, 16),
            (16, 4096, 64),
            (128, 4096, 64),
            (256, 8192, 64),
            (512, 16384, 64),
            (1024, 16384, 128),
        ]:
            ok = test_sparse_prefill_correctness(q_len, seq_len, nsb, dtype)
            all_pass &= ok

        print(f"\n{'='*60}")
        print(f"  Sparse Decode Correctness ({dtype_name})")
        print(f"{'='*60}")
        for batch, seq_len in [
            (1, 2048),
            (1, 8192),
            (4, 4096),
            (8, 8192),
            (1, 16384),
        ]:
            ok = test_sparse_decode_correctness(batch, seq_len, dtype)
            all_pass &= ok

        print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    if not args.skip_perf:
        print(f"\n{'='*60}")
        print(f"  Sparse Prefill Performance ({dtype_name})")
        print(f"  flops = q_len x nh(32) x kv_attended x hd(128) x 4")
        print(f"{'='*60}")
        for q_len, seq_len, nsb in [
            # Standard configs
            (4096, 16384, 32),
            (4096, 16384, 64),
            (4096, 16384, 128),
            (8192, 16384, 64),
            (16384, 16384, 64),
            # Long-context: 8K chunk over 32K/64K/128K total seq
            (8192, 32768, 64),
            (8192, 65536, 64),
            (8192, 131072, 64),
            # Varied sparse blocks at 128K
            (8192, 131072, 32),
            (8192, 131072, 128),
            (8192, 131072, 256),
        ]:
            bench_sparse_prefill(q_len, seq_len, nsb, dtype)

        print(f"\n{'='*60}")
        print(f"  Sparse Decode Performance ({dtype_name})")
        print(f"  BW%: fraction of BMG peak {BMG_PEAK_BW_GBS:.0f} GB/s")
        print(f"{'='*60}")
        for batch, seq_len in [
            (1, 2048),
            (1, 4096),
            (1, 8192),
            (1, 16384),
            (8, 4096),
            (8, 8192),
            # Long-context decode
            (1, 32768),
            (1, 65536),
            (1, 131072),
            (8, 16384),
            (8, 32768),
            (8, 65536),
        ]:
            bench_sparse_decode(batch, seq_len, dtype)

    print()
