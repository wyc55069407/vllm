"""
Standalone ULT for sparse paged prefill SDP — compares both old and fast kernels
against a PyTorch GPU reference implementation.
"""
import os
import torch
import math

os.environ["INFLLMV2_FAST_PREFILL"] = "0"  # will toggle dynamically


def gpu_sparse_prefill_reference(query, kv_cache, block_table, seq_lens,
                                  sparse_mask, sparse_mask_cnt,
                                  nh, nkvh, hd, bs, scale):
    """PyTorch GPU reference for sparse paged prefill SDP.

    query: [q_len, nh, hd] bf16/fp16 on xpu
    kv_cache: [2, num_blocks, block_size, nkvh, hd] on xpu
    block_table: [1, max_blocks] int32 on xpu
    seq_lens: [1] int32 on xpu
    sparse_mask: [nkvh, q_blocks, 1024] int32 on xpu
    sparse_mask_cnt: [nkvh, q_blocks] int32 on xpu
    """
    q_len = query.shape[0]
    q_blocks = (q_len + 15) // 16
    group_size = nh // nkvh
    seq_len = seq_lens[0].item()
    history_len = seq_len - q_len

    q = query.float()  # stay on xpu
    k_cache = kv_cache[0]  # [num_blocks, block_size, nkvh, hd]
    v_cache = kv_cache[1]
    bt = block_table[0]  # [max_blocks]

    output = torch.zeros(q_len, nh, hd, dtype=torch.float32, device=query.device)

    for kvh in range(nkvh):
        for qb in range(q_blocks):
            q_start = qb * 16
            q_end = min(q_start + 16, q_len)

            n_kv_blocks = sparse_mask_cnt[kvh, qb].item()
            if n_kv_blocks == 0:
                continue

            kv_indices = sparse_mask[kvh, qb, :n_kv_blocks]  # on xpu

            # Gather KV tokens from sparse blocks
            tok_list = []
            for blk_i in range(n_kv_blocks):
                kv_blk_id = kv_indices[blk_i].item()
                tok_start = kv_blk_id * 64
                for t in range(64):
                    tok = tok_start + t
                    if tok >= seq_len:
                        break
                    tok_list.append(tok)

            if len(tok_list) == 0:
                continue

            n_tokens = len(tok_list)
            K = torch.zeros(n_tokens, hd, dtype=torch.float32, device=query.device)
            V = torch.zeros(n_tokens, hd, dtype=torch.float32, device=query.device)

            for idx, tok in enumerate(tok_list):
                page_idx = tok // bs
                page_off = tok % bs
                phys_page = bt[page_idx].item()
                K[idx] = k_cache[phys_page, page_off, kvh].float()
                V[idx] = v_cache[phys_page, page_off, kvh].float()

            # For each Q head in this KV head group
            for qh_off in range(group_size):
                qh = kvh * group_size + qh_off

                q_slice = q[q_start:q_end, qh]  # [q_chunk, hd]
                scores = q_slice @ K.t() * scale  # [q_chunk, n_tokens]

                # Causal mask
                for qi_local in range(q_end - q_start):
                    qi = q_start + qi_local
                    causal_bound = history_len + qi
                    for idx, tok in enumerate(tok_list):
                        if tok > causal_bound:
                            scores[qi_local, idx] = float('-inf')

                # Softmax
                scores = scores - scores.max(dim=-1, keepdim=True).values
                scores = torch.exp(scores)
                scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-12)

                # Attn @ V
                output[q_start:q_end, qh] = scores @ V

    return output


def run_kernel(query, kv_cache, block_table, seq_lens,
               sparse_mask, sparse_mask_cnt,
               nh, nkvh, hd, bs, max_sl, scale, topk, use_fast):
    """Run the ESIMD kernel."""
    from vllm_kernel_custom import esimd_sdp_paged_sparse

    q_len = query.shape[0]
    qsl = torch.tensor([0, q_len], dtype=torch.int32, device='xpu')

    os.environ["INFLLMV2_FAST_PREFILL"] = "1" if use_fast else "0"
    out = torch.zeros(q_len, nh, hd, dtype=query.dtype, device='xpu')
    esimd_sdp_paged_sparse(query, kv_cache, out, block_table, seq_lens, qsl,
                            sparse_mask.int(), sparse_mask_cnt.int(),
                            nh, nkvh, hd, bs, max_sl, scale, 0, topk)
    torch.xpu.synchronize()
    return out


def compare(name, out_a, out_b, threshold=0.5):
    """Element-by-element comparison."""
    a = out_a.float()
    b = out_b.float()
    if a.device != b.device:
        b = b.to(a.device)

    diff = (a - b).abs()
    max_diff = diff.max().item()
    avg_diff = diff.mean().item()
    nan_cnt = torch.isnan(a).sum().item()
    large_cnt = (diff > threshold).sum().item()

    status = "PASS" if (max_diff < threshold and large_cnt == 0 and nan_cnt == 0) else "FAIL"
    print(f"  {name}: max_diff={max_diff:.6f} avg_diff={avg_diff:.6f} "
          f"nan={nan_cnt} large(>{threshold})={large_cnt} [{status}]")

    if status == "FAIL":
        q_len, nh, hd = a.shape
        for h in range(min(8, nh)):
            h_diff = diff[:, h, :].max().item()
            h_avg = diff[:, h, :].mean().item()
            print(f"    head {h:2d}: max={h_diff:.6f} avg={h_avg:.6f}")

    return status == "PASS"


def test_with_random_data(q_len=16, seq_len=1024, data_scale=0.1):
    """Test with random data."""
    print(f"\n=== Random data test: q_len={q_len}, seq_len={seq_len}, scale_factor={data_scale} ===")

    nh, nkvh, hd, bs = 32, 2, 128, 128
    topk = 64
    scale = 1.0 / math.sqrt(hd)

    num_pages = (seq_len + bs - 1) // bs + 1
    kv_cache = (torch.randn(2, num_pages, bs, nkvh, hd, dtype=torch.bfloat16) * data_scale).to('xpu')
    q = (torch.randn(q_len, nh, hd, dtype=torch.bfloat16) * data_scale).to('xpu')

    bt = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0).to('xpu')
    sl = torch.tensor([seq_len], dtype=torch.int32, device='xpu')
    max_sl = seq_len

    q_blocks = (q_len + 15) // 16
    num_kv_blocks = (seq_len + 63) // 64

    sparse_mask = torch.zeros(nkvh, q_blocks, 1024, dtype=torch.int32, device='xpu')
    sparse_mask_cnt = torch.zeros(nkvh, q_blocks, dtype=torch.int32, device='xpu')
    for kvh in range(nkvh):
        for qb in range(q_blocks):
            n = min(num_kv_blocks, topk)
            sparse_mask_cnt[kvh, qb] = n
            for i in range(n):
                sparse_mask[kvh, qb, i] = i

    # GPU PyTorch reference
    ref = gpu_sparse_prefill_reference(q, kv_cache, bt, sl,
                                        sparse_mask, sparse_mask_cnt,
                                        nh, nkvh, hd, bs, scale)

    out_old = run_kernel(q, kv_cache, bt, sl, sparse_mask, sparse_mask_cnt,
                          nh, nkvh, hd, bs, max_sl, scale, topk, use_fast=False)
    out_fast = run_kernel(q, kv_cache, bt, sl, sparse_mask, sparse_mask_cnt,
                           nh, nkvh, hd, bs, max_sl, scale, topk, use_fast=True)

    print(f"  Ref: mean={ref.mean():.6f} absmax={ref.abs().max():.6f}")
    ok1 = compare("Old vs Ref", out_old, ref, threshold=0.1)
    ok2 = compare("Fast vs Ref", out_fast, ref, threshold=0.1)
    ok3 = compare("Fast vs Old", out_fast, out_old, threshold=0.01)

    # Show first few values for q_pos=0
    if not ok2:
        print(f"\n  q_pos=0, first 5 HD elements:")
        for h in range(min(4, nh)):
            r = ref[0, h, :5].tolist()
            o = out_old[0, h, :5].float().tolist()
            f = out_fast[0, h, :5].float().tolist()
            print(f"    head {h}: ref =[{','.join(f'{v:.4f}' for v in r)}]")
            print(f"             old =[{','.join(f'{v:.4f}' for v in o)}]")
            print(f"             fast=[{','.join(f'{v:.4f}' for v in f)}]")

    return ok1 and ok2 and ok3


def test_with_real_data():
    """Test with real E2E dumped data."""
    dump_path = '/tmp/sparse_kv_dump.pt'
    if not os.path.exists(dump_path):
        print("\n=== No real data dump found, skipping ===")
        return True

    print(f"\n=== Real KV data test ===")
    dump = torch.load(dump_path, weights_only=False)

    kv_pages = dump['kv_pages']
    used_pages = dump['used_pages']
    full_shape = dump['kv_full_shape']

    kv_cache = torch.zeros(full_shape, dtype=kv_pages.dtype, device='xpu')
    for i, pg in enumerate(used_pages):
        kv_cache[:, pg] = kv_pages[:, i].to('xpu')

    q = dump['query'].to('xpu')
    bt = dump['block_table'].to('xpu')
    sl = dump['seq_lens'].to('xpu')
    mask = dump['sparse_mask'].to('xpu').int()
    mask_cnt = dump['sparse_mask_cnt'].to('xpu').int()
    nh, nkvh, hd, bs = dump['nh'], dump['nkvh'], dump['hd'], dump['bs']
    max_sl, scale, topk = dump['max_sl'], dump['scale'], dump['topk']

    test_q_len = 16
    q_blocks = 1
    q_t = q[:test_q_len]
    m_t = mask[:, :q_blocks, :].contiguous()
    mc_t = mask_cnt[:, :q_blocks].contiguous()

    print(f"  q_len={test_q_len}, seq_len={sl[0].item()}, nh={nh}, nkvh={nkvh}")

    ref = gpu_sparse_prefill_reference(q_t, kv_cache, bt, sl,
                                        m_t, mc_t, nh, nkvh, hd, bs, scale)

    out_old = run_kernel(q_t, kv_cache, bt, sl, m_t, mc_t,
                          nh, nkvh, hd, bs, max_sl, scale, topk, use_fast=False)
    out_fast = run_kernel(q_t, kv_cache, bt, sl, m_t, mc_t,
                           nh, nkvh, hd, bs, max_sl, scale, topk, use_fast=True)

    print(f"  Ref: mean={ref.mean():.6f} absmax={ref.abs().max():.6f}")
    ok1 = compare("Old vs Ref", out_old, ref, threshold=0.5)
    ok2 = compare("Fast vs Ref", out_fast, ref, threshold=0.5)
    ok3 = compare("Fast vs Old", out_fast, out_old, threshold=0.1)

    print(f"\n  q_pos=0, first 5 HD elements:")
    for h in range(min(4, nh)):
        r = ref[0, h, :5].tolist()
        o = out_old[0, h, :5].float().tolist()
        f = out_fast[0, h, :5].float().tolist()
        print(f"    head {h}: ref =[{','.join(f'{v:.4f}' for v in r)}]")
        print(f"             old =[{','.join(f'{v:.4f}' for v in o)}]")
        print(f"             fast=[{','.join(f'{v:.4f}' for v in f)}]")

    return ok1 and ok2 and ok3


if __name__ == '__main__':
    test_with_random_data(q_len=16, seq_len=1024, data_scale=0.1)
    test_with_random_data(q_len=16, seq_len=1024, data_scale=2.0)
    test_with_random_data(q_len=128, seq_len=2048, data_scale=2.0)
    test_with_real_data()
