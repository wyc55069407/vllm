"""E2E test: write KV with triton_reshape_and_cache_flash, read with ESIMD SDP.

This tests the exact data flow that happens in vLLM:
1. Write K/V to paged cache using triton_reshape_and_cache_flash
2. Read from cache using ESIMD SDP kernel
3. Compare against PyTorch reference reading from the same cache
"""
import torch
import math
import sys
sys.path.insert(0, '/home/sas/yuchen/vllm_env/vllm')


def test_esimd_sdp_with_triton_cache():
    from vllm_kernel_custom import esimd_sdp_paged
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )

    device = "xpu"
    num_heads = 16
    num_kv_heads = 4
    head_dim = 256
    block_size = 1024
    group_size = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    # Simulate decode: batch=1, seq_len=10 (prefill), then 5 decode steps
    prefill_len = 10
    total_decode_steps = 5
    total_len = prefill_len + total_decode_steps

    # Allocate cache: 1 block per sequence
    num_blocks = 2  # small cache
    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_dim,
                           dtype=torch.bfloat16, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

    # Pre-generate all keys, values for the sequence
    all_keys = torch.randn(total_len, num_kv_heads, head_dim,
                           dtype=torch.bfloat16, device=device) * 0.1
    all_values = torch.randn(total_len, num_kv_heads, head_dim,
                             dtype=torch.bfloat16, device=device) * 0.1
    all_queries = torch.randn(total_len, num_heads, head_dim,
                              dtype=torch.bfloat16, device=device) * 0.1

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Step 1: Prefill - write all prefill tokens to cache
    slot_mapping_prefill = torch.arange(prefill_len, dtype=torch.int64,
                                        device=device)
    key_cache, value_cache = kv_cache.unbind(0)
    triton_reshape_and_cache_flash(
        all_keys[:prefill_len], all_values[:prefill_len],
        key_cache, value_cache,
        slot_mapping_prefill, "auto", k_scale, v_scale,
    )
    torch.xpu.synchronize()

    # Step 2: Run decode steps one at a time
    for step in range(total_decode_steps):
        cur_len = prefill_len + step + 1  # seq_len including new token
        token_idx = prefill_len + step

        # Write new token to cache
        slot_mapping = torch.tensor([token_idx], dtype=torch.int64,
                                    device=device)
        key_cache, value_cache = kv_cache.unbind(0)
        triton_reshape_and_cache_flash(
            all_keys[token_idx:token_idx+1],
            all_values[token_idx:token_idx+1],
            key_cache, value_cache,
            slot_mapping, "auto", k_scale, v_scale,
        )
        torch.xpu.synchronize()

        # Query for this decode step
        query = all_queries[token_idx:token_idx+1]  # [1, num_heads, head_dim]
        seq_lens = torch.tensor([cur_len], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32,
                                       device=device)

        # ESIMD SDP
        output_esimd = torch.zeros_like(query)
        esimd_sdp_paged(
            query, kv_cache, output_esimd,
            block_table, seq_lens, query_start_loc,
            num_heads, num_kv_heads, head_dim, block_size,
            cur_len, scale, 1,
        )
        torch.xpu.synchronize()

        # PyTorch reference: read from cache manually
        output_ref = torch.zeros_like(query)
        key_cache_r = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, hd]
        val_cache_r = kv_cache[1]

        for h in range(num_heads):
            kv_h = h // group_size
            k_list = []
            v_list = []
            for pos in range(cur_len):
                bi = pos // block_size
                bo = pos % block_size
                blk = block_table[0, bi].item()
                k_list.append(key_cache_r[blk, bo, kv_h].float())
                v_list.append(val_cache_r[blk, bo, kv_h].float())
            k_dense = torch.stack(k_list, dim=0)  # [cur_len, head_dim]
            v_dense = torch.stack(v_list, dim=0)

            q_vec = query[0, h].float()
            scores = (k_dense @ q_vec) * scale
            weights = torch.softmax(scores, dim=0)
            out_val = (weights.unsqueeze(1) * v_dense).sum(0)
            output_ref[0, h] = out_val.to(torch.bfloat16)

        # Compare
        diff = (output_esimd.float() - output_ref.float()).abs()
        max_diff = diff.max().item()
        mean_abs = output_ref.float().abs().mean().item()
        rel_diff = max_diff / (mean_abs + 1e-8)

        status = "PASS" if rel_diff < 0.1 else "FAIL"
        print(f"Decode step {step}: seq_len={cur_len:3d}, "
              f"max_diff={max_diff:.6f}, mean_abs={mean_abs:.6f}, "
              f"rel_diff={rel_diff:.4f} [{status}]")

        if status == "FAIL":
            # Print some sample values
            worst_h = diff[0].sum(dim=-1).argmax().item()
            print(f"  Worst head: {worst_h}")
            print(f"  ESIMD: {output_esimd[0, worst_h, :5].float().tolist()}")
            print(f"  Ref:   {output_ref[0, worst_h, :5].float().tolist()}")

    # Also test prefill
    print("\n--- Prefill test ---")
    query_pf = all_queries[:prefill_len]
    seq_lens_pf = torch.tensor([prefill_len], dtype=torch.int32, device=device)
    qsl_pf = torch.tensor([0, prefill_len], dtype=torch.int32, device=device)

    # Reset cache and refill with just prefill data
    kv_cache2 = torch.zeros_like(kv_cache)
    key_cache2, value_cache2 = kv_cache2.unbind(0)
    triton_reshape_and_cache_flash(
        all_keys[:prefill_len], all_values[:prefill_len],
        key_cache2, value_cache2,
        slot_mapping_prefill, "auto", k_scale, v_scale,
    )
    torch.xpu.synchronize()

    output_pf_esimd = torch.zeros_like(query_pf)
    esimd_sdp_paged(
        query_pf, kv_cache2, output_pf_esimd,
        block_table, seq_lens_pf, qsl_pf,
        num_heads, num_kv_heads, head_dim, block_size,
        prefill_len, scale, 1,
    )
    torch.xpu.synchronize()

    # Reference for prefill (causal)
    output_pf_ref = torch.zeros_like(query_pf)
    key_cache2_r = kv_cache2[0]
    val_cache2_r = kv_cache2[1]
    for qi in range(prefill_len):
        q_pos = qi  # causal position
        kv_end = q_pos + 1
        for h in range(num_heads):
            kv_h = h // group_size
            k_list = []
            v_list = []
            for pos in range(kv_end):
                bi = pos // block_size
                bo = pos % block_size
                blk = block_table[0, bi].item()
                k_list.append(key_cache2_r[blk, bo, kv_h].float())
                v_list.append(val_cache2_r[blk, bo, kv_h].float())
            k_dense = torch.stack(k_list, dim=0)
            v_dense = torch.stack(v_list, dim=0)
            q_vec = query_pf[qi, h].float()
            scores = (k_dense @ q_vec) * scale
            weights = torch.softmax(scores, dim=0)
            out_val = (weights.unsqueeze(1) * v_dense).sum(0)
            output_pf_ref[qi, h] = out_val.to(torch.bfloat16)

    diff_pf = (output_pf_esimd.float() - output_pf_ref.float()).abs()
    max_diff_pf = diff_pf.max().item()
    mean_abs_pf = output_pf_ref.float().abs().mean().item()
    rel_diff_pf = max_diff_pf / (mean_abs_pf + 1e-8)
    status_pf = "PASS" if rel_diff_pf < 0.1 else "FAIL"
    print(f"Prefill: max_diff={max_diff_pf:.6f}, "
          f"rel_diff={rel_diff_pf:.4f} [{status_pf}]")

    # Run 3 times to check determinism
    print("\n--- Determinism test (3 runs, decode step 0) ---")
    results = []
    for run in range(3):
        kv_cache3 = torch.zeros_like(kv_cache)
        key_cache3, value_cache3 = kv_cache3.unbind(0)
        # Write prefill + 1 decode token
        slot_map_all = torch.arange(prefill_len + 1, dtype=torch.int64,
                                    device=device)
        triton_reshape_and_cache_flash(
            all_keys[:prefill_len+1], all_values[:prefill_len+1],
            key_cache3, value_cache3,
            slot_map_all, "auto", k_scale, v_scale,
        )
        torch.xpu.synchronize()

        query_d = all_queries[prefill_len:prefill_len+1]
        seq_lens_d = torch.tensor([prefill_len + 1], dtype=torch.int32,
                                  device=device)
        qsl_d = torch.tensor([0, 1], dtype=torch.int32, device=device)
        output_d = torch.zeros_like(query_d)
        esimd_sdp_paged(
            query_d, kv_cache3, output_d,
            block_table, seq_lens_d, qsl_d,
            num_heads, num_kv_heads, head_dim, block_size,
            prefill_len + 1, scale, 1,
        )
        torch.xpu.synchronize()
        results.append(output_d.float().cpu().clone())

    # Check if all 3 runs match
    for i in range(1, 3):
        diff_det = (results[i] - results[0]).abs().max().item()
        print(f"  Run {i} vs 0: max_diff={diff_det:.8f}")


if __name__ == "__main__":
    test_esimd_sdp_with_triton_cache()
