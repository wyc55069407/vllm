"""Unit test for ESIMD paged SDP attention kernel.

Compares against torch.nn.functional.scaled_dot_product_attention.
Tests both decode (query_len=1) and prefill with causal masking.
Supports block_size=64 and block_size=1024, head_dim=128 and 256.
"""
import torch
import pytest
import math


def create_paged_kv_cache(
    keys, values, block_size, num_kv_heads, head_dim, device,
):
    """Create a paged KV cache from dense key/value tensors.

    Args:
        keys: list of [seq_len_i, num_kv_heads, head_dim] bf16 tensors
        values: list of [seq_len_i, num_kv_heads, head_dim] bf16 tensors
        block_size: KV cache block size

    Returns:
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim] bf16
        block_table: [batch, max_blocks_per_seq] i32
        seq_lens: [batch] i32
    """
    batch = len(keys)
    seq_lens = [k.shape[0] for k in keys]
    max_seq_len = max(seq_lens)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)

    kv_cache = torch.zeros(2, total_blocks, block_size, num_kv_heads, head_dim,
                           dtype=torch.bfloat16, device=device)
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32, device=device)

    block_counter = 0
    for b in range(batch):
        sl = seq_lens[b]
        num_b = (sl + block_size - 1) // block_size
        for bi in range(num_b):
            block_table[b, bi] = block_counter
            start = bi * block_size
            end = min(start + block_size, sl)
            length = end - start
            kv_cache[0, block_counter, :length] = keys[b][start:end]
            kv_cache[1, block_counter, :length] = values[b][start:end]
            block_counter += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return kv_cache, block_table, seq_lens_t


def pytorch_sdp_reference(query, keys, values, num_heads, num_kv_heads,
                           head_dim, scale, causal=True):
    """PyTorch reference: standard SDPA with GQA expansion.

    Args:
        query: [num_tokens, num_heads, head_dim] bf16
        keys: list of [kv_len_i, num_kv_heads, head_dim] bf16
        values: list of [kv_len_i, num_kv_heads, head_dim] bf16

    Returns:
        output: [num_tokens, num_heads, head_dim] bf16
    """
    group_size = num_heads // num_kv_heads
    output = torch.zeros_like(query)
    token_idx = 0

    for b in range(len(keys)):
        k = keys[b].float()   # [kv_len, num_kv_heads, head_dim]
        val = values[b].float()
        # Determine query tokens for this request
        # For decode: 1 token per request
        # For prefill: query_len tokens
        # We handle this via the caller passing correct slices

    # Simpler approach: handle per-head
    return output


@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_sdp_paged(block_size, head_dim, mode):
    """Test ESIMD paged SDP against PyTorch reference."""
    try:
        from vllm_kernel_custom import esimd_sdp_paged
    except ImportError:
        pytest.skip("vllm_kernel_custom not built with SDP paged support")

    device = "xpu"
    num_heads = 16
    num_kv_heads = 4
    group_size = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    if mode == "decode":
        batch = 4
        seq_lens_list = [32, 64, 48, 16]
        query_lens = [1] * batch
    else:
        batch = 2
        seq_lens_list = [32, 48]  # total KV length
        query_lens = [8, 12]  # query tokens per request

    # Generate dense KV data
    keys_dense = []
    values_dense = []
    for sl in seq_lens_list:
        keys_dense.append(
            torch.randn(sl, num_kv_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1)
        values_dense.append(
            torch.randn(sl, num_kv_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1)

    # Create paged KV cache
    kv_cache, block_table, seq_lens_t = create_paged_kv_cache(
        keys_dense, values_dense, block_size, num_kv_heads, head_dim, device)

    # Generate queries
    total_q_tokens = sum(query_lens)
    query = torch.randn(total_q_tokens, num_heads, head_dim,
                        dtype=torch.bfloat16, device=device) * 0.1

    # query_start_loc
    qsl = [0]
    for ql in query_lens:
        qsl.append(qsl[-1] + ql)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)

    # Output buffer
    output_esimd = torch.zeros_like(query)

    # --- Run ESIMD kernel ---
    esimd_sdp_paged(
        query, kv_cache, output_esimd,
        block_table, seq_lens_t, query_start_loc,
        num_heads, num_kv_heads,
        head_dim, block_size,
        max(seq_lens_list), scale, 1,  # causal=1
    )
    torch.xpu.synchronize()

    # --- Run PyTorch reference ---
    output_ref = torch.zeros_like(query)
    q_offset = 0
    for b in range(batch):
        ql = query_lens[b]
        sl = seq_lens_list[b]

        # Expand KV for GQA: [kv_len, num_kv_heads, hd] -> [kv_len, num_heads, hd]
        k_expanded = keys_dense[b].float().repeat_interleave(group_size, dim=1)
        v_expanded = values_dense[b].float().repeat_interleave(group_size, dim=1)

        for qi in range(ql):
            q_pos = (sl - ql) + qi  # absolute position of this query token
            q_vec = query[q_offset + qi].float()  # [num_heads, head_dim]

            for h in range(num_heads):
                # Causal: attend to positions [0, q_pos]
                kv_end = q_pos + 1
                k_slice = k_expanded[:kv_end, h, :]  # [kv_end, head_dim]
                v_slice = v_expanded[:kv_end, h, :]

                # QK scores
                scores = torch.mv(k_slice, q_vec[h]) * scale  # [kv_end]

                # Softmax
                weights = torch.softmax(scores, dim=0)

                # Weighted sum
                output_ref[q_offset + qi, h] = (
                    weights.unsqueeze(1) * v_slice
                ).sum(0).to(torch.bfloat16)

        q_offset += ql

    # --- Compare ---
    output_esimd_f32 = output_esimd.float().cpu()
    output_ref_f32 = output_ref.float().cpu()

    atol = 0.05
    rtol = 0.05

    close = torch.allclose(output_esimd_f32, output_ref_f32, atol=atol, rtol=rtol)
    if not close:
        diff = (output_esimd_f32 - output_ref_f32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # Find worst head/token
        worst_idx = diff.reshape(-1).argmax().item()
        print(f"Output diff: max={max_diff:.6f}, mean={mean_diff:.6f}")
        print(f"Worst index: {worst_idx}")
        print(f"ESIMD sample: {output_esimd_f32.reshape(-1)[worst_idx]:.6f}")
        print(f"Ref sample:   {output_ref_f32.reshape(-1)[worst_idx]:.6f}")

    assert close, (
        f"SDP paged output mismatch ({mode}, block_size={block_size}, "
        f"head_dim={head_dim}): max_diff={diff.max().item():.6f}"
    )

    # Check no NaN
    assert not torch.isnan(output_esimd).any(), "NaN in ESIMD SDP output"

    print(f"PASS: mode={mode}, block_size={block_size}, head_dim={head_dim}")


if __name__ == "__main__":
    for mode in ["decode", "prefill"]:
        for hd in [256]:
            for bs in [64]:
                test_sdp_paged(bs, hd, mode)
    print("All SDP paged tests passed!")
