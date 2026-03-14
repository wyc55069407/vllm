"""Unit test for ESIMD GDN kernel — compare against PyTorch reference."""

import torch
import math


def pytorch_gdn_ref(A_log, dt_bias, a, b, q, k, v, state, cu_seqlens,
                     state_indices, scale):
    """PyTorch reference for GDN update (sequential per-token)."""
    N = cu_seqlens.shape[0] - 1
    T = q.shape[0]
    H = q.shape[1]
    K = q.shape[2]
    HV = v.shape[1]
    V = v.shape[2]
    heads_per_group = HV // H

    o = torch.zeros(T, HV, V, dtype=torch.bfloat16, device=q.device)

    for n in range(N):
        bos = cu_seqlens[n].item()
        eos = cu_seqlens[n + 1].item()
        if eos <= bos:
            continue
        state_idx = state_indices[n].item()
        if state_idx < 0:
            continue

        for t in range(bos, eos):
            for hv in range(HV):
                i_h = hv // heads_per_group

                # Gating
                a_val = a[t, hv].float()
                b_val = b[t, hv].float()
                A_log_val = A_log[hv].float()
                dt_bias_val = dt_bias[hv].float()

                x = a_val + dt_bias_val
                sp = x.item() if x.item() > 20.0 else math.log(1.0 + math.exp(x.item()))
                neg_exp_A = -math.exp(A_log_val.item())
                g = neg_exp_A * sp
                exp_g = math.exp(g)
                beta = 1.0 / (1.0 + math.exp(-b_val.item()))

                # L2 normalize q, k
                q_vec = q[t, i_h].float()
                k_vec = k[t, i_h].float()
                q_norm = torch.sqrt(torch.sum(q_vec * q_vec) + 1e-6)
                k_norm = torch.sqrt(torch.sum(k_vec * k_vec) + 1e-6)
                q_vec = q_vec / q_norm * scale
                k_vec = k_vec / k_norm

                # Process each V row
                for v_idx in range(V):
                    h = state[state_idx, hv, v_idx].float()
                    h *= exp_g
                    recon = torch.dot(h, k_vec)
                    v_val = v[t, hv, v_idx].float()
                    v_upd = (v_val - recon) * beta
                    h += v_upd * k_vec
                    o_val = torch.dot(h, q_vec)
                    o[t, hv, v_idx] = o_val.to(torch.bfloat16)
                    state[state_idx, hv, v_idx] = h.to(state.dtype)

    return o, state


def test_esimd_gdn():
    from vllm_kernel_custom import esimd_gdn_update

    device = "xpu"
    # Small test: 1 sequence, 4 tokens, H=2, HV=4, K=128, V=128
    N = 1
    T = 4
    H = 2
    HV = 4
    K = 128
    V = 128
    scale = K ** -0.5

    torch.manual_seed(42)

    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(T, HV, dtype=torch.bfloat16, device=device) * 0.5
    b = torch.randn(T, HV, dtype=torch.bfloat16, device=device) * 0.5
    q = torch.randn(T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(T, H, K, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(T, HV, V, dtype=torch.bfloat16, device=device) * 0.1
    state = torch.zeros(1, HV, V, K, dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
    state_indices = torch.tensor([0], dtype=torch.int32, device=device)

    # PyTorch reference
    state_ref = state.clone()
    output_ref, state_ref = pytorch_gdn_ref(
        A_log, dt_bias, a, b, q, k, v, state_ref, cu_seqlens,
        state_indices, scale)

    # ESIMD kernel
    state_esimd = state.clone()
    output_esimd = torch.zeros(T, HV, V, dtype=torch.bfloat16, device=device)
    esimd_gdn_update(
        A_log, dt_bias, a, b, q, k, v,
        state_esimd, output_esimd,
        cu_seqlens, state_indices,
        N, H, HV, K, V, scale, 1,  # inplace_state=1
    )
    torch.xpu.synchronize()

    # Compare output
    output_ref_f = output_ref.float()
    output_esimd_f = output_esimd.float()
    max_diff_out = (output_ref_f - output_esimd_f).abs().max().item()
    rms_diff_out = ((output_ref_f - output_esimd_f) ** 2).mean().sqrt().item()
    out_scale = output_ref_f.abs().mean().item()

    print(f"Output max_diff: {max_diff_out:.6f}")
    print(f"Output rms_diff: {rms_diff_out:.6f}")
    print(f"Output mean_abs: {out_scale:.6f}")
    print(f"Output rel_rms:  {rms_diff_out / (out_scale + 1e-8):.6f}")

    # Compare state
    max_diff_state = (state_ref - state_esimd).abs().max().item()
    rms_diff_state = ((state_ref - state_esimd) ** 2).mean().sqrt().item()
    state_scale = state_ref.abs().mean().item()

    print(f"State  max_diff: {max_diff_state:.6f}")
    print(f"State  rms_diff: {rms_diff_state:.6f}")
    print(f"State  mean_abs: {state_scale:.6f}")
    print(f"State  rel_rms:  {rms_diff_state / (state_scale + 1e-8):.6f}")

    # Check for NaN
    has_nan_out = torch.isnan(output_esimd).any().item()
    has_nan_state = torch.isnan(state_esimd).any().item()
    print(f"NaN in output: {has_nan_out}")
    print(f"NaN in state:  {has_nan_state}")

    # Print sample values
    print(f"\nRef output[0,0,:5]:   {output_ref_f[0, 0, :5].tolist()}")
    print(f"ESIMD output[0,0,:5]: {output_esimd_f[0, 0, :5].tolist()}")

    # Assert reasonable accuracy
    rel_rms = rms_diff_out / (out_scale + 1e-8)
    if has_nan_out or has_nan_state:
        print("\nFAIL: NaN detected!")
    elif rel_rms > 0.1:
        print(f"\nFAIL: rel_rms too high ({rel_rms:.4f})")
    else:
        print("\nPASS")


if __name__ == "__main__":
    test_esimd_gdn()
