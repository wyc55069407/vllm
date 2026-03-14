import random
import unittest

import torch
from vllm_kernel_custom import esimd_kernel_uni_lgrf

#from sglang.srt.layers.attention.triton_ops.decode_attention import (
#    decode_attention_fwd,
#    decode_attention_fwd_grouped,
#    decode_attention_fwd_normal,
#)
#from sglang.srt.layers.attention.triton_ops.extend_attention import (
#    extend_attention_fwd,
#    redundant_attention,
#)
#from sglang.srt.layers.attention.triton_ops.prefill_attention import (
#    context_attention_fwd,
#)
#from sglang.test.test_utils import CustomTestCase
from torch.nn.functional import scaled_dot_product_attention

# class TestTritonAttention(CustomTestCase):
class TestTritonAttention(unittest.TestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(19)
    
    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        k_extend: torch.Tensor,
        v_extend: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        #breakpoint()
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):

            extend_seq_len_q = extend_seq_lens[seq_idx]     # q extend
            prefill_seq_len_q = extend_prefix_lens[seq_idx]  # q prefill

            seq_len_kv = seq_lens[seq_idx]   # kv tokens num
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            #breakpoint()
            #per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_tokens = req_to_token[req_pool_idx, :prefill_seq_len_q]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            
            per_req_key_ext = k_extend[start_q:end_q].movedim(0, query.dim() - 2)
            per_req_value_ext = v_extend[start_q:end_q].movedim(0, query.dim() - 2)
            
            per_req_key = torch.cat((per_req_key, per_req_key_ext), 1)
            per_req_value = torch.cat((per_req_value, per_req_value_ext), 1)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0).cpu(),
                    per_req_key.unsqueeze(0).cpu(),
                    per_req_value.unsqueeze(0).cpu(),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            per_req_out_redudant = per_req_out_redudant.to(output.device)
            
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_extend_esimd(
        self,
        query: torch.Tensor,
        k_extend: torch.Tensor,
        v_extend: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_indices: torch.Tensor,
        extend_prefix_lens,
        extend_seq_lens,
        scaling=None,
        enable_gqa=False,
        causal=False,
        batch_size=1,
    ):
        #breakpoint()

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        #query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for batch_idx in range(batch_size):

            extend_seq_len = extend_seq_lens[batch_idx]     
            prefix_seq_len = extend_prefix_lens[batch_idx]  

            end_q = start_q + extend_seq_len
            end_kv = start_kv + prefix_seq_len

            per_req_query = query[start_q:end_q, :, :]

            per_req_tokens = kv_indices[start_kv:end_kv]
            
            per_req_key_ext = k_extend[start_q:end_q]
            per_req_value_ext = v_extend[start_q:end_q]
            
            esimd_out = output[start_q:end_q, :, :]

            esimd_kernel_uni_lgrf(
                per_req_query, per_req_key_ext, per_req_value_ext,k_cache, v_cache, per_req_tokens, esimd_out, esimd_out, esimd_out, esimd_out,
                1006, query.shape[-2], k_extend.shape[-2], extend_seq_len, prefix_seq_len, k_extend.shape[-1], v_extend.shape[-1], 
                0, 0, 0,    
                scaling, 1.0, 1.0, 1.0, 1.0)
            
            output[start_q:end_q, :, :] = esimd_out
            
            start_q, start_kv = end_q, end_kv
        return output
        
        
    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D, D_V):
        #breakpoint()
        dtype = torch.float16

        # b_seq_len_prefix = torch.randint(
            # 1, N_CTX // 2, (B,), dtype=torch.int32, device="xpu"
        # )
        # b_seq_len_extend = torch.randint(
            # 1, N_CTX // 2, (B,), dtype=torch.int32, device="xpu"
        # )
        
        b_seq_len_prefix = torch.randint(
            2048, 2049, (B,), dtype=torch.int32, device="xpu"
        )
        b_seq_len_extend = torch.randint(
            2048, 2049, (B,), dtype=torch.int32, device="xpu"
        )
        b_seq_len_prefix.zero_()

        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len.cpu(), 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device="xpu")
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="xpu")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="xpu")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="xpu")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        
        kv_indices = torch.zeros(
            (b_seq_len_prefix.cpu().sum().item(),), dtype=torch.int32, device="xpu"
        )
        kv_indices = kv_indices.to("xpu")

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )

        total_token_num = torch.sum(b_seq_len.cpu()).item()
        extend_token_num = torch.sum(b_seq_len_extend.cpu()).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="xpu"
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D_V), dtype=dtype, device="xpu"
        # ).normal_(mean=0.1, std=0.2)             *****************************try

        #k_buffer[0:1] = 1
        #v_buffer[:,0,0:1] = 1 
        #v_buffer[:,1,0:1] = 2 
        ).normal_(mean=0.1, std=0.2)
        #v_buffer[0:1] = seq_v
        #v_buffer[0:1]=1
        #v_buffer[1:2]=2
        #v_buffer[2:3]=2
        #v_buffer[3:4]=1
        
        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="xpu")
        v_extend = torch.empty((extend_token_num, H_KV, D_V), dtype=dtype, device="xpu")
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="xpu")
        #q_extend[0:1] = 1
        #seq_col = torch.arange(192)
        #q_extend[0:1] = seq_col
        
        #q_extend[0:2] = 1
        
        #breakpoint()
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="xpu"
            ).normal_(mean=0.1, std=0.2)

        o_extend = torch.empty((extend_token_num, H_Q, D_V), dtype=dtype, device="xpu")
        o_extend_mask = torch.empty(
            (extend_token_num, H_Q, D_V), dtype=dtype, device="xpu"
        )
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D_V), dtype=dtype, device="xpu"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend.cpu(), 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="xpu")
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        custom_mask = None
        mask_indptr = None

        #extend_attention_fwd(
        #    q_extend,
        #    k_extend,
        #    v_extend,
        #    o_extend,
        #    k_buffer,
        #    v_buffer,
        #    qo_indptr,
        #    kv_indptr,
        #    kv_indices,
        #    custom_mask,
        #    True,
        #    mask_indptr,
        #    max_len_extend,
        #)
        
        o_extend_ref = torch.empty((extend_token_num, H_Q, D_V), dtype=dtype, device="xpu")
        max_len = 0
        for i in range(B):
            curlen_prefix = kv_indptr[i+1] - kv_indptr[i]
            curlen_ext = qo_indptr[i+1] - qo_indptr[i]
            curlen = curlen_prefix + curlen_ext
            if curlen > max_len:
                max_len = curlen

        req_to_token = torch.zeros(B, max_len, dtype=torch.int32, device="xpu")
        b_req_idx = torch.arange(B, device="xpu").to(torch.int64)
        b_seq_len = torch.full((B,), max_len, device="xpu").to(torch.int64)
        extend_prefix_lens = torch.full((B,), max_len, device="xpu").to(torch.int64)
        extend_seq_lens = torch.full((B,), max_len, device="xpu").to(torch.int64)
        
        for i in range(B):
            curlen_prefix = kv_indptr[i+1] - kv_indptr[i]
            extend_prefix_lens[i] = curlen_prefix
            curlen_ext = qo_indptr[i+1] - qo_indptr[i]
            extend_seq_lens[i] = curlen_ext
            b_seq_len[i] = curlen_ext + curlen_prefix
            req_to_token[i,:curlen_prefix] = kv_indices[kv_indptr[i]:kv_indptr[i+1]]
        
        enable_gqa = H_Q != H_KV
        sm_scale = 1.0 / (D**0.5)

        #breakpoint()
        self._run_sdpa_forward_extend(
            q_extend,
            k_extend,
            v_extend,
            o_extend_ref,
            k_buffer,
            v_buffer,
            req_to_token,
            b_req_idx,
            b_seq_len,
            extend_prefix_lens,
            extend_seq_lens,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
            causal=True,
        )
        
        extend_prefix_lens_list = extend_prefix_lens.cpu().tolist() #forward_batch.extend_prefix_lens_cpu
        extend_seq_lens_list = extend_seq_lens.cpu().tolist()
        
        for i in range (100):
            self._run_sdpa_forward_extend_esimd(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                kv_indices,
                extend_prefix_lens_list,
                extend_seq_lens_list,
                scaling=sm_scale,
                enable_gqa=enable_gqa,
                causal=True,
                batch_size=B,
            )
        
        #breakpoint()
        
        #b_seq_mask_len = b_seq_len_extend * b_seq_len
        #custom_mask = torch.ones(
        #    (b_seq_mask_len.sum().item(),), dtype=torch.bool, device="xpu"
        #)
        #mask_indptr = torch.zeros((B + 1,), dtype=torch.int64, device="xpu")
        #mask_indptr[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
        #for i in range(B):
        #    causal_mask = (
        #        torch.tril(
        #            torch.ones(b_seq_len_extend[i], b_seq_len_extend[i]), diagonal=0
        #        )
        #        == 1
        #    )
        #    prefix_mask = torch.ones(
        #        b_seq_len_extend[i], b_seq_len_prefix[i], dtype=torch.bool
        #    )
        #    mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
        #    custom_mask[mask_indptr[i] : mask_indptr[i + 1]] = mask_flatten
        #
        #extend_attention_fwd(
        #    q_extend,
        #    k_extend,
        #    v_extend,
        #    o_extend_mask,
        #    k_buffer,
        #    v_buffer,
        #    qo_indptr,
        #    kv_indptr,
        #    kv_indices,
        #    custom_mask,
        #    True,
        #    mask_indptr,
        #    max_len_extend,
        #)
        #breakpoint()
        #
        #redundant_attention(
        #    q_extend,
        #    o_redundant,
        #    k_buffer,
        #    v_buffer,
        #    b_req_idx,
        #    b_start_loc,
        #    b_seq_len,
        #    b_seq_len_prefix,
        #    max_len_in_batch,
        #)
        #
        #self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2))
        #self.assertTrue(torch.allclose(o_extend_mask, o_redundant, rtol=1e-2))
        #self.assertTrue(torch.allclose(o_extend, o_extend_ref, rtol=1e-2))
        
        cos_sim = torch.nn.functional.cosine_similarity(
            o_extend_ref.flatten(), o_extend.flatten(), dim=0
        )
        print(cos_sim.item())
        #self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o_extend_ref, o_extend, atol=3e-2))
        
    def test_extend_attention(self):
        #return
        # Define the varying parameter values
        # attention_values = [128, 96, 80, 13]
        attention_values = [128]

        # Loop through the values and call the method
        for value in attention_values:
            #self._test_extend_attention_once(3, 100, 32, 32, value)
            self._test_extend_attention_once(2, 2877, 32, 32, 192, 128)
#            self._test_extend_attention_once(19, 12331, 12, 4, value)

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        num_heads = 4
        seq_lens = [8, 12]
        max_seq_len = max(seq_lens)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, device="xpu")
        k = torch.randn(sum(seq_lens), num_heads, head_dim, device="xpu")
        v = torch.randn(sum(seq_lens), num_heads, head_dim, device="xpu")
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, device="xpu")

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device="xpu")
        b_seq_len = torch.tensor(seq_lens, device="xpu")

        context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
        )

        cu_seq_lens = [0] * (len(seq_lens) + 1)
        for i, seq_len in enumerate(seq_lens):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

        for i in range(len(seq_lens)):
            start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
            o_torch = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].permute(1, 0, 2),
                k[start:end].permute(1, 0, 2),
                v[start:end].permute(1, 0, 2),
                is_causal=is_causal,
            ).permute(1, 0, 2)

            cos_sim = torch.nn.functional.cosine_similarity(
                o[start:end].flatten(), o_torch.flatten(), dim=0
            )
            self.assertTrue(cos_sim.item() > 1 - (1e-5))
            self.assertTrue(torch.allclose(o[start:end], o_torch, atol=1e-2))

    def test_context_attention(self):
        return

        head_dim = [128, 96, 80, 13]

        for dim in head_dim:
            for is_causal in [True, False]:
                self._test_context_attention_once(dim, is_causal)

    def _test_decode_attention_once(self, B, H_Q, H_KV, D):
        print("B, H_Q, H_KV, D : ", B, H_Q, H_KV, D)
        breakpoint()
        dtype = torch.bfloat16
        seq_len = 10  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="xpu")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="xpu")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="xpu")
        v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="xpu")

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D, dtype=dtype, device="xpu")

        b_seq_len = torch.full((B,), seq_len, device="xpu")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="xpu")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device="xpu")

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D),
            dtype=torch.float32,
            device="xpu",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="xpu",
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
        )

        print("q", q.shape, q.dtype)
        print("k_buffer", k_buffer.shape, k_buffer.dtype)
        print("v_buffer", v_buffer.shape, v_buffer.dtype)
        print("o", o.shape, o.dtype)
        print("kv_indptr", kv_indptr)
        print("kv_indices", kv_indices)
        print("attn_logits", attn_logits.shape, attn_logits.dtype)
        print("attn_lse", attn_lse.shape, attn_lse.dtype)
        print("num_kv_splits", num_kv_splits)
        print("max_kv_splits", max_kv_splits)
        print("sm_scale", sm_scale)
        print("------------------------")

    def test_decode_attention(self):
        return
        # Here we just to ensure there is no error
        # TODO: correctnesss test

        # Test configurations
        configs = [
            (2, 4, 4, 64),  # MHA
            (2, 4, 2, 64),  # GQA
            (2, 4, 4, 80),  # Non-standard head dim
            (2, 4, 4, 13),  # Prime number head dim
        ]

        for B, H_Q, H_KV, D in configs:
            self._test_decode_attention_once(B, H_Q, H_KV, D)

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode_esimd(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        attn_logits: torch.Tensor,
        attn_lse: torch.Tensor,
        num_kv_splits=None,
        max_kv_splits=None,
        sm_scale=None,
        batch_size=1,
    ):
        #breakpoint()
        for batch_idx in range(batch_size):  # B
            sdp_tmp = torch.empty(query.shape[-2], 512, v_cache.shape[-1], dtype=torch.float32, device='xpu') # max to alloc 
            torch.ops.torch_ipex.esimd_kernel_uni(
                query, k_cache, v_cache, kv_indptr, kv_indices, sdp_tmp, output, output, output, output,
                1013, query.shape[-2], k_cache.shape[-2], batch_idx,  k_cache.shape[-1], v_cache.shape[-1], 
                0, 0, 0, 0,    
                sm_scale, 1.0, 1.0, 1.0, 1.0)

        return output
        
    def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
        print("B, S, H_Q, H_KV, D, D_V : ", B, S, H_Q, H_KV, D, D_V)

        dtype = torch.float16
        seq_len = S  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="xpu")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="xpu")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="xpu")
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="xpu")

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="xpu")
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device="xpu")

        b_seq_len = torch.full((B,), seq_len, device="xpu")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="xpu")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens,dtype=torch.int32, device="xpu")

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="xpu",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="xpu",
        )

        #decode_attention_fwd_normal(
        #    q,
        #    k_buffer,
        #    v_buffer,
        #    o,
        #    kv_indptr,
        #    kv_indices,
        #    attn_logits,
        #    attn_lse,
        #    num_kv_splits,
        #    max_kv_splits,
        #    sm_scale,
        #)

        attn_logits1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="xpu",
        )
        attn_lse1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="xpu",
        )

        #for _ in range(1):
        #    decode_attention_fwd_grouped(
        #        q,
        #        k_buffer,
        #        v_buffer,
        #        o_grouped,
        #        kv_indptr,
        #        kv_indices,
        #        attn_logits1,
        #        attn_lse1,
        #        num_kv_splits,
        #        max_kv_splits,
        #        sm_scale,
        #    )
        print("q", q.shape, q.dtype)
        print("k_buffer", k_buffer.shape, k_buffer.dtype)
        print("v_buffer", v_buffer.shape, v_buffer.dtype)
        print("o_grouped", o_grouped.shape, o_grouped.dtype)
        print("kv_indptr", kv_indptr)
        print("kv_indices", kv_indices)
        print("attn_logits1", attn_logits1.shape, attn_logits1.dtype)
        print("attn_lse1", attn_lse1.shape, attn_lse1.dtype)
        print("num_kv_splits", num_kv_splits)
        print("max_kv_splits", max_kv_splits)
        print("sm_scale", sm_scale)
        print("------------------------")

        o_grouped_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device="xpu")
        max_len = 0
        for i in range(B):
            curlen = kv_indptr[i+1] - kv_indptr[i]
            if curlen > max_len:
                max_len = curlen
            
        req_to_token = torch.zeros(B, max_len, dtype=torch.int32, device="xpu")
        b_req_idx = torch.arange(B, device="xpu").to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device="xpu").to(torch.int64)
        for i in range(B):
            curlen = kv_indptr[i+1] - kv_indptr[i]
            b_seq_len[i] = curlen
            req_to_token[i,:curlen] = kv_indices[kv_indptr[i]:kv_indptr[i+1]]
        
        enable_gqa = H_Q != H_KV
        self._run_sdpa_forward_decode(
            q,
            o_grouped_ref,
            k_buffer,
            v_buffer,
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
        )
        
        
        # self._run_sdpa_forward_decode(
            # q,
            # o_grouped,
            # k_buffer,
            # v_buffer,
            # req_to_token,
            # b_req_idx,
            # b_seq_len,
            # scaling=sm_scale,
            # enable_gqa=enable_gqa,
        # )
        for i in range (5000):
            self._run_sdpa_forward_decode_esimd(
                q,
                k_buffer,
                v_buffer,
                o_grouped,
                kv_indptr,
                kv_indices,
                attn_logits1,
                attn_lse1,
                num_kv_splits,
                max_kv_splits,
                sm_scale,
                B
            )

        #breakpoint()
        cos_sim = torch.nn.functional.cosine_similarity(
            o_grouped_ref.flatten(), o_grouped.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o_grouped_ref, o_grouped, atol=3e-2))

    def test_grouped_decode_attention(self):
        return
        # seq_lens = [5, 100, 128, 500]
        seq_lens = [4096]
        configs = [
            #(2, 16, 16, 64, 64),
            #(2, 16, 1, 64, 64),
            #(2, 64, 1, 13, 13),
            #(2, 128, 1, 80, 80),
            #(2, 128, 2, 512, 512),
            (2, 32, 4, 576, 512),
        ]

        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)

    def _test_grouped_decode_attention_once_MLA(self, B, S, H_Q, H_KV, D, D_V):
        print("B, S, H_Q, H_KV, D, D_V : ", B, S, H_Q, H_KV, D, D_V)

        dtype = torch.float16
        seq_len = S  # This represents the number of tokens already in the sequence
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="xpu")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="xpu")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="xpu")
        v_buffer = k_buffer[:,:,:D_V]

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="xpu")
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device="xpu")

        b_seq_len = torch.full((B,), seq_len, device="xpu")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="xpu")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens,dtype=torch.int32, device="xpu")

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="xpu",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="xpu",
        )

        #decode_attention_fwd_normal(
        #    q,
        #    k_buffer,
        #    v_buffer,
        #    o,
        #    kv_indptr,
        #    kv_indices,
        #    attn_logits,
        #    attn_lse,
        #    num_kv_splits,
        #    max_kv_splits,
        #    sm_scale,
        #)

        attn_logits1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="xpu",
        )
        attn_lse1 = torch.empty(
            (B, H_Q, max_kv_splits, D_V),
            dtype=torch.float32,
            device="xpu",
        )

        #for _ in range(1):
        #    decode_attention_fwd_grouped(
        #        q,
        #        k_buffer,
        #        v_buffer,
        #        o_grouped,
        #        kv_indptr,
        #        kv_indices,
        #        attn_logits1,
        #        attn_lse1,
        #        num_kv_splits,
        #        max_kv_splits,
        #        sm_scale,
        #    )
        print("q", q.shape, q.dtype)
        print("k_buffer", k_buffer.shape, k_buffer.dtype)
        print("v_buffer", v_buffer.shape, v_buffer.dtype)
        print("o_grouped", o_grouped.shape, o_grouped.dtype)
        print("kv_indptr", kv_indptr)
        print("kv_indices", kv_indices)
        print("attn_logits1", attn_logits1.shape, attn_logits1.dtype)
        print("attn_lse1", attn_lse1.shape, attn_lse1.dtype)
        print("num_kv_splits", num_kv_splits)
        print("max_kv_splits", max_kv_splits)
        print("sm_scale", sm_scale)
        print("------------------------")

        o_grouped_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device="xpu")
        max_len = 0
        for i in range(B):
            curlen = kv_indptr[i+1] - kv_indptr[i]
            if curlen > max_len:
                max_len = curlen
            
        req_to_token = torch.zeros(B, max_len, dtype=torch.int32, device="xpu")
        b_req_idx = torch.arange(B, device="xpu").to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device="xpu").to(torch.int64)
        for i in range(B):
            curlen = kv_indptr[i+1] - kv_indptr[i]
            b_seq_len[i] = curlen
            req_to_token[i,:curlen] = kv_indices[kv_indptr[i]:kv_indptr[i+1]]
        
        enable_gqa = H_Q != H_KV
        self._run_sdpa_forward_decode(
            q,
            o_grouped_ref,
            k_buffer,
            v_buffer,
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
        )
        
        for i in range (5000):
            self._run_sdpa_forward_decode_esimd(
                q,
                k_buffer,
                v_buffer,
                o_grouped,
                kv_indptr,
                kv_indices,
                attn_logits1,
                attn_lse1,
                num_kv_splits,
                max_kv_splits,
                sm_scale,
                B
            )

        #breakpoint()

        cos_sim = torch.nn.functional.cosine_similarity(
            o_grouped_ref.flatten(), o_grouped.flatten(), dim=0
        )
        print(cos_sim.item())
        self.assertTrue(cos_sim.item() > 0.99)
        self.assertTrue(torch.allclose(o_grouped_ref, o_grouped, atol=3e-2))

    def test_grouped_decode_attention_MLA(self):
        return
        seq_lens = [4096]
        configs = [
            (2, 32, 1, 576, 512),
        ]
        for S in seq_lens:
            for B, H_Q, H_KV, D, D_V in configs:
                self._test_grouped_decode_attention_once_MLA(B, S, H_Q, H_KV, D, D_V)
if __name__ == "__main__":
    unittest.main()
