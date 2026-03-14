import itertools
import unittest

import torch
import torch.nn as nn

# import intel_extension_for_pytorch
from vllm_kernel_custom import esimd_kernel_uni

g_test_dequant = False
g_test_perf = True
g_test_M_size = 1024

def is_k_contiguous(tt):
    return tt.shape[-1] == tt.stride()[-2]

def fp8_gemm_opt(input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, block_n=128, block_k=128, bias: torch.Tensor = None):
    if not (is_k_contiguous(input) and is_k_contiguous(weight) and is_k_contiguous(weight_scale)):
        print("fp8_gemm_opt shape not supported!")
        return None
    if weight.dtype != torch.float8_e4m3fn or weight_scale.dtype != torch.float16:
        print("fp8_gemm_opt type not supported!")
        return None
    if weight.shape[-1] != input.shape[-1]:
        print("fp8_gemm_opt input weight k not same!")

    M = input.shape[-2]
    N = weight.shape[-2]
    K = input.shape[-1]

    if (weight_scale.shape[-1] != (K + block_k - 1) // block_k or weight_scale.shape[-2] != (N + block_n - 1) // block_n):
        print("fp8_gemm_opt weight_scale shape incorrect!")

    has_bias = 0
    bias_in = input
    if bias is not None:
        has_bias = 1
        bias_in = bias

    if (M > 8):  # GEMM not GEMV
        if g_test_perf:
            weight_list = []
            for i in range(32):
                weight_list.append(weight.clone())
            dq_weight_fp16 = torch.zeros(weight.shape, dtype=torch.float16, device=weight.device)
            for i in range(10000):
                esimd_kernel_uni(weight, weight_scale, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16,
                    4999, N, K, block_n, block_k, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0)                
                # dequant and use FP16 GEMM
                if has_bias:
                    output = torch.matmul(input, dq_weight_fp16.transpose(0, 1)) + bias
                else:
                    output = torch.matmul(input, dq_weight_fp16.transpose(0, 1))
        else:
            dq_weight_fp16 = torch.zeros(weight.shape, dtype=torch.float16, device=weight.device)
            esimd_kernel_uni(weight, weight_scale, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16, dq_weight_fp16,
                4999, N, K, block_n, block_k, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0)
            # dequant and use FP16 GEMM
            if has_bias:
                output = torch.matmul(input, dq_weight_fp16.transpose(0, 1)) + bias
            else:
                output = torch.matmul(input, dq_weight_fp16.transpose(0, 1))
        return output

    batch = 1
    if len(input.shape) == 4:
        batch = input.shape[-3]
        output = torch.zeros(input.shape[0], input.shape[1], M, N, device=input.device, dtype=input.dtype)
    elif len(input.shape) == 3:
        batch = input.shape[-3]
        output = torch.zeros(input.shape[0], M, N, device=input.device, dtype=input.dtype)
    elif len(input.shape) == 2:
        output = torch.zeros(M, N, device=input.device, dtype=input.dtype)

    if g_test_perf:
        weight_list = []
        for i in range(32):
            weight_list.append(weight.clone())
        for i in range(10000):
            esimd_kernel_uni(input, weight_list[i % 32], weight_scale, bias_in, output, output, output, output, output, output,
                5000, M, N, K, batch, block_n, block_k, has_bias, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        esimd_kernel_uni(input, weight, weight_scale, bias_in, output, output, output, output, output, output,
            5000, M, N, K, batch, block_n, block_k, has_bias, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0)

    return output

# TODO: use interface in cpu.py
# from vllm_kernel_custom.common_ops import (
#     convert_weight_packed,
#     fp8_scaled_mm_cpu,
#     int8_scaled_mm_cpu,
#     int8_scaled_mm_with_quant,
#     per_token_quant_int8_cpu,
#     weight_packed_linear,
# )
from utils_gemm import (
    convert_weight,
    native_w8a8_per_token_matmul,
    per_token_quant_int8,
    precision,
)

# from sglang.test.test_utils import CustomTestCase


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias, dtype=torch.float16)

    def forward(self, x):
        return self.linear(x)

class TestGemm(unittest.TestCase):
# class TestGemm(CustomTestCase):
    # M = [1, 101]
    # N = [32 * 13]
    # K = [32 * 16]
    # has_bias = [False, True]
    has_bias = [False]

    # M_int8 = [2, 128]
    # N_int8 = [32 * 12]
    # K_int8 = [32 * 17]

    # M_fp8 = [1, 11]
    # N_fp8 = [128, 224]
    # K_fp8 = [512, 576]
    # M_fp8 = [1]
    # N_fp8 = [1536+512+64]   # self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
    # K_fp8 = [256]

    M_fp8 = [g_test_M_size]
    N_fp8 = [7168]
    K_fp8 = [4096]
    
    # M_fp8 = [g_test_M_size]
    # N_fp8 = [1536+512+64]   # self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
    # K_fp8 = [7168]
    
    # M_fp8 = [g_test_M_size]
    # N_fp8 = [32*192]   
    # K_fp8 = [1536]
    
    # M_fp8 = [g_test_M_size]
    # N_fp8 = [18432*2//4]   
    # K_fp8 = [7168]
    
    # M_fp8 = [g_test_M_size]
    # N_fp8 = [7168]   
    # K_fp8 = [18432//4]
    
    # M_fp8 = [g_test_M_size]
    # N_fp8 = [256]   # gate
    # K_fp8 = [7168]
    
    # def _bf16_gemm(self, M, N, K, has_bias):

    #     mat1 = torch.randn(M, K, dtype=torch.bfloat16)
    #     mat2 = torch.randn(N, K, dtype=torch.bfloat16)

    #     ref = torch.matmul(mat1.float(), mat2.float().t())
    #     if has_bias:
    #         bias = torch.randn(N, dtype=torch.float32)
    #         ref.add_(bias.bfloat16())

    #     ref = ref.bfloat16()

    #     out = weight_packed_linear(mat1, mat2, bias if has_bias else None, False)

    #     packed_mat2 = convert_weight_packed(mat2)
    #     out2 = weight_packed_linear(mat1, packed_mat2, bias if has_bias else None, True)

    #     atol = rtol = precision[ref.dtype]
    #     self.assertTrue(torch.allclose(ref, out, atol=atol, rtol=rtol))
    #     self.assertTrue(torch.allclose(ref, out2, atol=atol, rtol=rtol))

    # def test_bf16_gemm(self):
    #     for params in itertools.product(
    #         self.M,
    #         self.N,
    #         self.K,
    #         self.has_bias,
    #     ):
    #         with self.subTest(
    #             M=params[0],
    #             N=params[1],
    #             K=params[2],
    #             has_bias=params[3],
    #         ):
    #             self._bf16_gemm(*params)

    # def _int8_gemm(self, M, N, K, has_bias):
    #     dtype = torch.bfloat16
    #     A = torch.randn((M, K), dtype=dtype) / 10
    #     Aq, As = per_token_quant_int8(A)

    #     factor_for_scale = 1e-2
    #     int8_max = 127
    #     int8_min = -128

    #     B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
    #     Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
    #     Bs = torch.rand(N) * factor_for_scale

    #     bias = torch.randn(N) if has_bias else None
    #     ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

    #     atol = rtol = precision[ref_out.dtype]

    #     # Aq2, As2 = per_token_quant_int8_cpu(A)
    #     # out = int8_scaled_mm_cpu(
    #     #     Aq2, Bq, As2, Bs, bias if has_bias else None, torch.bfloat16, False
    #     # )
    #     # self.assertTrue(torch.allclose(ref_out, out, atol=atol, rtol=rtol))

    #     # # test the fused version
    #     # fused_out = int8_scaled_mm_with_quant(
    #     #     A, Bq, Bs, bias if has_bias else None, torch.bfloat16, False
    #     # )
    #     if has_bias:
    #         bias = bias.to("xpu")

    #     fused_out = native_w8a8_per_token_matmul(Aq.to("xpu"), Bq.to("xpu"), As.to("xpu"), Bs.to("xpu"), bias, dtype)
    #     fused_out = fused_out.to("cpu")
    #     self.assertTrue(torch.allclose(ref_out, fused_out, atol=atol, rtol=rtol))

    # def test_int8_gemm(self):
    #     for params in itertools.product(
    #         self.M_int8,
    #         self.N_int8,
    #         self.K_int8,
    #         self.has_bias,
    #     ):
    #         with self.subTest(
    #             M=params[0],
    #             N=params[1],
    #             K=params[2],
    #             has_bias=params[3],
    #         ):
    #             self._int8_gemm(*params)

    def _fp8_gemm(self, M, N, K, has_bias):
        print("M, N, K, has_bias: ", M, " ", N, " ", K, " ", has_bias)
        prepack = True
        chunk = False
        scale_block_size_N = 128
        scale_block_size_K = 128
        assert scale_block_size_N <= N
        assert scale_block_size_K <= K
        A_dtype = torch.float16

        model = Mod(K, N, has_bias).eval()
        if chunk:
            data = torch.randn(M, K + 6, dtype=A_dtype).narrow(1, 0, K)
        else:
            data = torch.randn(M, K, dtype=A_dtype)

        # weight = model.linear.weight  # (N, K)
        weight = torch.randn(model.linear.weight.shape, dtype=A_dtype)

        # make weight layout more dynamic
        for i in range(weight.shape[-1] // scale_block_size_K):
            for j in range(weight.shape[-2] // scale_block_size_N):
                rand = torch.randn(1) + torch.randn(1) + torch.randn(1) + torch.randn(1)
                weight[j*scale_block_size_N:(j+1)*scale_block_size_N, i*scale_block_size_K:(i+1)*scale_block_size_K] *= rand

        if has_bias:
            bias = model.linear.bias

        fp8_weight, scales, dq_weight = convert_weight(
            weight, [scale_block_size_N, scale_block_size_K], A_dtype
        )

        print("fp8_weight.shape, scales.shape, dq_weight.shape :",fp8_weight.shape, scales.shape, dq_weight.shape)
        print("fp8_weight.stride(), scales.stride(), dq_weight.stride() :",fp8_weight.stride(), scales.stride(), dq_weight.stride())
        print("data.shape, data.stride()", data.shape, data.stride())
        print("scales.dtype", scales.dtype)

        # data[...] = 1.
        # dq_weight2 = dq_weight.clone()
        # dq_weight2 = torch.randn(dq_weight.shape, dtype=A_dtype)
        # dq_weight2[...] = 0.0098
        # fp8_weight = dq_weight2.to(torch.float8_e4m3fn)
        # scales[...] = 1
        # dq_weight = fp8_weight.to(torch.float16) * 1

        # scales *= 10
        # dq_weight *= 10
        
        if has_bias:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T) + bias.to(A_dtype)
        else:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T)
        
        # if has_bias:
        #     opt = torch.matmul(data.to(A_dtype).to("xpu"), dq_weight.T.to("xpu")) + bias.to(A_dtype).to("xpu")
        # else:
        #     opt = torch.matmul(data.to(A_dtype).to("xpu"), dq_weight.T.to("xpu"))
        # opt = opt.to("cpu")

        data = data.to(A_dtype).to("xpu")
        fp8_weight = fp8_weight.to("xpu")
        scales = scales.to("xpu")
        dq_weight = dq_weight.to("xpu")
        if has_bias:
            bias = bias.to(A_dtype).to("xpu")
            opt = fp8_gemm_opt(data, fp8_weight, scales, scale_block_size_K, scale_block_size_N, bias)
        else:
            opt = fp8_gemm_opt(data, fp8_weight, scales, scale_block_size_K, scale_block_size_N)

        opt = opt.to("cpu")

        if not g_test_perf:
            breakpoint()

        # if prepack:
        #     fp8_weight = convert_weight_packed(fp8_weight)

        # opt = fp8_scaled_mm_cpu(
        #     data,
        #     fp8_weight,
        #     scales,
        #     [scale_block_size_N, scale_block_size_K],
        #     bias if has_bias else None,
        #     data.dtype,
        #     prepack,
        # )
        # atol = rtol = precision[ref.dtype]
        atol = 0.5
        rtol = 0.01
        print(ref, opt)
        self.assertTrue(torch.allclose(ref, opt, atol=atol, rtol=rtol))

    def test_fp8_gemm(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._fp8_gemm(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)