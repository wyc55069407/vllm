import itertools
from typing import Optional, Tuple

import pytest
import torch
from vllm_kernel_custom import esimd_add, esimd_mul_lgrf

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")


def test_esimd_add(a, b, c, len):

    esimd_add(a, b, c, len)
    print(c)
    breakpoint()
    return c

def test_esimd_mul_lgrf(a, b, c, flag, len):

    esimd_mul_lgrf(a, b, c, flag, len)
    print(c)
    breakpoint()
    return c

if __name__ == "__main__":
    len = 40960 
    a = torch.rand((1, len), dtype=torch.float16, device="xpu")
    b = torch.rand((1, len), dtype=torch.float16, device="xpu")*0.1
    c = torch.zeros((1, len), dtype=torch.float16, device="xpu")
    flag = 1000
    len = len
    test_esimd_add(a, b, c, len)

    ref_res = a + b
    # Compare results
    torch.testing.assert_close(
        c.to(torch.float32), ref_res.to(torch.float32), rtol=1e-3, atol=1e-5
    )

    test_esimd_mul_lgrf(a, b, c, flag, len)

    ref_res = a * b
    # Compare results
    torch.testing.assert_close(
        c.to(torch.float32), ref_res.to(torch.float32), rtol=1e-3, atol=1e-5
    )
