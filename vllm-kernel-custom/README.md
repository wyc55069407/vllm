# SGL Esimd Kernel

## env
- torch==2.8
- pytorch-triton-xpu==3.3.1
- scikit-build-core>=0.10
- ninja

## install
```bash
python setup_sycl.py install
```

## kernel API
```bash
from vllm_kernel_custom import esimd_add
len = 1000
a = torch.rand((1, len), dtype=torch.float16, device="xpu")
b = torch.rand((1, len), dtype=torch.float16, device="xpu")*0.1
c = torch.zeros((1, len), dtype=torch.float16, device="xpu")  # result
flag = 1000

esimd_add(a, b, c, len)

```
