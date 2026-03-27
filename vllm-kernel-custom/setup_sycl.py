# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from pathlib import Path

from setuptools import find_packages, setup
# from torch.utils.cpp_extension import BuildExtension, SyclExtension
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()


def _get_version():
    with open(root / "pyproject_sycl.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "vllm_kernel_custom"
include_dirs = [
    root / "include",
    root / "csrc",
]

sources = [
    "csrc/xpu/awq_dequantize.sycl",
    "csrc/xpu/uni_esimd_kernel.sycl",
    "csrc/xpu/onednn_fp8.sycl",
    "csrc/xpu/moe_decode.sycl",
    "csrc/xpu/moe_prefill.sycl",
    "csrc/xpu/moe_sigmoid_topk.sycl",
    "csrc/xpu/torch_extension_sycl.cc",
]

# oneDNN include path (from PyTorch bundled headers)
import torch
torch_include = str(Path(torch.__file__).parent / "include")

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
             f"-I{torch_include}"],
}

extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu",
                   "-ldnnl"]

ext_modules = []
ext_modules.append(
    SyclExtension(
        name="vllm_kernel_custom.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )
)

### for lgrf esimd kernels
sources = [
    "csrc/xpu/uni_esimd_kernel_lgrf.sycl",
    "csrc/xpu/infllmv2_kernels_lgrf.sycl",
    "csrc/xpu/moe_prefill_lgrf.sycl",
    "csrc/xpu/torch_extension_sycl_lgrf.cc",
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
             "-fsycl-targets=spir64_gen", "-Xs", "-device bmg -options -doubleGRF"],
}

ext_modules.append(
    SyclExtension(
        name="vllm_kernel_custom.common_ops_lgrf",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )
)
### for lgrf esimd kernels

setup(
    name="vllm-kernel-custom",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp10"}},
)
