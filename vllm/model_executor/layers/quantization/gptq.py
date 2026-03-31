# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import os
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Union

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper
else:
    QuantizationMethods = str

logger = init_logger(__name__)


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        autoround_version: str = "",
        modules_in_block_to_quantize: list[str] | None = None,
        checkpoint_format: str = "",
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is dict[str, dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits."
            )
        # Somehow gptq_gemm 4-bit is buggy, maybe fix it in the future.
        # For now, show a warning, since gptq_marlin will be used by default.
        if self.weight_bits == 4:
            logger.warning_once(
                "Currently, the 4-bit gptq_gemm kernel for GPTQ is buggy. "
                "Please switch to gptq_marlin."
            )

        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []

        # used to identify GPTQ model quantized by autoround
        self.autoround_version = autoround_version

        # GPTQ v1 and v2 format deals with zero points differently.
        # Currently GPTQModel stores v1 format checkpoints by default,
        # but provides the option to set `format="gptq_v2"` in `QuantizeConfig`.
        self.checkpoint_format = checkpoint_format

    def __repr__(self) -> str:
        return (
            f"GPTQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}), "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize}), "
            f"checkpoint_format={self.checkpoint_format})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        autoround_version = cls.get_from_keys_or(
            config, ["autoround_version"], default=""
        )
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        checkpoint_format = cls.get_from_keys_or(
            config, ["checkpoint_format"], default=""
        )
        return cls(
            weight_bits,
            group_size,
            desc_act,
            lm_head_quantized,
            dynamic,
            autoround_version,
            modules_in_block_to_quantize,
            checkpoint_format,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["GPTQLinearMethod", "QuantizeMethodBase"] | None:
        if isinstance(layer, FusedMoE):
            # GPTQ MoE support: fall back to MoeWNA16 for broad compatibility
            from .moe_wna16 import MoeWNA16Config

            # TODO: maybe update this for GPTQv2 format checkpoints
            config = {
                "quant_method": "gptq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "sym": True,  # GPTQ typically uses symmetric quantization
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        return get_linear_quant_method(self, layer, prefix, GPTQLinearMethod)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_in_block_to_quantize is not None:
            self.modules_in_block_to_quantize = hf_to_vllm_mapper.apply_list(
                self.modules_in_block_to_quantize
            )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        if self.modules_in_block_to_quantize:
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten original modules_in_block_to_quantize
                self.modules_in_block_to_quantize = [
                    item
                    for sublist in self.modules_in_block_to_quantize
                    for item in sublist
                ]
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_in_block_to_quantize = list(quant_layers)


class ExllamaState(Enum):
    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

        # GPTQ v1 and v2 format deals with zero points differently
        self.use_v2_format = quant_config.checkpoint_format == "gptq_v2"

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor.numerator != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (
            input_size != input_size_per_partition
            and self.quant_config.group_size != -1
        ):
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )
        qzeros_args = {
            "data": torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }
        if scale_and_zero_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.exllama_state = exllama_state

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.platforms import current_platform

        # for torch.compile
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # On XPU, skip exllama shuffle (CUDA-only) and use oneDNN INT4 GEMM
        if current_platform.is_xpu():
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty(
                    (0,), dtype=torch.int, device=layer.g_idx.device
                )
            layer.exllama_state = ExllamaState.READY

            # Convert to oneDNN u4 format for native INT4 GEMM
            # (only when desc_act=false so groups are sequential)
            if (self.quant_config.weight_bits == 4
                    and not self.quant_config.desc_act):
                self._convert_to_onednn_u4(layer, self.use_v2_format)
            return

        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if layer.exllama_state == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty(
                    (0,), dtype=torch.int, device=layer.g_idx.device
                )
            layer.exllama_state = ExllamaState.READY
            ops.gptq_shuffle(layer.qweight, layer.g_idx, self.quant_config.weight_bits)

    @staticmethod
    def _convert_to_onednn_u4(
        layer: torch.nn.Module, use_v2_format: bool
    ) -> None:
        """Convert GPTQ packed int32 weights to oneDNN u4 format.

        Requires vllm-kernel-custom with onednn_w4a16_int4 op.
        Only works when desc_act=false (sequential K-groups).

        Stores onednn_weight, onednn_scales, onednn_zp on the layer.
        Falls back silently if vllm-kernel-custom is unavailable.
        """
        try:
            import vllm_kernel_custom  # noqa: F401
        except ImportError:
            return  # will use dequant+F.linear fallback

        qweight = layer.qweight.data   # [K/8, N] int32
        qzeros = layer.qzeros.data     # [num_groups, N/8] int32
        scales = layer.scales.data      # [num_groups, N] fp16

        K_packed, N = qweight.shape
        K = K_packed * 8
        num_groups = scales.shape[0]
        group_size = K // num_groups
        device = qweight.device

        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)

        # Unpack qweight: [K/8, N] int32 → [K, N] uint8
        w_unpacked = (
            (qweight.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF
        ).permute(0, 2, 1).reshape(K, N).to(torch.uint8)

        # Transpose to [N, K] and repack as oneDNN u4: [N, K/2] uint8
        # oneDNN u4 ba: physical [N,K], byte[n,k//2] = val[n,2k] | val[n,2k+1]<<4
        w_nk = w_unpacked.t().contiguous()
        w_u4 = (w_nk[:, 0::2] & 0xF) | ((w_nk[:, 1::2] & 0xF) << 4)

        # Unpack qzeros: [num_groups, N/8] int32 → [num_groups, N] uint8
        N_packed = qzeros.shape[1]
        z_unpacked = (
            (qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF
        ).permute(0, 2, 1).reshape(num_groups, N_packed * 8)[:, :N]
        z_unpacked = z_unpacked.to(torch.uint8)

        # Adjust zero points for GPTQ v1 format
        if not use_v2_format:
            z_unpacked = z_unpacked + 1

        # Pack zero points as oneDNN u4: [num_groups, N/2] uint8
        z_u4 = (z_unpacked[:, 0::2] & 0xF) | (
            (z_unpacked[:, 1::2] & 0xF) << 4
        )

        # Scales to fp32 for oneDNN: [num_groups, N]
        scales_fp32 = scales.to(torch.float32).contiguous()

        layer.onednn_weight = w_u4.contiguous()
        layer.onednn_scales = scales_fp32
        layer.onednn_zp = z_u4.contiguous()
        layer.onednn_K = K
        layer.onednn_N = N
        layer.onednn_group_size = group_size

        # ESIMD GEMV: pre-transpose scales to [N, K/GS] in bf16/fp16
        # (contiguous per-row for dequant_dot pattern)
        scales_t = scales_fp32.t().contiguous()  # [N, num_groups]
        layer.esimd_scales_fp16 = scales_t.to(torch.float16)
        layer.esimd_scales_bf16 = scales_t.to(torch.bfloat16)

        logger.info(
            "GPTQ layer converted to oneDNN u4: "
            "weight [%d, %d] u4, scales [%d, %d] fp32, "
            "group_size=%d",
            N, K, num_groups, N, group_size,
        )

    @staticmethod
    def _gptq_dequant_4bit(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        use_v2_format: bool,
    ) -> torch.Tensor:
        """Dequantize GPTQ 4-bit packed weights to fp16 using pure PyTorch.

        qweight: [K/8, N] int32 — 8 int4 values packed per int32 along K
        qzeros:  [num_groups, N/8] int32 — packed zero points along N
        scales:  [num_groups, N] fp16
        g_idx:   [K] int32 or empty tensor
        Returns: [N, K] dequantized weight (ready for F.linear)
        """
        K_packed, N = qweight.shape
        K = K_packed * 8
        num_groups = scales.shape[0]
        device = qweight.device

        # Unpack qweight: [K/8, N] -> [K, N]
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)
        w_unpacked = (
            (qweight.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF
        )  # [K/8, N, 8]
        w_unpacked = w_unpacked.permute(0, 2, 1).reshape(K, N).to(scales.dtype)

        # Unpack qzeros: [num_groups, N/8] -> [num_groups, N]
        z_unpacked = (
            (qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF
        )  # [num_groups, N/8, 8]
        N_packed = qzeros.shape[1]
        z_unpacked = z_unpacked.permute(0, 2, 1).reshape(
            num_groups, N_packed * 8
        )[:, :N].to(scales.dtype)

        # Map each input channel to its group
        if g_idx.numel() > 0:
            group_map = g_idx.long()
        else:
            group_size = K // num_groups
            group_map = torch.arange(K, device=device) // group_size

        # Gather per-channel scales and zeros
        s = scales[group_map]      # [K, N]
        z = z_unpacked[group_map]  # [K, N]

        # Dequantize
        if use_v2_format:
            w_deq = (w_unpacked - z) * s
        else:
            # GPTQ v1: zero point offset by 1
            w_deq = (w_unpacked - (z + 1)) * s

        return w_deq.t().contiguous()  # [N, K]

    @staticmethod
    def _onednn_gemm(reshaped_x, layer, M, N, K, output):
        """oneDNN W4A16 INT4 GEMM dispatch."""
        dummy_bias = torch.empty(
            0, dtype=reshaped_x.dtype, device=reshaped_x.device)
        torch.ops.vllm_kernel_custom.onednn_w4a16_int4(
            reshaped_x.contiguous(),
            layer.onednn_weight,
            layer.onednn_scales,
            layer.onednn_zp,
            dummy_bias,
            output,
            M, N, K,
            layer.onednn_group_size,
            1,  # has_zp
            0,  # has_bias
        )
        return output

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.platforms import current_platform

        out_shape = x.shape[:-1] + (layer.qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if current_platform.is_xpu() and self.quant_config.weight_bits == 4:
            if hasattr(layer, 'onednn_weight'):
                M = reshaped_x.shape[0]
                K = layer.onednn_K
                N = layer.onednn_N
                output = torch.empty(
                    M, N, dtype=reshaped_x.dtype,
                    device=reshaped_x.device,
                )

                # ESIMD GEMV fast path for decode (M<=8)
                if (M <= 8
                    and (os.environ.get("MINICPM5_ESIMD_GEMV", "0") == "1"
                         or os.environ.get("MINICPM4_ESIMD_GEMV", "0") == "1")
                    and hasattr(layer, 'esimd_scales_fp16')):
                    try:
                        esimd_scales = (
                            layer.esimd_scales_bf16
                            if reshaped_x.dtype == torch.bfloat16
                            else layer.esimd_scales_fp16
                        )
                        torch.ops.vllm_kernel_custom.esimd_w4a16_gemv(
                            reshaped_x.contiguous(),
                            layer.onednn_weight,
                            esimd_scales,
                            output,
                            layer.onednn_group_size,
                        )
                        if not getattr(GPTQLinearMethod,
                                       '_esimd_gemv_logged', False):
                            GPTQLinearMethod._esimd_gemv_logged = True
                            import logging
                            logging.getLogger(__name__).warning(
                                "ESIMD W4A16 GEMV active: M=%d, N=%d, K=%d",
                                M, N, K)
                    except Exception as e:
                        # Fall through to oneDNN on any error
                        if not getattr(GPTQLinearMethod,
                                       '_esimd_gemv_err_logged', False):
                            GPTQLinearMethod._esimd_gemv_err_logged = True
                            import logging
                            logging.getLogger(__name__).warning(
                                "ESIMD GEMV fallback to oneDNN: %s", e)
                        output = self._onednn_gemm(
                            reshaped_x, layer, M, N, K, output)
                else:
                    output = self._onednn_gemm(
                        reshaped_x, layer, M, N, K, output)
            else:
                # Fallback: dequantize + F.linear (desc_act=true or no
                # vllm-kernel-custom)
                w_deq = self._gptq_dequant_4bit(
                    layer.qweight,
                    layer.qzeros,
                    layer.scales,
                    layer.g_idx,
                    self.use_v2_format,
                )
                output = torch.nn.functional.linear(reshaped_x, w_deq)
        else:
            # GPTQ v1 and v2 format checkpoints deals with zero points
            # differently, and require different gemm kernels.
            output = ops.gptq_gemm(
                reshaped_x,
                layer.qweight,
                layer.qzeros,
                layer.scales,
                layer.g_idx,
                layer.exllama_state == ExllamaState.READY,
                self.use_v2_format,
                self.quant_config.weight_bits,
            )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
