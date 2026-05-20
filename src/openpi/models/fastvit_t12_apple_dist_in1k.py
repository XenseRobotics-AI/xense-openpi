"""Self-contained FastViT-T12 Apple distilled ImageNet encoder.

This module contains the architecture used by:

    timm/fastvit_t12.apple_dist_in1k

It keeps the FastViT-T12 model definition and the small helper layers local so
the file can be copied into another PyTorch project without depending on timm.
With ``num_classes=0`` the model follows the same feature extraction path used
by ``timm.create_model(..., num_classes=0)``:

    output = model(x)  # (B, 1024)

Original FastViT implementation and weights are from Apple ml-fastvit; the timm
source file is ``timm/models/fastvit.py``.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


MODEL_NAME = "fastvit_t12.apple_dist_in1k"
HF_HUB_ID = "timm/fastvit_t12.apple_dist_in1k"
ARCHITECTURE = "fastvit_t12"
SOURCE_FILE = "timm/models/fastvit.py"
FEATURE_DIM = 1024

DEFAULT_CFG: dict[str, Any] = {
    "url": "",
    "hf_hub_id": HF_HUB_ID,
    "architecture": ARCHITECTURE,
    "tag": "apple_dist_in1k",
    "custom_load": False,
    "input_size": (3, 256, 256),
    "fixed_input_size": False,
    "interpolation": "bicubic",
    "crop_pct": 0.9,
    "crop_mode": "center",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "num_classes": 1000,
    "pool_size": (8, 8),
    "first_conv": ("stem.0.conv_kxk.0.conv", "stem.0.conv_scale.conv"),
    "classifier": "head.fc",
    "license": "fastvit-license",
}


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None, round_limit: float = 0.9) -> int:
    """Round channel count to be divisible by ``divisor``."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def create_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding: Optional[int] = None,
    device=None,
    dtype=None,
) -> nn.Conv2d:
    if padding is None:
        padding = get_padding(kernel_size, stride=stride, dilation=dilation)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        device=device,
        dtype=dtype,
    )


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


class ConvNormAct(nn.Module):
    """Conv2d + BatchNorm2d + optional activation.

    Attribute names match timm's ConvNormAct for pretrained checkpoint
    compatibility: ``conv`` and ``bn``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_act: bool = True,
        act_layer: Type[nn.Module] = nn.ReLU,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **dd,
        )
        self.bn = nn.BatchNorm2d(out_channels, **dd)
        self.act = act_layer() if apply_act else nn.Identity()

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation block using timm-compatible ``fc1``/``fc2`` names."""

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        bias: bool = True,
        act_layer: Type[nn.Module] = nn.ReLU,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias, **dd)
        self.bn = nn.Identity()
        self.act = act_layer(inplace=True) if act_layer is nn.ReLU else act_layer()
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias, **dd)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class SelectAdaptivePool2d(nn.Module):
    """Small subset of timm's selectable global pooling used by this model."""

    def __init__(self, pool_type: str = "avg", flatten: bool = False) -> None:
        super().__init__()
        self.pool_type = pool_type or ""
        if self.pool_type in ("", "identity"):
            self.pool = nn.Identity()
        elif self.pool_type in ("avg", "fast", "fastavg"):
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.pool_type in ("max", "fastmax"):
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unsupported pool type: {pool_type!r}.")
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(x))

    def feat_mult(self) -> int:
        return 1


class ClassifierHead(nn.Module):
    """Classifier/embedding head with timm-compatible attributes."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: str = "avg",
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.use_conv = False
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes, **dd) if num_classes > 0 else nn.Identity()
        self.flatten = nn.Identity()

    def reset(self, num_classes: int, pool_type: Optional[str] = None) -> None:
        if pool_type is not None:
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)
        self.fc = nn.Linear(self.in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


def calculate_drop_path_rates(
    drop_path_rate: float,
    depths: int | list[int],
    stagewise: bool = False,
) -> list[float] | list[list[float]]:
    if isinstance(depths, int):
        if stagewise:
            raise ValueError("stagewise=True requires depths to be a list of stage depths.")
        return [x.item() for x in torch.linspace(0, drop_path_rate, depths, device="cpu")]

    total_depth = sum(depths)
    if stagewise:
        return [x.tolist() for x in torch.linspace(0, drop_path_rate, total_depth, device="cpu").split(depths)]
    return [x.item() for x in torch.linspace(0, drop_path_rate, total_depth, device="cpu")]


def checkpoint_seq(functions: nn.Sequential, x: torch.Tensor, every: int = 1) -> torch.Tensor:
    if isinstance(functions, nn.Sequential):
        functions = tuple(functions.children())
    else:
        functions = tuple(functions)

    def run_function(start: int, end: int):
        def forward(_x: torch.Tensor) -> torch.Tensor:
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x

        return forward

    for start in range(0, len(functions), every):
        end = min(start + every - 1, len(functions) - 1)
        x = torch_checkpoint(run_function(start, end), x, use_reentrant=False)
    return x


def num_groups(group_size: int, channels: int) -> int:
    if not group_size:
        return 1
    if channels % group_size != 0:
        raise ValueError(f"channels={channels} must be divisible by group_size={group_size}.")
    return channels // group_size


class MobileOneBlock(nn.Module):
    """MobileOne block used by FastViT."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 0,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        act_layer: Type[nn.Module] = nn.GELU,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.inference_mode = inference_mode
        self.groups = num_groups(group_size, in_chs)
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_conv_branches = num_conv_branches

        self.se = SqueezeExcite(out_chs, rd_divisor=1, **dd) if use_se else nn.Identity()

        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                bias=True,
                **dd,
            )
        else:
            self.reparam_conv = None
            self.identity = nn.BatchNorm2d(num_features=in_chs, **dd) if out_chs == in_chs and stride == 1 else None

            if num_conv_branches > 0:
                self.conv_kxk = nn.ModuleList(
                    [
                        ConvNormAct(
                            self.in_chs,
                            self.out_chs,
                            kernel_size=kernel_size,
                            stride=self.stride,
                            groups=self.groups,
                            apply_act=False,
                            **dd,
                        )
                        for _ in range(self.num_conv_branches)
                    ]
                )
            else:
                self.conv_kxk = None

            self.conv_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.conv_scale = ConvNormAct(
                    self.in_chs,
                    self.out_chs,
                    kernel_size=1,
                    stride=self.stride,
                    groups=self.groups,
                    apply_act=False,
                    **dd,
                )

        self.act = act_layer() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return self.act(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.identity is not None:
            identity_out = self.identity(x)

        scale_out = 0
        if self.conv_scale is not None:
            scale_out = self.conv_scale(x)

        out = scale_out + identity_out
        if self.conv_kxk is not None:
            for conv_branch in self.conv_kxk:
                out += conv_branch(x)

        return self.act(self.se(out))

    def reparameterize(self) -> None:
        """Fuse train-time branches into one convolution for inference."""
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = create_conv2d(
            in_channels=self.in_chs,
            out_channels=self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for name, param in self.named_parameters():
            if "reparam_conv" in name:
                continue
            param.detach_()

        self.__delattr__("conv_kxk")
        self.__delattr__("conv_scale")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        self.inference_mode = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.conv_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.conv_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        kernel_conv = 0
        bias_conv = 0
        if self.conv_kxk is not None:
            for idx in range(self.num_conv_branches):
                branch_kernel, branch_bias = self._fuse_bn_tensor(self.conv_kxk[idx])
                kernel_conv += branch_kernel
                bias_conv += branch_bias

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_bn_tensor(self, branch: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not isinstance(branch, nn.BatchNorm2d):
                raise TypeError(f"Expected ConvNormAct or BatchNorm2d, got {type(branch)!r}.")
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_chs // self.groups
                kernel_value = torch.zeros(
                    (self.in_chs, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_chs):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    """Large-kernel convolution block used by FastViT downsampling."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int,
        group_size: int,
        small_kernel: Optional[int] = None,
        use_se: bool = False,
        act_layer: Optional[Type[nn.Module]] = None,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.stride = stride
        self.groups = num_groups(group_size, in_chs)
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=1,
                groups=self.groups,
                bias=True,
                **dd,
            )
        else:
            self.reparam_conv = None
            self.large_conv = ConvNormAct(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=self.stride,
                groups=self.groups,
                apply_act=False,
                **dd,
            )
            if small_kernel is not None:
                if small_kernel > kernel_size:
                    raise ValueError("small_kernel cannot be larger than kernel_size.")
                self.small_conv = ConvNormAct(
                    in_chs,
                    out_chs,
                    kernel_size=small_kernel,
                    stride=self.stride,
                    groups=self.groups,
                    apply_act=False,
                    **dd,
                )

        self.se = SqueezeExcite(out_chs, rd_ratio=0.25, **dd) if use_se else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            out = self.reparam_conv(x)
        else:
            out = self.large_conv(x)
            if hasattr(self, "small_conv"):
                out = out + self.small_conv(x)
        out = self.se(out)
        out = self.act(out)
        return out

    def get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        eq_k, eq_b = self._fuse_bn(self.large_conv.conv, self.large_conv.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def reparameterize(self) -> None:
        if self.reparam_conv is not None:
            return

        eq_k, eq_b = self.get_kernel_bias()
        self.reparam_conv = create_conv2d(
            self.in_chs,
            self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = eq_k
        self.reparam_conv.bias.data = eq_b
        self.__delattr__("large_conv")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


def convolutional_stem(
    in_chs: int,
    out_chs: int,
    act_layer: Type[nn.Module] = nn.GELU,
    inference_mode: bool = False,
    use_scale_branch: bool = True,
    device=None,
    dtype=None,
) -> nn.Sequential:
    dd = {"device": device, "dtype": dtype}
    return nn.Sequential(
        MobileOneBlock(
            in_chs=in_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=use_scale_branch,
            **dd,
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            group_size=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=use_scale_branch,
            **dd,
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=1,
            stride=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=use_scale_branch,
            **dd,
        ),
    )


class PatchEmbed(nn.Module):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_chs: int,
        embed_dim: int,
        act_layer: Type[nn.Module] = nn.GELU,
        lkc_use_act: bool = False,
        use_se: bool = False,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_chs=in_chs,
                out_chs=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                group_size=1,
                small_kernel=3,
                use_se=use_se,
                act_layer=act_layer if lkc_use_act else None,
                inference_mode=inference_mode,
                **dd,
            ),
            MobileOneBlock(
                in_chs=embed_dim,
                out_chs=embed_dim,
                kernel_size=1,
                stride=1,
                use_se=False,
                act_layer=act_layer,
                inference_mode=inference_mode,
                **dd,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LayerScale2d(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim, 1, 1, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RepMixer(nn.Module):
    """Reparameterizable token mixer used in FastViT-T12."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        layer_scale_init_value: Optional[float] = 1e-5,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
                **dd,
            )
        else:
            self.reparam_conv = None
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
                **dd,
            )
            self.mixer = MobileOneBlock(dim, dim, kernel_size, group_size=1, use_act=False, **dd)
            self.layer_scale = (
                LayerScale2d(dim, layer_scale_init_value, **dd)
                if layer_scale_init_value is not None
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return self.reparam_conv(x)
        return x + self.layer_scale(self.mixer(x) - self.norm(x))

    def reparameterize(self) -> None:
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if isinstance(self.layer_scale, LayerScale2d):
            w = self.mixer.id_tensor + self.layer_scale.gamma.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale.gamma) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = create_conv2d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for name, param in self.named_parameters():
            if "reparam_conv" in name:
                continue
            param.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        self.__delattr__("layer_scale")
        self.inference_mode = True


class ConvMlp(nn.Module):
    """Convolutional FFN module."""

    def __init__(
        self,
        in_chs: int,
        hidden_channels: Optional[int] = None,
        out_chs: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        out_chs = out_chs or in_chs
        hidden_channels = hidden_channels or in_chs
        self.conv = ConvNormAct(in_chs, out_chs, kernel_size=7, groups=in_chs, apply_act=False, **dd)
        self.fc1 = nn.Conv2d(in_chs, hidden_channels, kernel_size=1, **dd)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_chs, kernel_size=1, **dd)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixerBlock(nn.Module):
    """MetaFormer block with RepMixer token mixing."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Module] = nn.GELU,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
            **dd,
        )
        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            **dd,
        )
        self.layer_scale = (
            LayerScale2d(dim, layer_scale_init_value, **dd)
            if layer_scale_init_value is not None
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        x = x + self.drop_path(self.layer_scale(self.mlp(x)))
        return x


class FastVitStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        downsample: bool = True,
        se_downsample: bool = False,
        down_patch_size: int = 7,
        down_stride: int = 2,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Module] = nn.GELU,
        proj_drop_rate: float = 0.0,
        drop_path_rate: list[float] | float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        lkc_use_act: bool = False,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.grad_checkpointing = False

        if downsample:
            self.downsample = PatchEmbed(
                patch_size=down_patch_size,
                stride=down_stride,
                in_chs=dim,
                embed_dim=dim_out,
                use_se=se_downsample,
                act_layer=act_layer,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
                **dd,
            )
        else:
            if dim != dim_out:
                raise ValueError("dim must equal dim_out when downsample is False.")
            self.downsample = nn.Identity()

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * depth

        self.blocks = nn.Sequential(
            *[
                RepMixerBlock(
                    dim_out,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                    **dd,
                )
                for block_idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class FastVit(nn.Module):
    """FastViT classification backbone.

    The T12 distilled ImageNet encoder uses only RepMixer stages. When
    ``num_classes=0`` the classifier is an identity and ``forward`` returns a
    global-pooled image embedding.
    """

    def __init__(
        self,
        in_chans: int = 3,
        layers: tuple[int, ...] = (2, 2, 6, 2),
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        mlp_ratios: tuple[float, ...] = (3, 3, 3, 3),
        downsamples: tuple[bool, ...] = (False, True, True, True),
        se_downsamples: tuple[bool, ...] = (False, False, False, False),
        repmixer_kernel_size: int = 3,
        num_classes: int = 1000,
        down_patch_size: int = 7,
        down_stride: int = 2,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        lkc_use_act: bool = False,
        stem_use_scale_branch: bool = True,
        cls_ratio: float = 2.0,
        global_pool: str = "avg",
        act_layer: Type[nn.Module] = nn.GELU,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_info: list[dict[str, Any]] = []

        self.stem = convolutional_stem(
            in_chans,
            embed_dims[0],
            act_layer,
            inference_mode,
            use_scale_branch=stem_use_scale_branch,
            **dd,
        )

        prev_dim = embed_dims[0]
        scale = 1
        dpr = calculate_drop_path_rates(drop_path_rate, list(layers), stagewise=True)
        stages = []
        for i, depth in enumerate(layers):
            downsample = downsamples[i] or prev_dim != embed_dims[i]
            stage = FastVitStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=depth,
                downsample=downsample,
                se_downsample=se_downsamples[i],
                down_patch_size=down_patch_size,
                down_stride=down_stride,
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                proj_drop_rate=proj_drop_rate,
                drop_path_rate=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
                **dd,
            )
            stages.append(stage)
            prev_dim = embed_dims[i]
            if downsample:
                scale *= 2
            self.feature_info.append(dict(num_chs=prev_dim, reduction=4 * scale, module=f"stages.{i}"))

        self.stages = nn.Sequential(*stages)
        self.num_stages = len(self.stages)
        final_features = int(embed_dims[-1] * cls_ratio)
        self.num_features = self.head_hidden_size = final_features
        self.final_conv = MobileOneBlock(
            in_chs=embed_dims[-1],
            out_chs=final_features,
            kernel_size=3,
            stride=1,
            group_size=1,
            inference_mode=inference_mode,
            use_se=True,
            act_layer=act_layer,
            num_conv_branches=1,
            **dd,
        )
        self.head = ClassifierHead(
            final_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            **dd,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> dict[str, Any]:
        return {
            "stem": r"^stem",
            "blocks": r"^stages\.(\d+)" if coarse else [
                (r"^stages\.(\d+).downsample", (0,)),
                (r"^stages\.(\d+)\.\w+\.(\d+)", None),
            ],
        }

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        for stage in self.stages:
            stage.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class FastVitT12AppleDistIn1kEncoder(nn.Module):
    """Image encoder returning the FastViT-T12 1024-d embedding."""

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        reparameterize: bool = False,
        strict: bool = True,
        in_chans: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs.pop("num_classes", None)
        self.model = fastvit_t12_apple_dist_in1k(
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            reparameterize=reparameterize,
            strict=strict,
            in_chans=in_chans,
            num_classes=0,
            **kwargs,
        )
        self.num_features = self.model.num_features
        self.head_hidden_size = self.model.head_hidden_size
        self.default_cfg = self.model.default_cfg
        self.pretrained_cfg = self.model.pretrained_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


def _model_kwargs(**kwargs: Any) -> dict[str, Any]:
    return {
        "layers": (2, 2, 6, 2),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (3, 3, 3, 3),
        **kwargs,
    }


def reparameterize_model(module: nn.Module) -> int:
    """Fuse all child modules that expose a timm-style ``reparameterize`` method."""
    fused = 0
    for child in reversed(list(module.modules())):
        reparameterize = getattr(child, "reparameterize", None)
        if callable(reparameterize):
            reparameterize()
            fused += 1
    return fused


def fastvit_t12_apple_dist_in1k(
    pretrained: bool = False,
    checkpoint_path: str | Path | None = None,
    reparameterize: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> FastVit:
    """Build the local FastViT-T12 Apple distilled ImageNet model.

    Args:
        pretrained: Load local weights when True.
        checkpoint_path: Optional file or directory containing local weights.
        reparameterize: Fuse structural reparameterization branches after loading.
        strict: Strictness passed to ``load_state_dict`` after classifier filtering.
        **kwargs: Model overrides such as ``num_classes``, ``in_chans`` or ``drop_rate``.
    """
    if kwargs.pop("features_only", False):
        raise NotImplementedError("This extracted file defines the classification/embedding model only.")

    model = FastVit(**_model_kwargs(**kwargs))
    cfg = deepcopy(DEFAULT_CFG)
    model.default_cfg = cfg
    model.pretrained_cfg = cfg

    if pretrained:
        load_local_pretrained(model, checkpoint_path=checkpoint_path, strict=strict)
    if reparameterize:
        reparameterize_model(model)
    return model


def _resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if path.is_file():
            return path
        if path.is_dir():
            found = _find_checkpoint_in_dir(path)
            if found is not None:
                return found
        raise FileNotFoundError(f"No checkpoint file found at {path}")

    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        project_root / "checkpoint" / "fastvit_t12_apple_dist_in1k",
        Path(__file__).resolve().parent,
    )
    for directory in candidates:
        found = _find_checkpoint_in_dir(directory)
        if found is not None:
            return found
    raise FileNotFoundError(
        "No local checkpoint found. Expected model.safetensors or pytorch_model.bin "
        "under checkpoint/fastvit_t12_apple_dist_in1k."
    )


def _find_checkpoint_in_dir(directory: Path) -> Path | None:
    for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pth", "model.pth", "model.pt"):
        path = directory / name
        if path.is_file():
            return path
    return None


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)!r}.")

    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            name = key[7:] if key.startswith("module.") else key
            state_dict[name] = value
    if not state_dict:
        raise ValueError("Checkpoint does not contain tensor weights.")
    return state_dict


def load_state_dict_from_path(path: str | Path, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load a local PyTorch or safetensors checkpoint without timm."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires safetensors. "
                "Install it or pass a .bin/.pth checkpoint_path."
            ) from exc
        return dict(load_file(str(path), device=str(device)))

    checkpoint = torch.load(path, map_location=device)
    return _unwrap_state_dict(checkpoint)


def _filter_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key not in model_state:
            if key.startswith("head.fc.") and getattr(model, "num_classes", None) == 0:
                continue
            filtered[key] = value
            continue

        model_value = model_state[key]
        if tuple(value.shape) != tuple(model_value.shape):
            if key.startswith("head.fc."):
                continue
        filtered[key] = value
    return filtered


def load_local_pretrained(
    model: nn.Module,
    checkpoint_path: str | Path | None = None,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load local FastViT-T12 weights into ``model``."""
    path = _resolve_checkpoint_path(checkpoint_path)
    state_dict = load_state_dict_from_path(path, device=device)
    state_dict = _filter_state_dict_for_model(state_dict, model)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint": str(path),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def create_encoder(
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
    reparameterize: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> FastVitT12AppleDistIn1kEncoder:
    """Compatibility helper for constructing the embedding encoder."""
    return FastVitT12AppleDistIn1kEncoder(
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        reparameterize=reparameterize,
        strict=strict,
        **kwargs,
    )


def create_model(
    pretrained: bool = False,
    checkpoint_path: str | Path | None = None,
    reparameterize: bool = False,
    strict: bool = True,
    **kwargs: Any,
) -> FastVit:
    """Compatibility alias for constructing the extracted model."""
    return fastvit_t12_apple_dist_in1k(
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        reparameterize=reparameterize,
        strict=strict,
        **kwargs,
    )


__all__ = [
    "ARCHITECTURE",
    "DEFAULT_CFG",
    "FEATURE_DIM",
    "HF_HUB_ID",
    "MODEL_NAME",
    "FastVit",
    "FastVitT12AppleDistIn1kEncoder",
    "create_encoder",
    "create_model",
    "fastvit_t12_apple_dist_in1k",
    "load_local_pretrained",
    "reparameterize_model",
]
