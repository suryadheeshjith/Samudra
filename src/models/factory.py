from typing import Type

import torch.nn as nn

from models.activations import CappedGELU, ReLU
from models.blocks import (
    AvgPool,
    BilinearUpsample,
    ConvNeXtBlock,
    MaxPool,
    TransposedConvUpsample,
)

BLOCK_REGISTRY = {
    "conv_next_block": ConvNeXtBlock,
}

DOWNSAMPLE_REGISTRY = {
    "avg_pool": AvgPool,
    "max_pool": MaxPool,
}

UPSAMPLE_REGISTRY = {
    "bilinear_upsample": BilinearUpsample,
    "transposed_conv": TransposedConvUpsample,
}

ACTIVATION_REGISTRY = {
    "relu": ReLU,
    "capped_gelu": CappedGELU,
}


def create_block(block_type: str, **kwargs) -> nn.Module:
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}")
    return BLOCK_REGISTRY[block_type](**kwargs)


def create_downsample(block_type: str, **kwargs) -> nn.Module:
    if block_type not in DOWNSAMPLE_REGISTRY:
        raise ValueError(f"Unknown downsample type: {block_type}")
    return DOWNSAMPLE_REGISTRY[block_type](**kwargs)


def create_upsample(block_type: str, **kwargs) -> nn.Module:
    if block_type not in UPSAMPLE_REGISTRY:
        raise ValueError(f"Unknown upsample type: {block_type}")
    return UPSAMPLE_REGISTRY[block_type](**kwargs)


def get_activation_cl(activation_type: str) -> Type[nn.Module]:
    if activation_type not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation type: {activation_type}")
    return ACTIVATION_REGISTRY[activation_type]
