# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
from sam3.device_utils import get_accelerator

_USE_FUSED_ADDMM = False
try:
    addmm_act_op = torch.ops.aten._addmm_activation
    if get_accelerator() == "cuda":
        _USE_FUSED_ADDMM = True
except (AttributeError, RuntimeError):
    pass


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    bias = linear.bias.detach()
    weight = linear.weight.detach()
    bias = bias.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])

    if _USE_FUSED_ADDMM:
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            y = addmm_act_op(bias, mat1_flat, weight.t(), beta=1, alpha=1, use_gelu=False)
            return y.view(mat1.shape[:-1] + (y.shape[-1],))
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            y = addmm_act_op(bias, mat1_flat, weight.t(), beta=1, alpha=1, use_gelu=True)
            return y.view(mat1.shape[:-1] + (y.shape[-1],))

    # NPU / fallback: separate addmm + activation (no CPU fallback warning)
    y = torch.addmm(bias, mat1_flat, weight.t())
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = torch.nn.functional.relu(y)
    elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = torch.nn.functional.gelu(y)
    else:
        raise ValueError(f"Unexpected activation {activation}")
    return y.view(mat1.shape[:-1] + (y.shape[-1],))
