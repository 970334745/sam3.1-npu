# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
import math

import torch

logger = logging.getLogger(__name__)

_FA3_AVAILABLE = False
try:
    from flash_attn_interface import flash_attn_func as _fa3_impl

    _FA3_AVAILABLE = True
except ImportError:
    logger.debug("flash_attn_interface not available; FA3 will fall back to SDPA.")

_NPU_AVAILABLE = False
try:
    import torch_npu

    if hasattr(torch_npu, "npu_fusion_attention"):
        _NPU_AVAILABLE = True
        logger.info("NPU fusion attention available; will use npu_fusion_attention.")
except ImportError:
    pass


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from flash_attn_interface import flash_attn_func as fa3

    return fa3(q, k, v)


def _npu_fusion_attention_impl(q, k, v):
    """NPU native fusion attention via torch_npu.npu_fusion_attention.

    q/k/v shape: (batch, seq_len, num_heads, head_dim) — BSND layout.
    """
    orig_dtype = q.dtype
    head_num = q.shape[2]
    head_dim = q.shape[3]
    scale = 1.0 / math.sqrt(head_dim)

    if orig_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = orig_dtype
    else:
        compute_dtype = torch.float16

    q_c = q.to(compute_dtype).contiguous()
    k_c = k.to(compute_dtype).contiguous()
    v_c = v.to(compute_dtype).contiguous()

    out = torch_npu.npu_fusion_attention(
        q_c,
        k_c,
        v_c,
        head_num,
        "BSND",
        scale=scale,
        keep_prob=1.0,
    )[0]

    return out.to(orig_dtype)


def _sdpa_fallback(q, k, v):
    """SDPA-based fallback when FA3 is not available (e.g. on NPU)."""
    # q/k/v shape: (batch, seq_len, num_heads, head_dim) -> SDPA expects (batch, heads, seq, dim)
    q_t = q.transpose(1, 2).to(torch.bfloat16)
    k_t = k.transpose(1, 2).to(torch.bfloat16)
    v_t = v.transpose(1, 2).to(torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.transpose(1, 2).to(q.dtype)


def flash_attn_func(q, k, v):
    if _FA3_AVAILABLE and q.is_cuda:
        dtype = torch.float8_e4m3fn
        return flash_attn_func_op(q.to(dtype), k.to(dtype), v.to(dtype)).to(q.dtype)
    if _NPU_AVAILABLE and q.device.type == "npu":
        return _npu_fusion_attention_impl(q, k, v)
    return _sdpa_fallback(q, k, v)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
