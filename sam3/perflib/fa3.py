# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging

import torch

logger = logging.getLogger(__name__)

_FA3_AVAILABLE = False
try:
    from flash_attn_interface import flash_attn_func as _fa3_impl

    _FA3_AVAILABLE = True
except ImportError:
    logger.debug("flash_attn_interface not available; FA3 will fall back to SDPA.")


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from flash_attn_interface import flash_attn_func as fa3

    return fa3(q, k, v)


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
    return _sdpa_fallback(q, k, v)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
