# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
from contextlib import contextmanager
from functools import wraps

import torch

__all__ = ["retry_if_cuda_oom"]


def _is_accelerator_oom_error(e: RuntimeError) -> bool:
    """Match PyTorch CUDA / Ascend NPU out-of-memory RuntimeError messages."""
    msg = str(e).lower()
    if "out of memory" not in msg:
        return False
    return "cuda" in msg or "npu" in msg


@contextmanager
def _ignore_accelerator_oom():
    """
    A context which ignores accelerator (CUDA / NPU) OOM exceptions from PyTorch.
    """
    try:
        yield
    except RuntimeError as e:
        if _is_accelerator_oom_error(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    a PyTorch CUDA or NPU out-of-memory error.
    It will first retry after calling `sam3.device_utils.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to move them back to the accelerator if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            dt = x.device.type
            like_accelerator_tensor = dt in ("cuda", "npu") and hasattr(x, "to")
        except AttributeError:
            like_accelerator_tensor = False
        if like_accelerator_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_accelerator_oom():
            return func(*args, **kwargs)

        from sam3.device_utils import empty_cache

        empty_cache()
        with _ignore_accelerator_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info(
            "Attempting to copy inputs of {} to CPU due to accelerator OOM".format(
                str(func)
            )
        )
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped
