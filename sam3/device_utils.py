# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Ascend NPU adaptation layer for SAM 3.1

import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NPU availability detection
# ---------------------------------------------------------------------------
_npu_available = False
try:
    import torch_npu  # noqa: F401

    _npu_available = torch.npu.is_available()
except (ImportError, AttributeError):
    pass


def is_npu_available() -> bool:
    return _npu_available


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def get_accelerator() -> str:
    """Return the best available accelerator: 'npu', 'cuda', or 'cpu'."""
    if is_npu_available():
        return "npu"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def get_device(index: int = 0) -> torch.device:
    """Return a torch.device for the best available accelerator."""
    acc = get_accelerator()
    if acc == "cpu":
        return torch.device("cpu")
    return torch.device(f"{acc}:{index}")


def get_default_device_string() -> str:
    """Return 'npu', 'cuda', or 'cpu'."""
    return get_accelerator()


# ---------------------------------------------------------------------------
# Device capability helpers (replaces torch.cuda.get_device_properties)
# ---------------------------------------------------------------------------
def get_device_capability(device_index: int = 0):
    """
    Return (major, minor) for CUDA, or a synthetic capability for NPU.
    NPU is treated as equivalent to Ampere+ (8, 0) for feature gating.
    """
    acc = get_accelerator()
    if acc == "cuda":
        props = torch.cuda.get_device_properties(device_index)
        return props.major, props.minor
    if acc == "npu":
        return 8, 0  # treat NPU as Ampere-equivalent for TF32 / flash-attn gating
    return 0, 0


def is_ampere_or_newer(device_index: int = 0) -> bool:
    major, _ = get_device_capability(device_index)
    return major >= 8


# ---------------------------------------------------------------------------
# TF32 setup
# ---------------------------------------------------------------------------
def setup_tf32() -> None:
    """Enable TF32 on Ampere+ CUDA GPUs. No-op on NPU / CPU."""
    acc = get_accelerator()
    if acc == "cuda" and is_ampere_or_newer():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif acc == "npu":
        pass  # NPU handles precision internally


# ---------------------------------------------------------------------------
# Autocast context helpers
# ---------------------------------------------------------------------------
def get_autocast_device_type() -> str:
    """Return the device_type string for torch.autocast."""
    acc = get_accelerator()
    if acc == "npu":
        return "npu"
    if acc == "cuda":
        return "cuda"
    return "cpu"


def autocast_context(dtype=torch.bfloat16, enabled=True):
    """Return a torch.autocast context for the current accelerator."""
    device_type = get_autocast_device_type()
    if device_type == "cpu" and dtype == torch.bfloat16:
        dtype = torch.bfloat16  # CPU supports bfloat16 autocast
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


# ---------------------------------------------------------------------------
# Distributed backend helpers
# ---------------------------------------------------------------------------
def get_dist_backend() -> str:
    """Return the best distributed backend for the current accelerator."""
    acc = get_accelerator()
    if acc == "npu":
        return "hccl"
    if acc == "cuda":
        return "nccl"
    return "gloo"


# ---------------------------------------------------------------------------
# Device movement helpers
# ---------------------------------------------------------------------------
def to_device(x, device=None, non_blocking=True):
    """Move a tensor or module to the target device (default: best accelerator)."""
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    return x.to(device=device, non_blocking=non_blocking)


def model_to_device(model: torch.nn.Module, device=None, eval_mode=True):
    """Move a model to device and optionally set eval mode."""
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device=device)
    if eval_mode:
        model.eval()
    return model


# ---------------------------------------------------------------------------
# Memory helpers (replaces torch.cuda.* memory API)
# ---------------------------------------------------------------------------
def empty_cache():
    acc = get_accelerator()
    if acc == "cuda":
        torch.cuda.empty_cache()
    elif acc == "npu":
        torch.npu.empty_cache()


def memory_allocated(device=None) -> int:
    acc = get_accelerator()
    if acc == "cuda":
        return torch.cuda.memory_allocated(device)
    if acc == "npu":
        return torch.npu.memory_allocated(device)
    return 0


def memory_reserved(device=None) -> int:
    acc = get_accelerator()
    if acc == "cuda":
        return torch.cuda.memory_reserved(device)
    if acc == "npu":
        return torch.npu.memory_reserved(device)
    return 0


def max_memory_allocated(device=None) -> int:
    acc = get_accelerator()
    if acc == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if acc == "npu":
        return torch.npu.max_memory_allocated(device)
    return 0


def max_memory_reserved(device=None) -> int:
    acc = get_accelerator()
    if acc == "cuda":
        return torch.cuda.max_memory_reserved(device)
    if acc == "npu":
        return torch.npu.max_memory_reserved(device)
    return 0


def reset_peak_memory_stats(device=None):
    acc = get_accelerator()
    if acc == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif acc == "npu":
        torch.npu.reset_peak_memory_stats(device)


def synchronize(device=None):
    acc = get_accelerator()
    if acc == "cuda":
        torch.cuda.synchronize(device)
    elif acc == "npu":
        torch.npu.synchronize(device)


def set_device(device_index: int):
    acc = get_accelerator()
    if acc == "cuda":
        torch.cuda.set_device(device_index)
    elif acc == "npu":
        torch.npu.set_device(device_index)


def current_device() -> int:
    acc = get_accelerator()
    if acc == "cuda":
        return torch.cuda.current_device()
    if acc == "npu":
        return torch.npu.current_device()
    return 0


def device_count() -> int:
    acc = get_accelerator()
    if acc == "cuda":
        return torch.cuda.device_count()
    if acc == "npu":
        return torch.npu.device_count()
    return 0


def manual_seed_all(seed: int):
    acc = get_accelerator()
    if acc == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif acc == "npu":
        torch.npu.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Feature gating: SDPA / Flash Attention settings
# ---------------------------------------------------------------------------
def get_sdpa_settings():
    """
    Return (old_gpu, use_flash_attn, math_kernel_on) in a device-agnostic way.
    On NPU, flash attention is disabled in favor of SDPA math kernel.
    """
    acc = get_accelerator()
    if acc == "cuda":
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
        return old_gpu, use_flash_attn, math_kernel_on
    if acc == "npu":
        return False, False, True  # NPU: use math kernel path (SDPA)
    return True, False, True  # CPU fallback


# ---------------------------------------------------------------------------
# torch.compile compatibility
# ---------------------------------------------------------------------------
def supports_torch_compile() -> bool:
    """Check whether the current backend supports torch.compile."""
    acc = get_accelerator()
    if acc == "npu":
        npu_compile = os.environ.get("SAM3_NPU_ENABLE_COMPILE", "0") == "1"
        return npu_compile
    return acc == "cuda"


def maybe_compile(fn, **kwargs):
    """Conditionally apply torch.compile based on backend support."""
    if supports_torch_compile():
        return torch.compile(fn, **kwargs)
    return fn


# ---------------------------------------------------------------------------
# Tensor device checks (replaces .is_cuda)
# ---------------------------------------------------------------------------
def is_on_accelerator(tensor: torch.Tensor) -> bool:
    """Check if a tensor is on any accelerator (CUDA or NPU)."""
    if tensor.is_cuda:
        return True
    if is_npu_available() and hasattr(tensor, "is_npu") and tensor.is_npu:
        return True
    return False
