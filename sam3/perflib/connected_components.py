# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
import logging
import math

import torch
import torch.nn.functional as F

try:
    from cc_torch import get_connected_components

    HAS_CC_TORCH = True
except ImportError:
    logging.debug(
        "cc_torch not found. Consider installing for better performance. Command line:"
        " pip install git+https://github.com/ronghanghu/cc_torch.git"
    )
    HAS_CC_TORCH = False

logger = logging.getLogger(__name__)


def connected_components_cpu_single(values: torch.Tensor):
    assert values.dim() == 2
    from skimage.measure import label

    labels, num = label(values.cpu().numpy(), return_num=True)
    labels = torch.from_numpy(labels)
    counts = torch.zeros_like(labels)
    for i in range(1, num + 1):
        cur_mask = labels == i
        cur_count = cur_mask.sum()
        counts[cur_mask] = cur_count
    return labels, counts


def connected_components_cpu(input_tensor: torch.Tensor):
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert input_tensor.dim() == 3, (
            "Input tensor must be (B, H, W) or (B, 1, H, W)."
        )

    batch_size = input_tensor.shape[0]
    labels_list = []
    counts_list = []
    for b in range(batch_size):
        labels, counts = connected_components_cpu_single(input_tensor[b])
        labels_list.append(labels)
        counts_list.append(counts)
    labels_tensor = torch.stack(labels_list, dim=0).to(input_tensor.device)
    counts_tensor = torch.stack(counts_list, dim=0).to(input_tensor.device)
    return labels_tensor.view(out_shape), counts_tensor.view(out_shape)


# ---------------------------------------------------------------------------
# NPU-native connected components via iterative label propagation
# ---------------------------------------------------------------------------

def connected_components_npu(input_tensor: torch.Tensor):
    """
    Pure-PyTorch connected components using iterative min-propagation + pointer
    jumping.  Runs entirely on the tensor's device -- no CPU fallback.

    Uses 8-connectivity (matching skimage default and the Triton backend).
    Min-propagation exploits ``min(x) == -max(-x)`` with ``F.max_pool2d``.
    The inner propagation loop stays in negated-float32 space to avoid
    per-step int<->float conversions.

    Args:
        input_tensor: (B, 1, H, W) uint8/bool, non-zero = foreground.
    Returns:
        (labels, counts) -- both ``(B, 1, H, W)`` int64.
    """
    B, C, H, W = input_tensor.shape
    N = H * W
    device = input_tensor.device

    if N == 0:
        z = torch.zeros(B, 1, H, W, device=device, dtype=torch.int64)
        return z, z.clone()

    mask = input_tensor.bool().view(B, H, W)
    mask_flat = mask.view(B, N)

    arange = (
        torch.arange(1, N + 1, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(B, -1)
    )
    labels_flat = torch.where(
        mask_flat, arange, torch.zeros(1, device=device, dtype=torch.int64)
    )

    max_val_f = float(N + 1)
    neg_bg = -max_val_f

    # Number of inner propagation steps between convergence checks.
    # Each step propagates labels by 1 pixel (kernel=3).
    prop_batch = min(max(H, W), 512)
    jump_rounds = max(int(math.ceil(math.log2(max(prop_batch, 2)))), 1)
    max_outer = max(H, W) // prop_batch + 2

    for _outer in range(max_outer):
        prev_flat = labels_flat.clone()

        # Enter negated-float space: fg in [-N, -1], bg = neg_bg.
        neg_f = -(labels_flat.view(B, H, W).float())
        neg_f = torch.where(mask, neg_f, neg_bg)

        # Tight inner loop: only max_pool2d + mask, all in float space.
        for _inner in range(prop_batch):
            pooled = F.max_pool2d(
                neg_f.unsqueeze(1), kernel_size=3, stride=1, padding=1,
            ).squeeze(1)
            neg_f = torch.where(mask, pooled, neg_bg)

        # Back to int64 label space.
        labels_flat = ((-neg_f).long() * mask.long()).view(B, N)

        # Pointer jumping -- compress every label chain to its root.
        for _j in range(jump_rounds):
            idx = (labels_flat - 1).clamp(min=0)
            jumped = torch.gather(labels_flat, 1, idx)
            labels_flat = torch.where(labels_flat > 0, jumped, labels_flat)

        if torch.equal(labels_flat, prev_flat):
            break

    # -- Compute per-component pixel counts via scatter_add_ --
    labels_flat = labels_flat.view(B, N)
    ones = torch.ones(B, N, device=device, dtype=torch.int64)
    label_counts = torch.zeros(B, N + 1, device=device, dtype=torch.int64)
    label_counts.scatter_add_(1, labels_flat, ones)
    counts_flat = torch.gather(label_counts, 1, labels_flat)
    counts_flat = counts_flat * mask_flat.long()

    return labels_flat.view(B, 1, H, W), counts_flat.view(B, 1, H, W)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def connected_components(input_tensor: torch.Tensor):
    """
    Computes connected components labeling on a batch of 2D tensors, using the best available backend.

    Args:
        input_tensor (torch.Tensor): A BxHxW integer tensor or Bx1xHxW. Non-zero values are considered foreground. Bool tensor also accepted

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Both tensors have the same shape as input_tensor.
            - A tensor with dense labels. Background is 0.
            - A tensor with the size of the connected component for each pixel.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)

    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1, (
        "Input tensor must be (B, H, W) or (B, 1, H, W)."
    )

    from sam3.device_utils import is_on_accelerator, is_npu_available

    if is_on_accelerator(input_tensor):
        # Check NPU first: torch_npu may set .is_cuda = True for compat.
        _is_npu = (
            is_npu_available()
            and hasattr(input_tensor, "is_npu")
            and input_tensor.is_npu
        )
        if _is_npu:
            return connected_components_npu(input_tensor)
        if input_tensor.is_cuda and HAS_CC_TORCH:
            return get_connected_components(input_tensor.to(torch.uint8))
        elif input_tensor.is_cuda:
            from sam3.perflib.triton.connected_components import (
                connected_components_triton,
            )
            return connected_components_triton(input_tensor)
        # Other non-CUDA accelerator fallback
        return connected_components_npu(input_tensor)

    # CPU fallback
    return connected_components_cpu(input_tensor)
