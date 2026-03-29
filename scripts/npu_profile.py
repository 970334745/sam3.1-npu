#!/usr/bin/env python3
"""
SAM 3.1 NPU Performance Profiler

Profiles the SAM 3.1 model on Ascend NPU, identifies CPU fallback operations,
and measures NPU vs CPU time distribution.

CRITICAL FINDING: torch_npu makes tensor.is_cuda return True for NPU tensors,
causing the connected_components and NMS code to incorrectly enter Triton paths.
This script calls the correct fallback functions directly.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch_npu  # noqa: F401

from sam3.device_utils import (
    get_accelerator,
    get_device,
    synchronize,
    empty_cache,
    max_memory_allocated,
    reset_peak_memory_stats,
)

DEVICE = get_device()
ACC = get_accelerator()
print(f"[INFO] Accelerator: {ACC}, Device: {DEVICE}")


def npu_timer():
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    return start, end


def measure_npu_time(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    synchronize()
    times = []
    for _ in range(repeats):
        start, end = npu_timer()
        start.record()
        fn()
        end.record()
        synchronize()
        times.append(start.elapsed_time(end))
    avg = sum(times) / len(times)
    return avg, min(times), max(times)


def measure_wall_time(fn, warmup=3, repeats=10):
    """Measure with wall-clock time (for CPU-bound ops)."""
    for _ in range(warmup):
        fn()
    synchronize()
    times = []
    for _ in range(repeats):
        synchronize()
        t0 = time.perf_counter()
        fn()
        synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    avg = sum(times) / len(times)
    return avg, min(times), max(times)


def cpu_profiler_context():
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    )


# ============================================================================
# Part 1: Profile individual CPU-fallback operations
# ============================================================================

def profile_sdpa_fallback():
    """Profile the SDPA fallback (replacing Flash Attention 3)."""
    from sam3.perflib.fa3 import _sdpa_fallback, flash_attn_func

    print("\n" + "=" * 70)
    print("[1/5] SDPA Fallback (替代 Flash Attention 3)")
    print("=" * 70)
    print("  路径: sam3/perflib/fa3.py -> _sdpa_fallback()")
    print("  说明: FA3 不可用时, 使用 PyTorch SDPA 替代, 在 NPU 上通过")
    print("        npu_fusion_attention 算子执行, 无 CPU 回退")

    batch, seq_len, num_heads, head_dim = 1, 4096, 16, 64
    q = torch.randn(batch, seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float32)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float32)
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float32)

    avg, mn, mx = measure_npu_time(lambda: flash_attn_func(q, k, v))
    print(f"  Timing (1,4096,16,64): avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

    with cpu_profiler_context() as prof:
        with torch.profiler.record_function("SDPA_FALLBACK"):
            out = flash_attn_func(q, k, v)
            synchronize()
    print(f"  Output: shape={out.shape}, device={out.device}, dtype={out.dtype}")
    _print_top_events(prof.key_averages(), "SDPA Fallback", top_n=15)

    print("\n  Scaling test:")
    for sl in [1024, 2048, 4096, 8192]:
        q2 = torch.randn(1, sl, 16, 64, device=DEVICE, dtype=torch.float32)
        k2 = torch.randn_like(q2); v2 = torch.randn_like(q2)
        avg2, _, _ = measure_npu_time(lambda: flash_attn_func(q2, k2, v2), warmup=2, repeats=5)
        print(f"    seq_len={sl:>5}: {avg2:>8.2f} ms")
    return avg


def profile_connected_components():
    """Profile the connected components CPU fallback.
    
    BUG: is_cuda returns True on NPU, so the original code tries Triton.
    We directly call the CPU fallback.
    """
    from sam3.perflib.connected_components import connected_components_cpu

    print("\n" + "=" * 70)
    print("[2/5] Connected Components (CPU Fallback via skimage)")
    print("=" * 70)
    print("  路径: sam3/perflib/connected_components.py -> connected_components_cpu()")
    print("  说明: NPU 上无 cc_torch/Triton, 回退到 CPU skimage.measure.label")
    print("  ⚠ BUG: is_cuda=True on NPU causes Triton path (needs fix)")

    batch, h, w = 4, 256, 256
    masks_4d = (torch.rand(batch, 1, h, w, device=DEVICE) > 0.5).to(torch.uint8)

    def run_cc():
        return connected_components_cpu(masks_4d)

    avg, mn, mx = measure_wall_time(run_cc, warmup=2, repeats=5)
    print(f"  Timing (4x1x256x256): avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

    with cpu_profiler_context() as prof:
        with torch.profiler.record_function("CONNECTED_COMPONENTS_CPU"):
            labels, counts = connected_components_cpu(masks_4d)
            synchronize()
    print(f"  Labels: shape={labels.shape}, device={labels.device}")
    _print_top_events(prof.key_averages(), "Connected Components (CPU)", top_n=15)

    print("\n  Scaling test:")
    for sz in [128, 256, 512, 1024]:
        m = (torch.rand(4, 1, sz, sz, device=DEVICE) > 0.5).to(torch.uint8)
        avg2, _, _ = measure_wall_time(lambda: connected_components_cpu(m), warmup=1, repeats=3)
        print(f"    {sz}x{sz} B=4: {avg2:>8.2f} ms")
    return avg


def profile_nms():
    """Profile the NMS CPU fallback.
    
    BUG: is_cuda returns True on NPU, so generic_nms tries Triton.
    We directly call the CPU fallback.
    """
    from sam3.perflib.nms import generic_nms_cpu
    from sam3.perflib.masks_ops import mask_iou

    print("\n" + "=" * 70)
    print("[3/5] NMS (CPU Fallback via numpy)")
    print("=" * 70)
    print("  路径: sam3/perflib/nms.py -> generic_nms_cpu()")
    print("  说明: NPU 上无 torch_generic_nms/Triton, 回退到 CPU numpy 循环")
    print("  ⚠ BUG: is_cuda=True on NPU causes Triton path (needs fix)")

    num_det, h, w = 50, 256, 256
    masks = (torch.rand(num_det, h, w, device=DEVICE) > 0.7).float()
    scores = torch.rand(num_det, device=DEVICE)
    ious = mask_iou(masks > 0, masks > 0)
    synchronize()

    def run_nms():
        return generic_nms_cpu(ious, scores, iou_threshold=0.5)

    avg, mn, mx = measure_wall_time(run_nms, warmup=2, repeats=5)
    print(f"  Timing (50 dets): avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

    with cpu_profiler_context() as prof:
        with torch.profiler.record_function("NMS_CPU"):
            kept = generic_nms_cpu(ious, scores, iou_threshold=0.5)
            synchronize()
    print(f"  Kept: {kept.shape}, device={kept.device}")
    _print_top_events(prof.key_averages(), "NMS (CPU)", top_n=15)

    print("\n  Scaling test:")
    for nd in [10, 50, 100, 200]:
        m = (torch.rand(nd, 128, 128, device=DEVICE) > 0.7).float()
        s = torch.rand(nd, device=DEVICE)
        io = mask_iou(m > 0, m > 0)
        synchronize()
        avg2, _, _ = measure_wall_time(lambda: generic_nms_cpu(io, s, 0.5), warmup=1, repeats=3)
        print(f"    ndets={nd:>3}: {avg2:>8.2f} ms")
    return avg


def profile_nms_triton_fallback():
    """Profile the Triton NMS CPU fallback (in triton/nms.py)."""
    from sam3.perflib.triton.nms import _nms_cpu_fallback
    from sam3.perflib.masks_ops import mask_iou

    print("\n" + "=" * 70)
    print("[3b] Triton NMS CPU Fallback")
    print("=" * 70)
    print("  路径: sam3/perflib/triton/nms.py -> _nms_cpu_fallback()")

    num_det = 50
    masks = (torch.rand(num_det, 256, 256, device=DEVICE) > 0.7).float()
    scores = torch.rand(num_det, device=DEVICE)
    ious = mask_iou(masks > 0, masks > 0)
    synchronize()

    avg, mn, mx = measure_wall_time(
        lambda: _nms_cpu_fallback(ious, scores, 0.5), warmup=2, repeats=5
    )
    print(f"  Timing (50 dets): avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")
    return avg


def profile_edt():
    """Profile EDT (Euclidean Distance Transform) - Triton kernel."""
    print("\n" + "=" * 70)
    print("[4/5] EDT (Euclidean Distance Transform - Triton)")
    print("=" * 70)
    print("  路径: sam3/model/edt.py -> edt_triton()")
    print("  说明: Triton kernel 需要 CUDA 后端, NPU 上不可用")

    try:
        from sam3.model.edt import edt_triton
        data = (torch.rand(4, 256, 256, device=DEVICE) > 0.5).float()
        avg, mn, mx = measure_npu_time(lambda: edt_triton(data), warmup=2, repeats=5)
        print(f"  Timing: avg={avg:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")
        return avg
    except Exception as e:
        print(f"  [FAILED] {type(e).__name__}: {e}")
        print("  Triton 需要 CUDA 后端; NPU 无 Triton 支持")

        # Profile a pure-PyTorch EDT approximation for reference
        print("\n  [REF] cv2.distanceTransform CPU 参考时间:")
        try:
            import cv2
            import numpy as np
            data_np = (np.random.rand(4, 256, 256) > 0.5).astype(np.uint8)
            # warmup
            for b in range(4):
                cv2.distanceTransform(data_np[b], cv2.DIST_L2, 0)
            t0 = time.perf_counter()
            for _ in range(5):
                for b in range(4):
                    cv2.distanceTransform(data_np[b], cv2.DIST_L2, 0)
            t1 = time.perf_counter()
            cv2_ms = (t1 - t0) / 5 * 1000
            print(f"  cv2 EDT (4x256x256): {cv2_ms:.2f} ms")
        except ImportError:
            print("  cv2 not available")
        return None


def profile_compile():
    """Check torch.compile status."""
    print("\n" + "=" * 70)
    print("[5/5] torch.compile 状态")
    print("=" * 70)
    print("  路径: sam3/perflib/compile.py -> compile_wrapper()")

    from sam3.device_utils import supports_torch_compile
    enabled = supports_torch_compile()
    print(f"  supports_torch_compile(): {enabled}")
    print(f"  SAM3_NPU_ENABLE_COMPILE env: {os.environ.get('SAM3_NPU_ENABLE_COMPILE', '0')}")
    if not enabled:
        print("  torch.compile 在 NPU 上默认禁用（无 Inductor 后端）")
        print("  性能影响: 无法使用 operator fusion / CUDA graph 优化")


# ============================================================================
# Part 2: Profile the full model backbone forward pass
# ============================================================================

def profile_model_backbone():
    """Profile the model's backbone (image encoder) forward pass."""
    print("\n" + "=" * 70)
    print("Model Backbone (Image Encoder) Forward Pass")
    print("=" * 70)

    ckpt = os.path.join(os.path.dirname(__file__), "..", "weights", "sam3.1_multiplex.pt")
    if not os.path.exists(ckpt):
        print(f"  [WARN] Checkpoint not found: {ckpt}")
        return None, None

    from sam3.model_builder import build_sam3_multiplex_video_model
    from sam3.device_utils import model_to_device

    print("  Building model...")
    model = build_sam3_multiplex_video_model(
        checkpoint_path=ckpt,
        load_from_HF=False,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        strict_state_dict_loading=False,
    )
    model_to_device(model, eval_mode=True)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {param_count:.1f}M")
    print(f"  Device: {next(model.parameters()).device}")

    if model.backbone is None:
        print("  [SKIP] backbone is None")
        return None, model

    img = torch.randn(1, 3, 1008, 1008, device=DEVICE, dtype=torch.float32)

    print("  Warming up backbone (3 iterations)...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.backbone(img)
            synchronize()

    empty_cache()
    reset_peak_memory_stats()

    avg, mn, mx = measure_npu_time(
        lambda: model.backbone(img).__class__, warmup=0, repeats=5
    )

    # Actually measure with output materialized
    with torch.no_grad():
        synchronize()
        start, end = npu_timer()
        start.record()
        backbone_out = model.backbone(img)
        end.record()
        synchronize()
        backbone_ms = start.elapsed_time(end)

    print(f"  Backbone timing: {backbone_ms:.2f} ms")

    # Profile op breakdown
    print("  Profiling op breakdown...")
    with torch.no_grad():
        with cpu_profiler_context() as prof:
            with torch.profiler.record_function("BACKBONE_FORWARD"):
                _ = model.backbone(img)
                synchronize()

    _print_top_events(prof.key_averages(), "Backbone Forward", top_n=25)

    peak_mem = max_memory_allocated() / (1024 ** 2)
    print(f"\n  Peak memory: {peak_mem:.1f} MB")

    return prof, model


# ============================================================================
# Utilities
# ============================================================================

KNOWN_CPU_FALLBACK_PATTERNS = [
    "numpy", "skimage", "label", "connected_component",
    "nms_cpu", "generic_nms_cpu", "cpu_fallback",
    "scipy", "cv2", "distanceTransform",
    "aten::item", "aten::numpy", "aten::_local_scalar_dense",
]


def _print_top_events(events, title, top_n=20):
    print(f"\n  === Top {top_n} Operations ({title}) ===")
    all_ops = []
    for evt in events:
        name = evt.key
        if any(skip in name for skip in ["profiler", "Profiler"]):
            continue
        cpu_time_ms = evt.cpu_time_total / 1000
        if cpu_time_ms < 0.001:
            continue
        is_fallback = any(p.lower() in name.lower() for p in KNOWN_CPU_FALLBACK_PATTERNS)
        all_ops.append((name, cpu_time_ms, evt.count, is_fallback))

    all_ops.sort(key=lambda x: x[1], reverse=True)
    total_time = sum(x[1] for x in all_ops)

    copy_ops = [op for op in all_ops if "copy_" in op[0].lower() or "aten::to" in op[0].lower()]
    copy_time = sum(x[1] for x in copy_ops)
    fallback_ops = [op for op in all_ops if op[3]]
    fallback_time = sum(x[1] for x in fallback_ops)

    print(f"  Total CPU-side time (all ops): {total_time:.2f} ms")
    if copy_time > 0.01:
        print(f"  D2H/H2D copy overhead: {copy_time:.2f} ms")
    if fallback_ops:
        print(f"  CPU fallback overhead: {fallback_time:.2f} ms")

    print(f"\n  {'#':<3} {'Operation':<55} {'CPU(ms)':>10} {'Count':>6} {'Flag':>8}")
    print("  " + "-" * 84)
    for i, (name, cpu_ms, count, is_fb) in enumerate(all_ops[:top_n]):
        flag = "⚠ FB" if is_fb else ""
        if "copy_" in name.lower() or "aten::to" in name.lower():
            flag = "COPY"
        print(f"  {i+1:<3} {name[:55]:<55} {cpu_ms:>10.3f} {count:>6} {flag:>8}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print(" SAM 3.1 NPU Performance Profiler")
    print(f" Device: {DEVICE} | Accelerator: {ACC}")
    print(f" torch: {torch.__version__} | torch_npu: {torch_npu.__version__}")
    print("=" * 70)

    # Verify the is_cuda bug
    x_test = torch.randn(2, 2, device=DEVICE)
    print(f"\n[BUG CHECK] NPU tensor.is_cuda = {x_test.is_cuda}")
    print(f"[BUG CHECK] NPU tensor.is_npu = {x_test.is_npu}")
    print(f"[BUG CHECK] NPU tensor.device.type = {x_test.device.type}")
    if x_test.is_cuda:
        print("[BUG] ⚠ is_cuda=True on NPU! This causes Triton path bugs.")
    del x_test

    # Warmup
    print("\n[INIT] Warming up NPU...")
    x = torch.randn(256, 256, device=DEVICE)
    _ = torch.mm(x, x)
    synchronize()
    del x
    print("[INIT] Done.\n")

    # ---- Part 1: Individual ops ----
    print("#" * 70)
    print("# PART 1: Individual Operation Profiles")
    print("#" * 70)

    sdpa_time = profile_sdpa_fallback()
    cc_time = profile_connected_components()
    nms_time = profile_nms()
    nms_triton_fb_time = profile_nms_triton_fallback()
    edt_time = profile_edt()
    profile_compile()

    # ---- Part 2: Model backbone ----
    print("\n\n" + "#" * 70)
    print("# PART 2: Model Backbone Profiling")
    print("#" * 70)
    model = None
    try:
        result = profile_model_backbone()
        if result is not None:
            _, model = result
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

    # ---- Summary ----
    print("\n\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    edt_str = "N/A (Triton不可用)" if edt_time is None else f"{edt_time:.2f} ms"

    print(f"""
  {'='*65}
  SAM 3.1 NPU CPU 回退性能分析汇总
  {'='*65}

  1. SDPA Fallback (替代 FA3)
     时间: {sdpa_time:.2f} ms (seq=4096)
     运行位置: NPU (通过 npu_fusion_attention)
     性能损失: 低 - SDPA 原生支持 NPU, 但无 FP8 优化

  2. Connected Components
     时间: {cc_time:.2f} ms (4x256x256)
     运行位置: CPU (skimage.measure.label + numpy)
     性能损失: 高 - NPU->CPU->NPU 数据搬运 + 串行 CPU 计算
     ⚠ BUG: is_cuda=True 导致代码尝试 Triton 路径

  3. NMS (Non-Maximum Suppression)
     时间: {nms_time:.2f} ms (50 detections)
     运行位置: CPU (numpy 循环)
     性能损失: 中 - 数据量小时可接受, 检测数多时瓶颈显著
     ⚠ BUG: is_cuda=True 导致代码尝试 Triton 路径

  4. EDT (Euclidean Distance Transform)
     时间: {edt_str}
     运行位置: 不可用 (Triton 需要 CUDA)
     性能损失: 高 - 完全不可用, 需要纯 PyTorch 重写

  5. torch.compile
     状态: 已禁用
     性能损失: 中 - 无 operator fusion / CUDA graph
     
  {'='*65}
  关键发现: torch_npu 使 tensor.is_cuda=True,
  导致 connected_components 和 NMS 错误进入 Triton 路径!
  需要修复: 用 device.type != 'cuda' 替代 is_cuda 检查
  {'='*65}
""")
    print("Profiling complete.")


if __name__ == "__main__":
    main()
