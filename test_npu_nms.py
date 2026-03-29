"""
Correctness & performance test for NPU NMS vs CPU NMS.

Usage:
    python test_npu_nms.py
"""

import time

import numpy as np
import torch
import torch_npu  # noqa: F401


# ── Reference: original CPU fallback (numpy) ──────────────────────────────

def nms_old_numpy(ious, scores, iou_threshold=0.5):
    """Original generic_nms_cpu (numpy-based), always pulls to CPU."""
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept_inds = []
    while order.size > 0:
        i = order.item(0)
        kept_inds.append(i)
        inds = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[inds + 1]
    return torch.tensor(kept_inds, dtype=torch.int64, device=scores.device)


# ── New NPU adaptive implementation (mirrors sam3/perflib/nms.py) ─────────

_NPU_NMS_THRESHOLD = 1500


def nms_npu_adaptive(ious, scores, iou_threshold=0.5):
    """Adaptive: small N → numpy CPU, large N → NPU sort + bool mask."""
    n = scores.size(0)
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=scores.device)
    if n >= _NPU_NMS_THRESHOLD:
        return _npu_sort_path(ious, scores, iou_threshold)
    else:
        return _cpu_numpy_path(ious, scores, iou_threshold)


def _cpu_numpy_path(ious, scores, iou_threshold):
    device = scores.device
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order.item(0)
        kept.append(i)
        surviving = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[surviving + 1]
    return torch.tensor(kept, dtype=torch.int64, device=device)


def _npu_sort_path(ious, scores, iou_threshold):
    device = scores.device
    order_npu = torch.argsort(scores, descending=True)
    ious_sorted = ious[order_npu][:, order_npu]
    iou_mask_cpu = (ious_sorted > iou_threshold).cpu()
    order_cpu = order_npu.cpu()
    n = order_cpu.size(0)
    keep_mask = torch.ones(n, dtype=torch.bool)
    for i in range(n - 1):
        if not keep_mask[i]:
            continue
        keep_mask[i + 1:] &= ~iou_mask_cpu[i, i + 1:]
    return order_cpu[keep_mask].to(dtype=torch.int64, device=device)


# ── Test utilities ────────────────────────────────────────────────────────

def make_synthetic_iou_matrix(n, density=0.3, seed=42):
    gen = torch.Generator().manual_seed(seed)
    raw = torch.rand(n, n, generator=gen)
    sym = (raw + raw.T) / 2
    sym.fill_diagonal_(1.0)
    mask = torch.rand(n, n, generator=gen) > density
    sym = sym * mask.float()
    sym = (sym + sym.T) / 2
    sym.fill_diagonal_(1.0)
    return sym.clamp(0, 1)


def check_correctness(n, iou_threshold=0.5, density=0.3, seed=42):
    ious_cpu = make_synthetic_iou_matrix(n, density=density, seed=seed)
    scores_cpu = torch.rand(n)

    ref = nms_old_numpy(ious_cpu, scores_cpu, iou_threshold)
    ref_set = set(ref.tolist())

    ious_npu = ious_cpu.to("npu:0")
    scores_npu = scores_cpu.to("npu:0")
    result = nms_npu_adaptive(ious_npu, scores_npu, iou_threshold)
    result_set = set(result.cpu().tolist())

    ok = ref_set == result_set
    tag = "PASS" if ok else "FAIL"
    print(f"  N={n:>4d} thr={iou_threshold:.1f}  kept={len(result_set):>3d}  {tag}")
    if not ok:
        print(f"    diff: only_ref={sorted(ref_set - result_set)} only_new={sorted(result_set - ref_set)}")
    return ok


def timeit(fn, args, warmup=5, repeats=30, sync_npu=True):
    for _ in range(warmup):
        fn(*args)
        if sync_npu:
            torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
        if sync_npu:
            torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def benchmark(n, iou_threshold=0.5, density=0.3, seed=42):
    ious_cpu = make_synthetic_iou_matrix(n, density=density, seed=seed)
    scores_cpu = torch.rand(n)
    ious_npu = ious_cpu.to("npu:0")
    scores_npu = scores_cpu.to("npu:0")

    t_old = timeit(nms_old_numpy, (ious_npu, scores_npu, iou_threshold))
    t_new = timeit(nms_npu_adaptive, (ious_npu, scores_npu, iou_threshold))

    path = "numpy-cpu" if n < _NPU_NMS_THRESHOLD else "npu-sort"
    speedup = t_old / t_new if t_new > 0 else float("inf")
    delta_pct = (t_old - t_new) / t_old * 100

    print(
        f"  N={n:>5d}  "
        f"old={t_old:7.2f}ms  "
        f"new({path:>9s})={t_new:7.2f}ms  "
        f"{'speedup' if speedup >= 1 else 'slowdown'}={speedup:.2f}x  "
        f"delta={delta_pct:+.1f}%"
    )


def main():
    print("=" * 75)
    print("1. Correctness Verification (NPU adaptive vs CPU reference)")
    print("=" * 75)

    all_pass = True
    for n in [0, 1, 5, 10, 50, 100, 200, 500, 1000, 2000]:
        for thr in [0.3, 0.5, 0.7]:
            if n == 0:
                ious_npu = torch.empty(0, 0).to("npu:0")
                scores_npu = torch.empty(0).to("npu:0")
                result = nms_npu_adaptive(ious_npu, scores_npu, thr)
                ok = result.numel() == 0
                print(f"  N=   0 thr={thr:.1f}  kept=  0  {'PASS' if ok else 'FAIL'}")
                all_pass = all_pass and ok
            else:
                ok = check_correctness(n, iou_threshold=thr, seed=n * 100 + int(thr * 10))
                all_pass = all_pass and ok

    print()
    print("ALL PASSED!" if all_pass else "SOME FAILED!")

    print()
    print("=" * 75)
    print("2. Performance Benchmark (input tensors on NPU)")
    print("   old = original generic_nms_cpu (numpy D2H)")
    print(f"   new = adaptive (numpy-cpu if N<{_NPU_NMS_THRESHOLD}, npu-sort otherwise)")
    print("=" * 75)
    for n in [50, 100, 200, 500, 1000, 2000, 4000]:
        benchmark(n, iou_threshold=0.5)

    print()
    print("=" * 75)
    print("3. Integration test: sam3.perflib.nms.generic_nms on NPU")
    print("=" * 75)
    try:
        import sys
        sys.path.insert(0, ".")
        from sam3.perflib.nms import generic_nms, _is_npu_tensor

        for n in [100, 2000]:
            ious = make_synthetic_iou_matrix(n).to("npu:0")
            scores = torch.rand(n).to("npu:0")
            assert _is_npu_tensor(ious), "_is_npu_tensor returned False"
            result = generic_nms(ious, scores, 0.5)
            ref = nms_old_numpy(ious, scores, 0.5)
            assert set(result.cpu().tolist()) == set(ref.cpu().tolist()), f"mismatch at N={n}"
            print(f"  N={n}: kept {result.size(0)} indices, device={result.device} ✓")
        print("  Integration test PASSED!")
    except Exception as e:
        print(f"  Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
