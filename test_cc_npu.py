"""
Correctness & performance tests for the NPU-native connected-components
implementation vs. the CPU skimage reference.
"""

import sys
import time

import torch

sys.path.insert(0, ".")

from sam3.perflib.connected_components import (
    connected_components,
    connected_components_cpu,
    connected_components_npu,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _component_sizes(labels: torch.Tensor):
    B = labels.shape[0]
    result = []
    for b in range(B):
        flat = labels[b].view(-1)
        uniq = flat.unique()
        sizes = []
        for u in uniq:
            if u.item() == 0:
                continue
            sizes.append((flat == u).sum().item())
        result.append(sorted(sizes))
    return result


def _equiv_classes_match(lab_a: torch.Tensor, lab_b: torch.Tensor):
    B = lab_a.shape[0]
    for b in range(B):
        fa = lab_a[b].view(-1)
        fb = lab_b[b].view(-1)
        mapping_a2b = {}
        mapping_b2a = {}
        for i in range(fa.numel()):
            a_val = fa[i].item()
            b_val = fb[i].item()
            if a_val == 0 and b_val == 0:
                continue
            if a_val == 0 or b_val == 0:
                return False
            if a_val in mapping_a2b:
                if mapping_a2b[a_val] != b_val:
                    return False
            else:
                mapping_a2b[a_val] = b_val
            if b_val in mapping_b2a:
                if mapping_b2a[b_val] != a_val:
                    return False
            else:
                mapping_b2a[b_val] = a_val
    return True


def _check(name, tensor_npu, tensor_cpu, npu_labels, npu_counts, ref_labels, ref_counts):
    B = tensor_cpu.shape[0]
    npu_labels_cpu = npu_labels.cpu()
    npu_counts_cpu = npu_counts.cpu()

    ref_sizes = _component_sizes(ref_labels.view(B, *ref_labels.shape[-2:]))
    npu_sizes = _component_sizes(npu_labels_cpu.view(B, *npu_labels_cpu.shape[-2:]))
    sizes_ok = ref_sizes == npu_sizes

    ref_flat = ref_labels.view(B, -1)
    npu_flat = npu_labels_cpu.view(B, -1)
    bg_ok = torch.equal(ref_flat == 0, npu_flat == 0)
    equiv_ok = _equiv_classes_match(ref_flat, npu_flat)

    ref_c = ref_counts.view(B, -1)
    npu_c = npu_counts_cpu.view(B, -1)
    counts_ok = torch.equal(ref_c, npu_c)

    ok = sizes_ok and bg_ok and equiv_ok and counts_ok
    status = "PASS" if ok else "FAIL"

    detail = ""
    if not sizes_ok:
        detail += f" sizes"
    if not bg_ok:
        detail += " bg"
    if not equiv_ok:
        detail += " equiv"
    if not counts_ok:
        detail += " counts"
    print(f"  [{status}] {name}{detail}")
    return ok


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def make_test_cases(device_npu):
    cases = []

    t = torch.zeros(1, 1, 6, 6, dtype=torch.uint8)
    t[0, 0, 1:4, 1:4] = 1
    cases.append(("single_block_3x3", t.to(device_npu)))

    t = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
    t[0, 0, 0:3, 0:3] = 1
    t[0, 0, 5:8, 5:8] = 1
    cases.append(("two_rects", t.to(device_npu)))

    t = torch.zeros(1, 1, 5, 5, dtype=torch.uint8)
    for i in range(5):
        t[0, 0, i, i] = 1
    cases.append(("diagonal_line", t.to(device_npu)))

    t = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)
    cases.append(("all_bg", t.to(device_npu)))

    t = torch.ones(1, 1, 4, 4, dtype=torch.uint8)
    cases.append(("all_fg", t.to(device_npu)))

    t = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)
    t[0, 0, 0::2, 0::2] = 1
    t[0, 0, 1::2, 1::2] = 1
    cases.append(("checkerboard", t.to(device_npu)))

    t = torch.zeros(1, 1, 5, 5, dtype=torch.uint8)
    t[0, 0, 0:5, 0] = 1
    t[0, 0, 4, 0:5] = 1
    cases.append(("L_shape", t.to(device_npu)))

    t = torch.zeros(1, 1, 7, 7, dtype=torch.uint8)
    t[0, 0, 0, :] = 1
    t[0, 0, 0:4, 6] = 1
    t[0, 0, 3, 2:7] = 1
    t[0, 0, 3:7, 2] = 1
    t[0, 0, 6, 2:7] = 1
    cases.append(("spiral", t.to(device_npu)))

    t = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    t[0, 0, :2, :2] = True
    t[0, 0, 2:, 2:] = True
    cases.append(("bool_input", t.to(device_npu)))

    t = torch.zeros(4, 1, 8, 8, dtype=torch.uint8)
    t[0, 0, 0:3, 0:3] = 1
    t[1, 0, 5:8, 5:8] = 1
    t[2, 0, :, :] = 1
    t[3, 0, 0, 0] = 1
    t[3, 0, 7, 7] = 1
    cases.append(("batch4", t.to(device_npu)))

    torch.manual_seed(42)
    t = (torch.rand(1, 1, 64, 64) > 0.6).to(torch.uint8)
    cases.append(("random_64x64", t.to(device_npu)))

    torch.manual_seed(123)
    t = (torch.rand(1, 1, 128, 128) > 0.5).to(torch.uint8)
    cases.append(("random_128x128", t.to(device_npu)))

    torch.manual_seed(7)
    t = (torch.rand(1, 1, 256, 256) > 0.5).to(torch.uint8)
    cases.append(("random_256x256", t.to(device_npu)))

    return cases


def run_tests(device_npu):
    cases = make_test_cases(device_npu)
    passed = 0
    failed = 0

    print(" -- connected_components_npu (direct) --")
    for name, tensor_npu in cases:
        tensor_cpu = tensor_npu.cpu()
        ref_labels, ref_counts = connected_components_cpu(tensor_cpu)
        npu_labels, npu_counts = connected_components_npu(tensor_npu)
        ok = _check(name, tensor_npu, tensor_cpu, npu_labels, npu_counts, ref_labels, ref_counts)
        passed += ok
        failed += not ok

    # Also test the public dispatcher routes to NPU when on NPU
    print("\n -- connected_components (dispatcher) --")
    for name, tensor_npu in cases:
        tensor_cpu = tensor_npu.cpu()
        ref_labels, ref_counts = connected_components_cpu(tensor_cpu)
        disp_labels, disp_counts = connected_components(tensor_npu)
        ok = _check("disp_" + name, tensor_npu, tensor_cpu,
                     disp_labels, disp_counts, ref_labels, ref_counts)
        passed += ok
        failed += not ok

    return passed, failed


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def benchmark(device_npu, sizes=None, warmup=3, repeats=10):
    if sizes is None:
        sizes = [(1, 64, 64), (1, 128, 128), (1, 256, 256), (1, 512, 512)]

    print("\n--- Performance benchmark ---")
    print(f"{'Shape':>16s}  {'NPU (ms)':>10s}  {'CPU+xfer (ms)':>14s}  {'CPU only (ms)':>14s}  {'vs xfer':>8s}")
    print("-" * 70)

    for B, H, W in sizes:
        torch.manual_seed(0)
        t = (torch.rand(B, 1, H, W) > 0.5).to(torch.uint8)
        t_npu = t.to(device_npu)
        t_cpu = t.cpu()

        # Warmup NPU
        for _ in range(warmup):
            connected_components_npu(t_npu)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()

        # Benchmark NPU (pure on-device)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            connected_components_npu(t_npu)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        npu_ms = (time.perf_counter() - t0) / repeats * 1000

        # Benchmark CPU with round-trip transfer (realistic fallback cost)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            connected_components_cpu(t_npu)  # includes npu->cpu and cpu->npu transfer
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        cpu_xfer_ms = (time.perf_counter() - t0) / repeats * 1000

        # Benchmark pure CPU (no transfer)
        t0 = time.perf_counter()
        for _ in range(repeats):
            connected_components_cpu(t_cpu)
        cpu_ms = (time.perf_counter() - t0) / repeats * 1000

        speedup_xfer = cpu_xfer_ms / max(npu_ms, 1e-6)
        shape_str = f"({B},{H},{W})"
        print(f"{shape_str:>16s}  {npu_ms:>10.2f}  {cpu_xfer_ms:>14.2f}  {cpu_ms:>14.2f}  {speedup_xfer:>7.2f}x")


if __name__ == "__main__":
    try:
        import torch_npu  # noqa: F401
        assert torch.npu.is_available(), "NPU not available"
        device = torch.device("npu:0")
        print(f"Using device: {device}")
    except Exception as e:
        print(f"NPU not available ({e}), falling back to CPU for testing")
        device = torch.device("cpu")

    print("\n=== Correctness Tests ===")
    passed, failed = run_tests(device)
    print(f"\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n*** Some tests FAILED ***")
        sys.exit(1)

    benchmark(device)
    print("\nAll done.")
