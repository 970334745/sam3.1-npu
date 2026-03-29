"""SAM 3.1 NPU Inference Benchmark — FPS + Memory"""
import gc, os, time, warnings
os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import torch, torch_npu
from PIL import Image

from sam3.device_utils import (
    get_device, get_autocast_device_type,
    memory_allocated, max_memory_allocated,
    synchronize, empty_cache, reset_peak_memory_stats,
)

DEVICE = get_device()
DTYPE = torch.bfloat16
AC = get_autocast_device_type()
CKPT = "/root/autodl-tmp/sam3.1/weights/sam3.1_multiplex.pt"
RES = 1008
WARMUP, RUNS = 3, 10

def mb(x): return x / 1024 / 1024

def bench(fn, warmup=WARMUP, runs=RUNS, label=""):
    gc.collect(); empty_cache(); reset_peak_memory_stats()
    try:
        for _ in range(warmup): fn()
        synchronize(); reset_peak_memory_stats(); synchronize()
        t0 = time.perf_counter()
        for _ in range(runs): fn()
        synchronize(); t1 = time.perf_counter()
        fps = runs / (t1 - t0); lat = (t1 - t0) / runs * 1000; peak = mb(max_memory_allocated())
        print(f"  {label}: {fps:.2f} FPS  |  {lat:.1f} ms  |  peak {peak:.0f} MB")
        return fps, lat, peak
    except Exception as e:
        print(f"  {label}: FAILED — {type(e).__name__}: {e}")
        return 0, 0, 0

print("=" * 64)
print("     SAM 3.1 NPU Inference Benchmark")
print(f"     Device: {DEVICE}   Resolution: {RES}x{RES}   Precision: BF16")
print("=" * 64)

# ── Build model ─────────────────────────────────────────────
print("\n[1/5] Loading SAM 3.1 multiplex model ...")
t0 = time.time()
from sam3.model_builder import build_sam3_multiplex_video_predictor
predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=CKPT, compile=False, warm_up=False, use_fa3=False,
)
model = predictor.model
synchronize()
load_time = time.time() - t0
mem_model = mb(memory_allocated())
print(f"  Done in {load_time:.1f}s  |  Model memory: {mem_model:.0f} MB")
R = {}

# ── ViT Vision Backbone ────────────────────────────────────
print("\n[2/5] ViT Vision Backbone (single image encoding)")
backbone = model.detector.backbone
img = torch.randn(1, 3, RES, RES, device=DEVICE)

@torch.inference_mode()
def f_backbone():
    with torch.autocast(device_type=AC, dtype=DTYPE):
        backbone.forward_image(img)

R["ViT backbone"] = bench(f_backbone, label="ViT backbone")

# ── Full end-to-end image segmentation via Sam3Processor ───
print("\n[3/5] Image Segmentation (Sam3Processor end-to-end)")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    img_model = build_sam3_image_model(
        checkpoint_path=CKPT, compile=False, eval_mode=True, load_from_HF=False,
    )
    processor = Sam3Processor(img_model, resolution=RES)
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))

    @torch.inference_mode()
    def f_image_seg():
        state = processor.set_image(dummy_pil)
        processor.set_text_prompt(state=state, prompt="a person")

    R["Image segmentation"] = bench(f_image_seg, label="Image seg e2e")
except Exception as e:
    print(f"  Image seg: skipped ({type(e).__name__}: {e})")

# ── Video frame throughput ──────────────────────────────────
print("\n[4/5] Video Frame Throughput (backbone per-frame)")
N_FRAMES = 30
frames = [torch.randn(1, 3, RES, RES, device=DEVICE) for _ in range(N_FRAMES)]

@torch.inference_mode()
def f_video():
    with torch.autocast(device_type=AC, dtype=DTYPE):
        for fr in frames:
            backbone.forward_image(fr)

gc.collect(); empty_cache(); reset_peak_memory_stats()
for _ in range(2): f_video()
synchronize(); reset_peak_memory_stats(); synchronize()
t0 = time.perf_counter()
for _ in range(3): f_video()
synchronize(); t1 = time.perf_counter()
total = 3 * N_FRAMES
vfps = total / (t1 - t0); vlat = (t1 - t0) / total * 1000; vpeak = mb(max_memory_allocated())
print(f"  Video {N_FRAMES}f: {vfps:.2f} FPS  |  {vlat:.1f} ms/frame  |  peak {vpeak:.0f} MB")
R[f"Video backbone ({N_FRAMES}f)"] = (vfps, vlat, vpeak)

# ── Memory footprint ───────────────────────────────────────
print("\n[5/5] Memory Footprint")
synchronize()
mem_now = mb(memory_allocated())
print(f"  Model loaded:    {mem_model:.0f} MB")
print(f"  After inference: {mem_now:.0f} MB")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  BENCHMARK RESULTS")
print("=" * 64)
print(f"\n  {'Component':<30} {'FPS':>8} {'Latency':>10} {'Peak Mem':>10}")
print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*10}")
for name, (fps, lat, peak) in R.items():
    if fps > 0:
        print(f"  {name:<30} {fps:>7.1f}  {lat:>8.1f} ms  {peak:>7.0f} MB")

print(f"\n  {'Metric':<30} {'Value':>20}")
print(f"  {'─'*30} {'─'*20}")
print(f"  {'Device':<30} {str(DEVICE):>20}")
print(f"  {'Precision':<30} {'BF16 autocast':>20}")
print(f"  {'Resolution':<30} {f'{RES}x{RES}':>20}")
print(f"  {'Model Load Time':<30} {f'{load_time:.1f}s':>20}")
print(f"  {'Model Memory':<30} {f'{mem_model:.0f} MB':>20}")
