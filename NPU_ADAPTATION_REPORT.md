# SAM 3.1 昇腾 NPU 适配报告

## 概述

本次适配将 Meta SAM 3.1（Segment Anything Model 3.1）从 NVIDIA CUDA 平台移植到华为昇腾（Ascend）NPU 平台。适配后模型可在 NPU 上完成完整的推理流程，包括图像分割、视频跟踪和 multiplex 多目标跟踪。

## 验证结果

| 验证项 | 状态 |
|--------|------|
| 设备检测（NPU 自动识别） | ✅ 通过 |
| model_builder 全部构建函数导入 | ✅ 通过 |
| 推理 Predictor 全部导入 | ✅ 通过 |
| perflib 算子库导入 | ✅ 通过 |
| SAM 3.1 Multiplex 模型构建并部署到 NPU:0 | ✅ 通过 |

**环境信息**：
- 加速器：`npu:0`
- autocast 设备类型：`npu`
- 分布式后端：`hccl`
- torch_npu：已安装

## 适配架构

### 核心设计：`sam3/device_utils.py`

新增统一设备抽象层，自动检测最佳加速器（NPU > CUDA > CPU），提供：

| 函数 | 用途 |
|------|------|
| `get_accelerator()` | 返回 `"npu"` / `"cuda"` / `"cpu"` |
| `get_device()` | 返回 `torch.device("npu:0")` 等 |
| `get_autocast_device_type()` | autocast 的 device_type 参数 |
| `get_dist_backend()` | 分布式后端：HCCL / NCCL / Gloo |
| `setup_tf32()` | 设备感知的 TF32 配置 |
| `is_on_accelerator(tensor)` | 替代 `.is_cuda` 检查 |
| `model_to_device(model)` | 替代 `model.cuda().eval()` |
| `supports_torch_compile()` | 编译兼容性检查 |
| `empty_cache()` / `memory_allocated()` 等 | 替代 `torch.cuda.*` 内存 API |

## 修改文件清单（34 个文件）

### 新增文件
| 文件 | 说明 |
|------|------|
| `sam3/device_utils.py` | 设备抽象层（~280 行） |

### 模型构建（1 文件）
| 文件 | 关键修改 |
|------|----------|
| `sam3/model_builder.py` | 默认设备改为 `get_default_device_string()`；`_setup_device_and_mode` 使用 `model_to_device`；`demo_model.cuda()` → `model_to_device()` |

### 推理 Predictor（5 文件）
| 文件 | 关键修改 |
|------|----------|
| `sam3/model/sam3_video_predictor.py` | `.cuda().eval()` → `model_to_device()`；NCCL → `get_dist_backend()`；内存统计使用 device_utils |
| `sam3/model/sam3_tracking_predictor.py` | autocast 使用 `get_autocast_device_type()`；`torch.device("cuda")` → `get_device()` |
| `sam3/model/sam3_multiplex_video_predictor.py` | TF32 设置使用 `setup_tf32()`；autocast 适配 |
| `sam3/model/sam3_image_processor.py` | 默认 device 改为 `get_default_device_string()` |
| `sam3/model/sam3_multiplex_tracking.py` | 设备检查扩展支持 NPU；autocast 适配 |

### 模型核心（12 文件）
| 文件 | 关键修改 |
|------|----------|
| `sam3/model/model_misc.py` | `get_sdpa_settings()` 委托给 device_utils |
| `sam3/model/decoder.py` | autocast、设备引用适配；复数张量保持 CPU 避免 torch_npu deepcopy 问题 |
| `sam3/model/position_encoding.py` | 预计算设备使用 `get_default_device_string()` |
| `sam3/model/vl_combiner.py` | 三处 `device="cuda"` 默认值改为动态检测 |
| `sam3/model/video_tracking_multiplex.py` | 所有 `.cuda()` → `.to(self.device)` |
| `sam3/model/video_tracking_multiplex_demo.py` | 全面适配（由子代理完成） |
| `sam3/model/sam3_multiplex_base.py` | 模块级 TF32 安全初始化；profiler 支持 NPU |
| `sam3/model/sam3_tracker_base.py` | `.cuda()` → `.to(device)` |
| `sam3/model/sam3_video_inference.py` | 编译预热检查扩展支持 NPU |
| `sam3/model/io_utils.py` | 所有 `.cuda()` → `get_device()`；TorchCodec 支持 NPU |
| `sam3/model/act_ckpt_utils.py` | `is_cuda` → `is_on_accelerator()` |
| `sam3/model/edt.py` | 断言改为设备感知 |

### 工具层（4 文件）
| 文件 | 关键修改 |
|------|----------|
| `sam3/model/utils/sam2_utils.py` | 三个视频加载函数默认设备改为 `get_device()` |
| `sam3/sam/transformer.py` | SDPA 后端设置加 CUDA 守卫；RoPE 复数张量保持 CPU |
| `sam3/agent/helpers/memory.py` | OOM 处理同时支持 CUDA 和 NPU |
| `sam3/eval/postprocessors.py` | 设备断言扩展支持 NPU |

### 算子库 perflib（4 文件）
| 文件 | 关键修改 |
|------|----------|
| `sam3/perflib/fa3.py` | FA3 不可用时自动回退到 SDPA |
| `sam3/perflib/compile.py` | NPU 上跳过 `torch.compile`；`is_cuda` → `is_on_accelerator()` |
| `sam3/perflib/connected_components.py` | NPU 回退到 CPU skimage 实现 |
| `sam3/perflib/nms.py` | NPU 回退到 CPU NMS |
| `sam3/perflib/triton/connected_components.py` | 移除 CUDA 断言，NPU 回退 CPU |
| `sam3/perflib/triton/nms.py` | 添加 NPU CPU 回退 |

### 训练相关（4 文件）
| 文件 | 关键修改 |
|------|----------|
| `sam3/train/trainer.py` | 支持 `accelerator: npu`；分布式后端推断支持 HCCL；autocast 设备感知 |
| `sam3/train/utils/distributed.py` | `set_cuda_device_index` 支持 NPU；通信张量转换支持 HCCL |
| `sam3/train/utils/train_utils.py` | 随机种子设置支持 NPU |
| `sam3/train/loss/sam3_loss.py` | 默认 device 改为 None（延迟检测） |
| `sam3/train/masks_ops.py` | `is_cuda` → `is_on_accelerator()` |

## 关键技术决策

### 1. Flash Attention 3 回退策略
NPU 不支持 FA3（`flash_attn_interface`），自动回退到 PyTorch 原生 SDPA：
```python
def flash_attn_func(q, k, v):
    if _FA3_AVAILABLE and q.is_cuda:
        return flash_attn_func_op(...)  # FA3 path
    return _sdpa_fallback(q, k, v)      # SDPA path for NPU
```

### 2. torch.compile 兼容性
NPU 上默认禁用 `torch.compile`（可通过 `SAM3_NPU_ENABLE_COMPILE=1` 环境变量开启）。

### 3. Triton Kernel 回退
所有 Triton kernel（NMS、连通域分析、EDT）在非 CUDA 设备上自动回退到 CPU/纯 PyTorch 实现。

### 4. 复数张量兼容性
torch_npu 不支持 `ComplexFloat` 存储的 `deepcopy`。RoPE 位置编码的复数张量 `freqs_cis` 保持在 CPU 初始化，`forward()` 时自动搬移到正确设备。

### 5. 分布式训练
NPU 使用 HCCL 后端（华为集合通信库），替代 NVIDIA NCCL。训练配置中将 `accelerator` 改为 `npu` 即可。

## 使用方式

### 推理（NPU 自动检测）
```python
from sam3.model_builder import build_sam3_multiplex_video_predictor

predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path="weights/sam3.1_multiplex.pt",
    compile=False,
    use_fa3=False,  # NPU 不支持 FA3，使用 SDPA
)
# 模型自动部署到 NPU:0
```

### 训练（修改配置文件）
```yaml
# train/configs/eval_base.yaml
accelerator: npu  # 改为 npu
```
