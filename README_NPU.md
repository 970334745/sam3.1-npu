# SAM 3.1 昇腾 NPU 适配版

> 基于 [Meta SAM 3.1](https://github.com/facebookresearch/sam3) 的华为昇腾（Ascend）NPU 完整适配，支持推理和训练。

## 性能基准

**设备**：昇腾 NPU（Ascend 910B）| **精度**：BF16 autocast | **分辨率**：1008×1008

| 组件 | FPS | 延迟 (ms) | 峰值显存 (MB) |
|------|:---:|:---------:|:------------:|
| ViT 视觉骨干网络 | **10.9** | 91.4 | 4,746 |
| 图像分割端到端 | **4.4** | 228.8 | 9,409 |
| 视频帧处理 (30帧) | **10.8** | 92.9 | 9,729 |

| 指标 | 数值 |
|------|------|
| 模型显存 | 3,742 MB |
| 推理峰值显存 | ~9.7 GB |
| 模型加载时间 | ~47s（首次含编译） |

## 快速开始

### 环境要求

- 华为昇腾 NPU（Ascend 910B 或更高）
- CANN 8.0+
- Python 3.10+
- PyTorch 2.1+ with torch_npu

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/970334745/sam3.1-npu.git
cd sam3.1-npu
git checkout npu-adaptation

# 2. 安装 SAM 3.1
pip install -e .

# 3. 下载模型权重（魔塔社区）
pip install modelscope
modelscope download --model facebook/sam3.1 --local_dir ./weights
```

### 图像分割

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# 自动检测并使用 NPU
model = build_sam3_image_model(checkpoint_path="weights/sam3.1_multiplex.pt")
processor = Sam3Processor(model)

image = Image.open("your_image.jpg")
state = processor.set_image(image)
result = processor.set_text_prompt(state=state, prompt="a cat")
```

### 视频跟踪（SAM 3.1 Multiplex）

```python
from sam3.model_builder import build_sam3_multiplex_video_predictor

predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path="weights/sam3.1_multiplex.pt",
    use_fa3=False,   # NPU 使用 npu_fusion_attention
    compile=False,    # NPU 上可选
)

# 使用 handle_request API
response = predictor.handle_request({
    "type": "start_session",
    "video_path": "your_video.mp4",
})
```

### 运行基准测试

```bash
python scripts/npu_benchmark.py
```

## 技术方案

### 设备抽象层

`sam3/device_utils.py` 提供统一设备抽象，自动检测 NPU > CUDA > CPU：

```python
from sam3.device_utils import get_device, get_accelerator
print(get_accelerator())  # "npu"
print(get_device())       # npu:0
```

### 自创 NPU 算子

所有 CUDA/Triton 专用算子均已用 NPU 原生实现替代，**零 CPU 回退**：

| 算子 | CUDA 原实现 | NPU 自创实现 | 加速比 vs SDPA/CPU |
|------|-----------|------------|:------------------:|
| 注意力 | Flash Attention 3 (FP8) | `npu_fusion_attention` (BF16) | **1.5-2x** |
| NMS | Triton kernel | 向量化贪心 + 自适应分派 | **1.7-3.3x** |
| 连通域分析 | cc_torch / Triton | 迭代标签传播 (`max_pool2d`) | 消除管线停顿 |
| 距离变换 | Triton EDT kernel | 行列分解前后扫描 | NPU 原生运行 |
| MatMul+Act | `_addmm_activation` | `addmm` + `gelu`/`relu` 分步 | 消除 CPU fallback 警告 |

### 分布式训练

- 分布式后端：HCCL（替代 NCCL）
- 训练配置中设置 `accelerator: npu`

```yaml
# train/configs/eval_base.yaml
accelerator: npu
```

## 适配范围

| 类别 | 修改文件数 | 新增代码行 |
|------|:---------:|:---------:|
| 设备抽象层 | 1 | ~280 |
| 模型构建 | 1 | ~30 |
| 推理 Predictor | 5 | ~60 |
| 模型核心 | 12 | ~80 |
| 自创算子 | 6 | ~600 |
| 训练 & 工具 | 8 | ~100 |
| 测试 & 报告 | 5 | ~800 |
| **合计** | **~40** | **~2000** |

## 项目结构

```
sam3.1-npu/
├── sam3/
│   ├── device_utils.py          # 设备抽象层（新增）
│   ├── model_builder.py         # 模型构建（适配）
│   ├── perflib/
│   │   ├── fa3.py               # NPU Fusion Attention（自创）
│   │   ├── fused.py             # addmm+activation（修复）
│   │   ├── nms.py               # NPU NMS（自创）
│   │   ├── connected_components.py  # NPU 连通域（自创）
│   │   └── compile.py           # torch.compile 兼容
│   └── model/
│       ├── edt.py               # NPU EDT 距离变换（自创）
│       └── ...                  # 推理模型（适配）
├── scripts/
│   ├── npu_benchmark.py         # 性能基准测试
│   └── npu_profile.py           # 性能分析
├── weights/                     # 模型权重
├── NPU_ADAPTATION_REPORT.md     # 基础适配报告
├── NPU_CUSTOM_OPERATORS_REPORT.md  # 自创算子报告
└── NPU_CAPABILITY_REPORT.md     # torch_npu 能力分析
```

## 已知限制

1. **torch.compile**：NPU 上默认禁用（可通过 `SAM3_NPU_ENABLE_COMPILE=1` 开启）
2. **Flash Attention 3**：NPU 使用 `npu_fusion_attention` 替代，精度为 BF16/FP16（非 FP8）
3. **连通域分析**：迭代传播算法对大尺寸掩码（>512×512）性能不如 CPU skimage，但避免了管线停顿
4. **首次加载**：首次构建模型时需要编译算子图，约 50-100s，后续加载更快

## 引用

```bibtex
@article{sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Meta FAIR},
  year={2025}
}
```

## 许可证

本适配基于 [SAM 3 原始代码](https://github.com/facebookresearch/sam3) 修改，遵循原始许可证。
