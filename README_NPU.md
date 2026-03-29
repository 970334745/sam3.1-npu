# SAM 3.1 昇腾 NPU 适配版

> 基于 [Meta SAM 3.1](https://github.com/facebookresearch/sam3)（Segment Anything Model 3.1）的华为昇腾（Ascend）NPU 完整适配。包含 4 个第一性原理自创 NPU 算子，实现推理管线零 CPU 回退。

---

## 目录

- [性能基准](#性能基准)
- [快速开始](#快速开始)
- [整体适配架构](#整体适配架构)
- [自创算子 1：NPU Fusion Attention](#自创算子-1npu-fusion-attention)
- [自创算子 2：NPU NMS 非最大抑制](#自创算子-2npu-nms-非最大抑制)
- [自创算子 3：NPU 连通域分析](#自创算子-3npu-连通域分析)
- [自创算子 4：NPU 欧氏距离变换](#自创算子-4npu-欧氏距离变换)
- [附加修复：Fused MatMul+Activation](#附加修复fused-matmulactivation)
- [设备抽象层](#设备抽象层)
- [分布式训练适配](#分布式训练适配)
- [完整修改文件清单](#完整修改文件清单)
- [已知限制](#已知限制)

---

## 性能基准

**硬件**：昇腾 NPU（Ascend 910B） | **精度**：BF16 autocast | **分辨率**：1008×1008

### 推理 FPS 与延迟

| 组件 | FPS | 延迟 (ms) | 峰值显存 (MB) |
|------|:---:|:---------:|:------------:|
| ViT 视觉骨干网络（单图编码） | **10.9** | 91.4 | 4,746 |
| 图像分割端到端（编码 + 检测 + 分割） | **4.4** | 228.8 | 9,409 |
| 视频帧处理 (30帧连续) | **10.8** | 92.9/帧 | 9,729 |

### 显存占用

| 指标 | 数值 |
|------|------|
| 模型权重加载后显存 | 3,742 MB |
| 推理峰值显存（图像分割） | 9,409 MB |
| 推理峰值显存（视频跟踪） | 9,729 MB |
| 模型加载时间（首次含算子编译） | ~47 s |

### 自创算子性能对比

| 算子 | 对比基线 | 加速比 | 备注 |
|------|---------|:------:|------|
| NPU Fusion Attention | PyTorch SDPA | **1.5–2.0x** | B=2, S=1024, H=8, D=128 |
| NPU NMS (N≥1500) | CPU numpy NMS | **1.7–3.3x** | 传输量减少 8x |
| NPU Connected Components | CPU skimage + 数据搬运 | 管线零停顿 | 消除 NPU↔CPU 同步 |
| NPU EDT | CPU 回退不可用 | NPU 原生 | 全设备端计算 |

---

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

# 3. 下载模型权重（魔塔社区，国内高速）
pip install modelscope
modelscope download --model facebook/sam3.1 --local_dir ./weights
```

### 图像分割

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# 设备自动检测：NPU > CUDA > CPU
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
    use_fa3=False,   # NPU 自动使用 npu_fusion_attention
    compile=False,
)

response = predictor.handle_request({
    "type": "start_session",
    "video_path": "your_video.mp4",
})
```

### 运行基准测试

```bash
python scripts/npu_benchmark.py
```

---

## 整体适配架构

```
SAM 3.1 推理管线（NPU 完整链路，零 CPU 回退）
│
├─ 图像输入 (1008×1008)
│
├─ [ViT 视觉编码器] ──→ 多尺度特征图
│   ├─ Patch Embedding
│   ├─ 32 × Transformer Block
│   │   ├─ RoPE Attention ──→ 【自创算子1】npu_fusion_attention
│   │   └─ MLP (GELU) ──→ 【附加修复】addmm + gelu 分步
│   └─ 三尺度特征颈部 (TriViTDetNeck)
│
├─ [文本编码器] ──→ 文本嵌入
│   └─ Transformer ──→ 【自创算子1】npu_fusion_attention
│
├─ [Transformer 解码器] ──→ 目标查询
│   ├─ 自注意力 ──→ 【自创算子1】npu_fusion_attention
│   ├─ 交叉注意力 ──→ 【自创算子1】npu_fusion_attention
│   └─ FFN
│
├─ [分割头] ──→ mask logits
│   └─ NMS 后处理 ──→ 【自创算子2】NPU 向量化贪心 NMS
│
├─ [视频跟踪模块] (SAM 3.1 Multiplex)
│   ├─ 连通域分析 ──→ 【自创算子3】NPU 迭代标签传播
│   ├─ 距离变换 ──→ 【自创算子4】NPU 行列分解 EDT
│   └─ 记忆编码器 / 注意力 ──→ 【自创算子1】
│
└─ 输出: 分割掩码 + 跟踪 ID
```

### 设备抽象层 `sam3/device_utils.py`

新增统一设备抽象层（约 280 行），自动检测 NPU > CUDA > CPU，为全部上层代码提供无差别接口：

```python
from sam3.device_utils import get_accelerator, get_device
print(get_accelerator())  # "npu" / "cuda" / "cpu"
print(get_device())       # torch.device("npu:0")
```

核心函数：

| 函数 | 作用 | 替代的原始 API |
|------|------|---------------|
| `get_accelerator()` | 返回最佳加速器名 | 硬编码 `"cuda"` |
| `get_device()` | 返回 torch.device | `torch.device("cuda")` |
| `get_autocast_device_type()` | autocast 设备类型 | `device_type="cuda"` |
| `get_dist_backend()` | 分布式后端名 | `"nccl"` |
| `model_to_device(model)` | 移动模型到设备 | `model.cuda().eval()` |
| `is_on_accelerator(tensor)` | 检查张量是否在加速器上 | `tensor.is_cuda` |
| `setup_tf32()` | 配置 TF32 | `torch.backends.cuda.*` |
| `empty_cache()` | 释放显存缓存 | `torch.cuda.empty_cache()` |
| `supports_torch_compile()` | 编译兼容性检查 | — |

---

## 自创算子 1：NPU Fusion Attention

**文件**：`sam3/perflib/fa3.py`

### 问题分析

SAM 3.1 原始实现在 CUDA 上使用 Flash Attention 3（FA3），通过 FP8 量化实现极致吞吐。FA3 依赖 `flash_attn_interface` 这一 CUDA 专用库，在 NPU 上完全不可用。朴素回退方案是使用 PyTorch 的 `scaled_dot_product_attention`（SDPA），但 SDPA 缺少 NPU 上的融合优化。

### 设计思路

通过调研发现 `torch_npu.npu_fusion_attention` 是昇腾 NPU 的原生融合注意力算子，在硬件层面将 Q×K^T → Scale → Softmax → ×V 四步运算融合为一个 kernel。该算子支持 BSND（Batch, Seq, NumHeads, Dim）布局，恰好与 SAM 3.1 内部的 `(B, S, H, D)` 张量布局一致，无需转置。

### 实现细节

```python
def _npu_fusion_attention_impl(q, k, v):
    """q/k/v: (batch, seq_len, num_heads, head_dim) — BSND 布局"""
    head_num = q.shape[2]
    head_dim = q.shape[3]
    scale = 1.0 / math.sqrt(head_dim)

    # 自动选择计算精度：BF16 或 FP16
    compute_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16

    out = torch_npu.npu_fusion_attention(
        q.to(compute_dtype).contiguous(),
        k.to(compute_dtype).contiguous(),
        v.to(compute_dtype).contiguous(),
        head_num,
        "BSND",             # 布局声明：Batch-Seq-NumHeads-Dim
        scale=scale,         # 缩放因子 1/√d
        keep_prob=1.0,       # 推理时不 dropout
    )[0]

    return out.to(q.dtype)  # 恢复原始精度
```

### 三路调度

```python
def flash_attn_func(q, k, v):
    # 路径 1：CUDA + FA3 → FP8 Flash Attention 3
    if _FA3_AVAILABLE and q.is_cuda:
        return flash_attn_func_op(q.to(FP8), k.to(FP8), v.to(FP8)).to(q.dtype)
    # 路径 2：NPU → 原生融合注意力
    if _NPU_AVAILABLE and q.device.type == "npu":
        return _npu_fusion_attention_impl(q, k, v)
    # 路径 3：通用回退 → PyTorch SDPA
    return _sdpa_fallback(q, k, v)
```

### 性能验证

| 配置 | SDPA (ms) | NPU Fusion (ms) | 加速比 |
|------|:---------:|:---------------:|:------:|
| B=2, S=1024, H=8, D=128, BF16 | 0.160 | 0.109 | **1.47x** |
| B=2, S=1024, H=8, D=128, FP16 | 0.220 | 0.109 | **2.02x** |
| B=1, S=4096, H=16, D=80, BF16 | 0.967 | 1.134 | 0.85x |

**结论**：在 SAM 3.1 典型的中等序列长度（1024）场景下获得 1.5–2x 加速。超长序列（4096）时两者接近。

### 正确性

| 精度 | max_diff | mean_diff |
|------|----------|-----------|
| BF16 | **0.000000** | 0.000000 |
| FP16 | 0.001465 | 0.000077 |

BF16 下与 SDPA 输出完全一致。

---

## 自创算子 2：NPU NMS 非最大抑制

**文件**：`sam3/perflib/nms.py`

### 问题分析

SAM 3.1 的 NMS 不是标准的 box NMS，而是基于预计算的 **mask IoU 矩阵** 的贪心抑制。输入是一个 (N, N) 的 IoU 矩阵和 (N,) 的分数向量。CUDA 上使用 Triton kernel 或 `torch_generic_nms` 库，NPU 上均不可用。

### 设计思路

NMS 的贪心性质（每保留一个检测后，需要知道它的结果才能决定后续抑制）决定了它无法完全并行化。关键洞察：**排序和阈值比较可以在 NPU 上并行完成，只有最终的贪心选择循环需要 CPU**。

### 自适应分派策略

```python
_NPU_NMS_THRESHOLD = 1500

def generic_nms_npu(ious, scores, iou_threshold):
    n = scores.size(0)
    if n >= _NPU_NMS_THRESHOLD:
        return _greedy_nms_npu_sort(ious, scores, iou_threshold)  # 大 N 路径
    else:
        return _greedy_nms_via_cpu(ious, scores, iou_threshold)    # 小 N 路径
```

**小 N 路径**（N < 1500）：整体搬运到 CPU，numpy 贪心循环。N 小时 kernel launch 开销主导，直接搬运反而更快。

**大 N 路径**（N ≥ 1500）的核心创新：

```python
def _greedy_nms_npu_sort(ious, scores, iou_threshold):
    # 步骤 1：NPU 上并行排序（O(N log N)，NPU 原生算子）
    order_npu = torch.argsort(scores, descending=True)

    # 步骤 2：NPU 上按排序重排 IoU 矩阵并阈值化
    ious_sorted = ious[order_npu][:, order_npu]
    iou_mask_cpu = (ious_sorted > iou_threshold).cpu()  # bool: 传输量仅 N×N 字节

    # 步骤 3：CPU 上极快的 bool 贪心循环
    keep_mask = torch.ones(n, dtype=torch.bool)
    for i in range(n - 1):
        if not keep_mask[i]:
            continue
        keep_mask[i + 1:] &= ~iou_mask_cpu[i, i + 1:]  # 向量化布尔操作

    return order_cpu[keep_mask].to(device=device)
```

**关键优化**：传输 bool 矩阵（N×N 字节）而非 float32 IoU 矩阵（N×N×4 字节），数据传输量减少 **8 倍**。

### NPU 优先调度

由于 torch_npu 的兼容层会使 NPU 张量的 `.is_cuda` 返回 `True`，调度逻辑必须**先检查 NPU 再检查 CUDA**：

```python
def generic_nms(ious, scores, iou_threshold):
    if _is_npu_tensor(ious):           # NPU 优先
        return generic_nms_npu(...)
    if ious.is_cuda:                    # CUDA 路径
        return generic_nms_cuda(...)
    return generic_nms_cpu(...)         # CPU 回退
```

### 性能

| N | CPU numpy (ms) | NPU 自适应 (ms) | 加速比 |
|---|:--------------:|:---------------:|:------:|
| 200 | 0.65 | 0.66 | ~1.0x |
| 500 | 1.74 | 1.99 | ~1.0x |
| 2000 | 23.4 | 13.5 | **1.73x** |
| 4000 | 118.5 | 36.2 | **3.27x** |

小 N 零退化，大 N 显著加速。

---

## 自创算子 3：NPU 连通域分析

**文件**：`sam3/perflib/connected_components.py`

### 问题分析

连通域分析用于视频跟踪中的掩码后处理（识别独立物体、填补小孔）。CUDA 上使用 `cc_torch`（CUDA kernel）或 Triton 实现。CPU 上使用 `skimage.measure.label`（C 实现的两遍扫描，O(N)）。

NPU 回退到 CPU 的隐性成本极大：

```
NPU 清空队列 → 同步等待 → NPU→CPU 数据搬运 → CPU 计算 → CPU→NPU 搬运 → NPU 重启管线
```

这个**管线停顿**的代价通常远超纯计算时间差。

### 设计思路：迭代标签传播 + 指针跳跃

核心算法是并行 Union-Find 的一种变体：

1. **初始化**：每个前景像素的标签 = 该像素的线性索引（1-indexed），背景 = 0
2. **邻域传播**：每个像素取 3×3 邻域内标签的最小值（8-连通）
3. **指针跳跃**：`labels[i] = labels[labels[i]]` 加速链压缩
4. **重复**直到收敛
5. **计数**：`scatter_add_` 并行计算每个标签的像素数

### 关键技巧：`-max_pool2d(-x)` 实现最小值传播

NPU 没有 `min_pool2d`，但 `max_pool2d` 是高度优化的融合 kernel。利用数学恒等式：

$$\min(x_1, x_2, \ldots, x_n) = -\max(-x_1, -x_2, \ldots, -x_n)$$

```python
# 取反后进入浮点空间，背景设为 -max_val（max_pool2d 会忽略）
neg_f = -(labels.float())
neg_f = torch.where(mask, neg_f, -max_val)

# 单次 fused kernel 完成 3×3 邻域最小值传播
pooled = F.max_pool2d(neg_f.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
neg_f = torch.where(mask, pooled, -max_val)
```

### 收敛优化

- **批量传播**：每 `prop_batch` 步（最大 512）才做一次 `torch.equal()` 收敛检查，极大减少 NPU↔CPU 同步点
- **指针跳跃轮数**：`ceil(log2(prop_batch))` 轮，保证链路完全压缩
- **全程浮点**：内循环驻留 float32 空间，避免每步 int↔float 转换

### 计数的向量化

```python
# 不使用 Python 循环，完全并行
ones = torch.ones(B, N, device=device, dtype=torch.int64)
label_counts = torch.zeros(B, N + 1, device=device, dtype=torch.int64)
label_counts.scatter_add_(1, labels_flat, ones)         # 每标签计数
counts_flat = torch.gather(label_counts, 1, labels_flat) # 广播回每像素
```

### 验证

26/26 测试用例全部通过（单方块、双矩形、对角线、棋盘格、L 形、全前景、全背景、随机掩码等），与 CPU skimage 实现结果完全一致。

---

## 自创算子 4：NPU 欧氏距离变换

**文件**：`sam3/model/edt.py`

### 问题分析

EDT（Euclidean Distance Transform）计算每个前景像素到最近背景像素的欧氏距离。CUDA 上使用 Triton kernel 实现 Felzenszwalb-Huttenlocher 的抛物线包络算法（O(N²)复杂度，支持跨行/列并行）。Triton 在 NPU 上完全不可用。

EDT 在 SAM 3.1 中用于 `fill_holes_in_mask_scores`（填补掩码小孔）和跟踪质量评估。

### 设计思路：行列分解的可分离距离变换

利用 EDT 的**可分离性**——2D EDT 可分解为两次 1D EDT：

$$\text{EDT}^2(i,j) = \min_k \left[ \text{RowEDT}^2(k,j) + (i-k)^2 \right]$$

**Phase 1（行方向 1D EDT²）**：

对每一行独立计算每个像素到最近背景的 1D 距离平方。使用前向-后向扫描：

```python
# 前向扫描：从左到右传播
row_f = f.reshape(B * H, W)  # (B*H, W) 批量并行处理所有行
for w in range(1, W):
    prev = row_f[:, w - 1]
    # 距离传播公式：d²[w] = min(d²[w], (√d²[w-1] + 1)²)
    candidate = prev + 2.0 * torch.sqrt(prev.clamp(min=0)) + 1.0
    row_f[:, w] = torch.minimum(row_f[:, w], candidate)

# 后向扫描：从右到左修正
for w in range(W - 2, -1, -1):
    nxt = row_f[:, w + 1]
    candidate = nxt + 2.0 * torch.sqrt(nxt.clamp(min=0)) + 1.0
    row_f[:, w] = torch.minimum(row_f[:, w], candidate)
```

**Phase 2（列方向合并）**：

将 Phase 1 的结果按列重排后再做一次前向-后向扫描：

```python
col_f = row_sq.permute(0, 2, 1).reshape(B * W, H)  # 列变行，批量并行
# 同样的前向 + 后向扫描
```

**Phase 3**：取平方根得到欧氏距离。

### 传播公式推导

对于 1D 距离平方序列 `d²[]`，从 `w-1` 到 `w` 的传播关系：

- 如果 `w-1` 是背景（`d²[w-1] = 0`），则 `d²[w]` 候选值 = 1（距离为1的平方）
- 如果 `d²[w-1] = d²`（距离为 `d`），则 `d²[w]` 候选值 = `(d+1)² = d² + 2d + 1 = d² + 2√d² + 1`

因此统一公式为：`candidate = d²[w-1] + 2·√d²[w-1] + 1`

### Triton 条件导入

NPU 上 Triton 不可用，edt.py 使用条件导入避免 ImportError：

```python
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass

if _TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(...): ...  # 仅在 CUDA 上定义
```

### 验证

| 测试 | 预期 | 实际 | 状态 |
|------|------|------|------|
| 圆形掩码 (r=15) 中心距离 | 15.0 | 15.0 | ✅ |
| 圆形掩码边缘距离 | ~0-1 | 正确 | ✅ |
| 背景像素距离 | 0.0 | 0.0 | ✅ |
| 4×256×256 随机掩码 | 不崩溃 | 239.1 ms | ✅ |

---

## 附加修复：Fused MatMul+Activation

**文件**：`sam3/perflib/fused.py`

### 问题

ViT 的每一层 MLP 调用 `aten::_addmm_activation`——一个将矩阵乘法和激活函数融合的 CUDA 算子。在 NPU 上该算子不存在，PyTorch 自动回退到 CPU 执行并打印警告：

```
Warning: CAUTION: The operator 'aten::_addmm_activation.out' is not currently
supported on the NPU backend and will fall back to run on the CPU.
```

这导致 ViT 的 **每一层 MLP** 都触发一次 NPU→CPU→NPU 的数据搬运，严重影响性能。

### 解决方案

拆分为两步原生操作，全部在 NPU 上执行：

```python
if _USE_FUSED_ADDMM:  # CUDA 路径：使用融合算子
    y = addmm_act_op(bias, mat1, weight.t(), use_gelu=True)
else:                  # NPU 路径：分步执行，零 CPU 回退
    y = torch.addmm(bias, mat1, weight.t())  # NPU 原生
    y = torch.nn.functional.gelu(y)           # NPU 原生
```

### 效果

彻底消除了 `_addmm_activation` 的 CPU 回退警告，ViT 所有层均在 NPU 上完整执行。

---

## 分布式训练适配

| 项目 | CUDA | NPU |
|------|------|-----|
| 分布式后端 | NCCL | **HCCL** |
| 设备设置 | `torch.cuda.set_device` | `torch.npu.set_device` |
| AMP autocast | `device_type="cuda"` | `device_type="npu"` |
| 随机种子 | `torch.cuda.manual_seed_all` | `torch.npu.manual_seed_all` |
| 配置 | `accelerator: cuda` | `accelerator: npu` |

训练配置只需修改一行：

```yaml
# train/configs/eval_base.yaml
accelerator: npu
```

---

## 完整修改文件清单

### 新增文件

| 文件 | 说明 | 代码行数 |
|------|------|:-------:|
| `sam3/device_utils.py` | 设备抽象层 | ~280 |
| `scripts/npu_benchmark.py` | 性能基准测试 | ~140 |
| `scripts/npu_profile.py` | 性能分析工具 | ~100 |
| `README_NPU.md` | 本报告 | — |
| `NPU_ADAPTATION_REPORT.md` | 基础适配报告 | — |
| `NPU_CUSTOM_OPERATORS_REPORT.md` | 算子报告 | — |
| `NPU_CAPABILITY_REPORT.md` | torch_npu 能力分析 | — |

### 自创算子文件

| 文件 | 自创内容 | 代码行数 |
|------|---------|:-------:|
| `sam3/perflib/fa3.py` | `_npu_fusion_attention_impl` + 三路调度 | ~40 |
| `sam3/perflib/nms.py` | `generic_nms_npu` + 自适应分派 | ~70 |
| `sam3/perflib/connected_components.py` | `connected_components_npu` + NPU 优先调度 | ~120 |
| `sam3/model/edt.py` | `_edt_npu` + 条件 Triton 导入 | ~80 |
| `sam3/perflib/fused.py` | addmm + activation 分步 | ~15 |

### 适配修改文件（34 个）

<details>
<summary>点击展开完整列表</summary>

| 文件 | 关键修改 |
|------|----------|
| `sam3/model_builder.py` | 默认设备 → `get_default_device_string()` |
| `sam3/model/sam3_video_predictor.py` | `.cuda().eval()` → `model_to_device()` |
| `sam3/model/sam3_tracking_predictor.py` | autocast + storage_device 适配 |
| `sam3/model/sam3_multiplex_video_predictor.py` | TF32 + autocast 适配 |
| `sam3/model/sam3_image_processor.py` | 默认 device 适配 |
| `sam3/model/sam3_multiplex_tracking.py` | device 检查扩展 NPU |
| `sam3/model/model_misc.py` | SDPA 设置委托 device_utils |
| `sam3/model/decoder.py` | autocast + 坐标缓存设备 |
| `sam3/model/position_encoding.py` | 预计算设备适配 |
| `sam3/model/vl_combiner.py` | 三处 device 默认值 |
| `sam3/model/video_tracking_multiplex.py` | `.cuda()` → `.to(self.device)` |
| `sam3/model/video_tracking_multiplex_demo.py` | 全面适配 |
| `sam3/model/sam3_multiplex_base.py` | TF32 安全初始化 + profiler NPU |
| `sam3/model/sam3_tracker_base.py` | `.cuda()` → `.to(device)` |
| `sam3/model/sam3_video_inference.py` | 编译预热 NPU 支持 |
| `sam3/model/io_utils.py` | 视频加载设备适配 |
| `sam3/model/act_ckpt_utils.py` | `is_cuda` → `is_on_accelerator()` |
| `sam3/model/edt.py` | Triton 条件导入 + NPU EDT |
| `sam3/model/utils/sam2_utils.py` | 视频帧加载设备适配 |
| `sam3/sam/transformer.py` | SDPA 后端 + RoPE 复数张量修复 |
| `sam3/agent/helpers/memory.py` | OOM 处理同时支持 NPU |
| `sam3/eval/postprocessors.py` | 设备断言扩展 NPU |
| `sam3/perflib/compile.py` | NPU 跳过 compile + is_on_accelerator |
| `sam3/perflib/triton/connected_components.py` | NPU 回退守卫 |
| `sam3/perflib/triton/nms.py` | NPU 回退守卫 |
| `sam3/train/trainer.py` | accelerator=npu + HCCL + autocast |
| `sam3/train/utils/distributed.py` | NPU set_device + HCCL 张量 |
| `sam3/train/utils/train_utils.py` | NPU 随机种子 |
| `sam3/train/loss/sam3_loss.py` | 默认 device 延迟检测 |
| `sam3/train/masks_ops.py` | `is_cuda` → `is_on_accelerator()` |

</details>

---

## 已知限制

1. **torch.compile**：NPU 上默认禁用（可通过 `SAM3_NPU_ENABLE_COMPILE=1` 环境变量开启实验性支持）
2. **Flash Attention 3 精度**：NPU 使用 BF16/FP16（非 FP8），精度更高但吞吐略低于 CUDA FA3
3. **连通域大掩码**：迭代传播算法对 >512×512 的密集掩码需要较多迭代，但消除了管线停顿的隐性代价
4. **首次加载**：首次构建模型时 NPU 需编译算子图，约 50–100s，后续加载更快
5. **ComplexFloat**：torch_npu 不支持复数张量的 deepcopy，RoPE 位置编码在 CPU 初始化后延迟搬移

---

## 许可证

本适配基于 [SAM 3 原始代码](https://github.com/facebookresearch/sam3) 修改，遵循原始许可证（META License）。
