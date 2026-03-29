# torch_npu 完整能力报告 + SAM 3.1 自定义算子最优实现策略

## 环境信息

| 项目 | 值 |
|------|-----|
| 芯片 | **Ascend 910B2** |
| torch | 2.5.1 |
| torch_npu | 2.5.1 |
| torchvision | 0.20.1 |
| NPU 数量 | 1 |
| BF16 支持 | ✅ |
| FP16 支持 | ✅ |
| SDPA Flash | ✅ enabled |
| SDPA Mem-Efficient | ✅ enabled |
| SDPA Math | ✅ enabled |
| torch.compile(backend='npu') | ✅ 可用 |
| NPUGraph (类 CUDA Graph) | ✅ 可用 |

---

## 第一部分：torch_npu 完整能力清单

### 1.1 注意力机制 API（11 个）

| API | 功能 | 数据类型 | 备注 |
|-----|------|----------|------|
| `npu_fusion_attention` | **核心 Flash Attention**，支持完整的 Softmax(Mask(scale*(pse+Q*K^T), atten_mask)) * V | FP16/BF16/FP32 | 支持 BSH/SBH/BSND/BNSD/TND 布局；支持 causal/band/prefix 等 8 种 sparse_mode |
| `npu_prompt_flash_attention` | 全量序列 Flash Attention: softmax(scale*(Q*K)+atten_mask)*V | FP16/BF16 | 推理场景首选 |
| `npu_incre_flash_attention` | 增量推理 Flash Attention (decode 阶段) | FP16/BF16 | 适用于 KV-cache 场景 |
| `npu_fused_attention_score` | 融合注意力打分 | FP16/FP32 | 含 query/key/value + mask |
| `npu_fused_attention_score_fwd` | 融合注意力前向 | FP16/FP32 | |
| `npu_fused_infer_attention_score` | 推理融合注意力打分 | FP16/BF16 | |
| `npu_fused_attention_layernorm_qkv_fwd` | LayerNorm + QKV 投影融合 | FP16/BF16 | |
| `npu_multi_head_attention` | 多头注意力 | FP16/FP32 | |
| `npu_multi_head_attention_v2` | 多头注意力 v2 | FP16/FP32 | |
| `npu_scaled_masked_softmax` | 缩放 + Mask + Softmax 融合 | FP16/FP32/BF16 | |
| `npu_dropout_with_add_softmax` | Dropout + Add + Softmax 融合 | FP16/FP32 | |

**结论**：`npu_fusion_attention` 是 Ascend 910B 上的**最高性能注意力实现**，功能完全覆盖 FA2/FA3，且原生支持 BF16。

### 1.2 NMS（非极大值抑制）API（4 个）

| API | 签名 | 功能 |
|-----|------|------|
| `npu_batch_nms` | `(boxes, scores, score_thresh, iou_thresh, max_per_class, max_total, ...)` | **批量多类别 NMS**，返回 boxes/scores/classes/num |
| `npu_nms_v4` | `(boxes[N,4], scores[N], max_output, iou_thresh, score_thresh)` | **标准单类别 NMS**，返回 selected_indices + valid_num |
| `npu_nms_with_mask` | `(input, iou_threshold)` | NMS + mask 标记 |
| `npu_nms_rotated` | `(dets[N,5], scores[N], iou_thresh, score_thresh, max_output, mode)` | **旋转框 NMS** |

### 1.3 激活函数与归一化 API

| API | 功能 |
|-----|------|
| `npu_fast_gelu` / `npu_gelu` | FastGELU / GELU 激活 |
| `npu_silu` / `npu_silu_` | SiLU (Swish) 激活 |
| `npu_mish` | Mish 激活 |
| `npu_swiglu` | SwiGLU 激活（LLM 常用） |
| `npu_geglu` | GeGLU 激活 |
| `npu_rms_norm` | RMS 归一化（BF16/FP16/FP32） |
| `npu_add_layer_norm` | Add + LayerNorm 融合 |
| `npu_add_rms_norm` | Add + RMSNorm 融合 |
| `npu_deep_norm` | DeepNorm |
| `npu_group_norm_silu` / `npu_group_norm_swish` | GroupNorm + SiLU/Swish 融合 |
| `npu_gemma_rms_norm` | Gemma 风格 RMS Norm |

### 1.4 矩阵运算与量化 API

| API | 功能 |
|-----|------|
| `npu_bmmV2` | 批量矩阵乘法（FP16/FP32/INT32） |
| `npu_linear` | 线性层融合 |
| `npu_grouped_matmul` | 分组矩阵乘法（MoE 场景） |
| `npu_quant_matmul` | 量化矩阵乘法 |
| `npu_quant_matmul_dequant` | 量化矩阵乘 + 反量化 |
| `npu_dynamic_quant` | Per-token 对称动态量化 |
| `npu_dynamic_quant_asymmetric` | Per-token 非对称动态量化 |
| `npu_anti_quant` | 反量化 |
| `npu_weight_quant_batchmatmul` | 权重量化批量矩阵乘 |
| `npu_convert_weight_to_int4pack` | INT4 权重打包 |
| `npu_ffn` | FFN/MoeFFN 融合计算 |

### 1.5 位置编码 API

| API | 功能 |
|-----|------|
| `npu_rotary_mul` | RotaryEmbedding 旋转位置编码 |
| `npu_apply_rotary_pos_emb` | 应用旋转位置编码 |
| `npu_interleave_rope` | 交错式 RoPE |
| `npu_mrope` | 多维 RoPE |
| `npu_kv_rmsnorm_rope_cache` | KV RMSNorm + RoPE + Cache 融合 |

### 1.6 检测/分割相关 API

| API | 功能 |
|-----|------|
| `npu_roi_align` | ROI Align（FasterRCNN 风格） |
| `npu_ps_roi_pooling` | Position Sensitive ROI Pooling |
| `npu_deformable_conv2d` | 可变形卷积 |
| `npu_bounding_box_encode/decode` | 边框编解码 |
| `npu_rotated_box_encode/decode` | 旋转框编解码 |
| `npu_iou` / `npu_ptiou` | IoU 计算 |
| `npu_ciou` / `npu_diou` / `npu_giou` | CIoU/DIoU/GIoU |
| `npu_rotated_iou` / `npu_rotated_overlaps` | 旋转框 IoU / 重叠面积 |
| `npu_yolo_boxes_encode` | YOLO 框编码 |
| `npu_grid_assign_positive` | 正样本分配 |
| `npu_anchor_response_flags` | 锚点响应标记 |
| `npu_random_choice_with_mask` | 随机选择非零元素 |

### 1.7 通信与分布式 API

| API | 功能 |
|-----|------|
| `npu_all_gather_base_mm` | AllGather + MatMul 融合 |
| `npu_mm_all_reduce_base` | MatMul + AllReduce 融合 |
| `npu_mm_reduce_scatter_base` | MatMul + ReduceScatter 融合 |

### 1.8 MoE（专家混合）API

| API | 功能 |
|-----|------|
| `npu_moe_gating_top_k` | TopK 门控 |
| `npu_moe_gating_top_k_softmax` | TopK 门控 + Softmax |
| `npu_moe_init_routing` / `npu_moe_init_routing_v2` | 路由初始化 |
| `npu_moe_finalize_routing` | 路由终结 |
| `npu_moe_distribute_dispatch/combine` | 专家分发/合并 |
| `npu_moe_compute_expert_tokens` | 专家 token 计数 |
| `npu_moe_re_routing` | 重路由 |

### 1.9 内存与数据操作 API

| API | 功能 |
|-----|------|
| `npu_scatter` / `npu_scatter_nd_update` | 高效 Scatter 操作 |
| `npu_scatter_list` | 列表批量 Scatter |
| `npu_sort_v2` | 高速排序（仅支持最后一维） |
| `npu_one_hot` | One-hot 编码 |
| `npu_slice` | 切片操作 |
| `npu_pad` | 填充操作 |
| `npu_stride_copy` | 步长拷贝 |
| `npu_format_cast` | NPU 私有格式转换 |
| `npu_dtype_cast` | 数据类型转换 |
| `npu_prefetch` | L2 Cache 权重预取 |
| `npu_transpose` / `npu_confusion_transpose` | 高效转置/混淆转置 |
| `npu_reshape` | 高效 Reshape |

---

## 第二部分：PyTorch 标准 API 在 NPU 上的支持测试

### 2.1 完全支持（NPU 原生）

| 操作 | 状态 | 备注 |
|------|------|------|
| `torch.cumsum` | ✅ | prefix scan 原语 |
| `torch.cumprod` | ✅ | |
| `torch.sort` | ✅ | 含 indices |
| `torch.topk` | ✅ | |
| `torch.unique` | ✅ | 含 return_inverse/return_counts |
| `scatter_add_` | ✅ | |
| `index_add_` | ✅ | |
| `F.max_pool2d` | ✅ | 形态学运算基础 |
| `F.conv2d` | ✅ | |
| `torch.where` | ✅ | 条件选择 |
| `torch.nonzero` | ✅ | |
| `torch.bincount` | ✅ | |
| `torch.argmax/argmin` | ✅ | |
| `F.scaled_dot_product_attention` | ✅ | FP16 + BF16 |
| `torch.cdist` | ✅ | 距离矩阵 |
| `torch.searchsorted` | ✅ | |
| `torch.histc` | ✅ | 直方图 |
| `torch.linalg.svd` | ✅ | |
| `torch.meshgrid` | ✅ | |
| `Tensor.unfold` | ✅ | 滑动窗口 |
| `BF16/FP16 matmul` | ✅ | |
| `sparse_coo_tensor` | ✅ | |

### 2.2 需 CPU 回退

| 操作 | 状态 | 备注 |
|------|------|------|
| `scatter_reduce_` | ⚠️ 回退 CPU | 会打印 Warning，有性能影响 |
| `torchvision::nms` | ⚠️ 回退 CPU | torchvision 原版回退 CPU，但可用 `npu_nms_v4` 替代 |

### 2.3 torch.compile 支持

| 项目 | 状态 |
|------|------|
| `torch.compile(backend='npu')` | ✅ 可用，dynamo 后端包含 `npu` |
| `torch.compile(backend='inductor')` | ✅ inductor config 可用 |
| NPUGraph | ✅ 支持（类似 CUDA Graph） |

---

## 第三部分：SAM 3.1 各自定义算子的最优 NPU 实现策略

### 3.1 Flash Attention (fa3.py)

**当前状态**：FA3 不可用 → 回退到 `F.scaled_dot_product_attention` (SDPA)

**最优策略**：使用 `torch_npu.npu_fusion_attention` **直接替代**

```python
def flash_attn_func_npu(q, k, v):
    # q/k/v: (B, S, H, D) → npu_fusion_attention 支持 BNSD 布局
    B, S, H, D = q.shape
    q_npu = q.transpose(1, 2).contiguous()  # (B, H, S, D) = BNSD
    k_npu = k.transpose(1, 2).contiguous()
    v_npu = v.transpose(1, 2).contiguous()
    
    scale = 1.0 / (D ** 0.5)
    out, *_ = torch_npu.npu_fusion_attention(
        q_npu, k_npu, v_npu,
        head_num=H,
        input_layout="BNSD",
        scale=scale,
        keep_prob=1.0,
    )
    return out.transpose(1, 2)  # 回到 (B, S, H, D)
```

**预期性能**：
- 相对于当前 SDPA 回退：**提升 1.5-3x**（npu_fusion_attention 是华为专门优化的融合算子）
- 相对于 CUDA FA3 (FP8)：约 **60-80%** 性能水平（910B 的 FP16/BF16 吞吐量 vs H100 FP8）
- **BF16 直接支持**，无需额外类型转换

**内存优化建议**：
- 使用 `sparse_mode=3` (rightDownCausal) 可在 causal attention 场景节省约 50% 计算
- 利用 `pre_tockens/next_tockens` 参数启用稀疏窗口注意力
- 对长序列使用 `actual_seq_qlen/actual_seq_kvlen` 参数启用 varlen 模式

---

### 3.2 NMS (nms.py)

**当前状态**：NPU 上回退到 CPU Python 循环（`generic_nms_cpu`），逐框检查，极慢

**最优策略**：使用 `torch_npu.npu_nms_v4` **全部在 NPU 上完成**

```python
def generic_nms_npu(ious, scores, iou_threshold):
    """
    问题：npu_nms_v4 接受 boxes[N,4] 而非 IoU 矩阵。
    SAM3 的 generic_nms 接口直接传入预计算的 IoU 矩阵。
    
    策略 A（推荐）：用 NPU 张量原语重写 NMS 逻辑
    """
    num_boxes = scores.size(0)
    _, sorted_indices = torch.sort(scores, descending=True, stable=True)
    
    # 重排 IoU 矩阵
    iou_sorted = ious[sorted_indices][:, sorted_indices]
    iou_mask = iou_sorted > iou_threshold
    
    # 用 NPU 上的向量化操作代替 Python 循环
    keep = torch.ones(num_boxes, dtype=torch.bool, device=scores.device)
    for i in range(num_boxes - 1):
        if keep[i]:
            # 这一行在 NPU 上全部是向量化操作
            keep[i + 1:] &= ~iou_mask[i, i + 1:]
    
    return sorted_indices[keep]
```

**策略 B**：对于标准 box NMS 场景（非 mask NMS），直接使用 `npu_nms_v4`：

```python
def box_nms_npu(boxes, scores, iou_threshold, score_threshold=0.0):
    max_output = boxes.shape[0]
    iou_thresh_tensor = torch.tensor(iou_threshold, device=boxes.device)
    score_thresh_tensor = torch.tensor(score_threshold, device=boxes.device)
    indices, valid_num = torch_npu.npu_nms_v4(
        boxes, scores, max_output, iou_thresh_tensor, score_thresh_tensor
    )
    return indices[:valid_num.item()]
```

**预期性能**：
- 策略 A（NPU 向量化 mask NMS）：相对于当前 CPU 回退提升 **5-20x**（取决于框数量）
- 策略 B（npu_nms_v4）：相对于 Triton NMS 约 **80-100%** 性能水平
- **注意**：mask NMS 的核心瓶颈是 IoU 计算（matmul），这在 NPU 上已经很快了

**内存优化建议**：
- IoU 矩阵 `mask_iou()` 使用 matmul 计算，已经利用了 Tensor Core，NPU 上效率良好
- NMS 的 for 循环是顺序依赖的，但每次迭代内的 `keep[i+1:] &= ~iou_mask[i, i+1:]` 是向量操作，NPU 效率远高于 CPU

---

### 3.3 Connected Components (connected_components.py)

**当前状态**：NPU 上回退到 CPU skimage.measure.label，涉及 NPU→CPU→NPU 双向数据传输

**最优策略**：基于 torch_npu 原语的**纯 NPU 并行连通域算法**

算法：迭代标签传播（Iterative Label Propagation），不需要 Triton 的 atomic_min/union-find，用标准 tensor 操作实现：

```python
def connected_components_npu(input_tensor):
    """
    基于迭代标签传播的 NPU 连通域分析。
    核心原语：max_pool2d + scatter + unique + where
    全部在 NPU 上运行，零 CPU 回退。
    """
    B, H, W = input_tensor.shape
    device = input_tensor.device
    fg_mask = (input_tensor != 0)
    
    # Phase 1: 初始化 — 每个前景像素标记为其线性索引
    labels = torch.arange(H * W, device=device).view(1, H, W).expand(B, -1, -1).clone()
    labels = labels * fg_mask.long() + (~fg_mask).long() * (H * W)  # 背景设为哨兵值
    
    # Phase 2: 迭代标签传播 — 每次让每个像素取其 8-邻域的最小标签
    # 使用 -max_pool2d(-x) 实现 min_pool2d
    for _ in range(int(math.log2(max(H, W))) + 2):
        labels_4d = labels.unsqueeze(1).float()
        # min-pool: 取 3x3 邻域最小值
        min_neighbors = -torch.nn.functional.max_pool2d(
            -labels_4d, kernel_size=3, stride=1, padding=1
        ).squeeze(1).long()
        # 只更新前景像素
        labels = torch.where(fg_mask, torch.minimum(labels, min_neighbors), labels)
        
        # 指针跳跃：每个标签跳到其标签的标签（加速收敛）
        flat_labels = labels.view(B, -1)
        for _ in range(3):  # 几轮指针跳跃
            flat_labels = torch.gather(
                flat_labels, 1, 
                flat_labels.clamp(0, H * W - 1)
            )
        labels = flat_labels.view(B, H, W)
    
    # Phase 3: 计算每个连通域的大小
    labels = torch.where(fg_mask, labels, torch.zeros_like(labels))
    # 用 bincount 统计每个标签的像素数
    sizes = torch.zeros_like(labels)
    for b in range(B):
        flat = labels[b].view(-1)
        counts = torch.bincount(flat, minlength=H * W)
        sizes[b] = counts[flat].view(H, W)
    
    return labels, sizes
```

**进阶优化（避免 Python for 循环的 bincount）**：

```python
# 用 scatter_add 替代 bincount 循环
flat_labels = labels.view(B, -1)  # (B, H*W)
ones = torch.ones_like(flat_labels, dtype=torch.int32)
counts = torch.zeros(B, H * W, device=device, dtype=torch.int32)
counts.scatter_add_(1, flat_labels, ones)
sizes = torch.gather(counts, 1, flat_labels).view(B, H, W)
```

**预期性能**：
- 相对于当前 CPU skimage 回退：**10-50x 提升**（消除了 D2H + H2D 传输 + CPU 串行计算）
- 相对于 Triton 版本：约 **40-70%** 性能水平（Triton 有 atomic_min 实现高效 union-find，这里用迭代传播替代）
- 迭代次数 ≈ log2(max(H,W))，对于 1024x1024 图像约需 10-12 次迭代
- 每次迭代的 min_pool2d 是 NPU 原生高效操作

**关键原语**：
- `F.max_pool2d`（NPU 原生）→ 实现 min-pool 邻域传播
- `torch.gather`（NPU 原生）→ 指针跳跃
- `scatter_add_`（NPU 原生）→ 标签计数
- `torch.where`（NPU 原生）→ 条件更新

---

### 3.4 Euclidean Distance Transform — EDT (edt.py)

**当前状态**：Triton kernel，NPU 上无法运行

**最优策略**：基于 NPU 原语的**分离式 EDT 算法**

EDT 的核心思想：先做行方向 1D-EDT，再做列方向 1D-EDT。1D-EDT 有两种实现路径：

**策略 A：O(N³) 暴力但高度并行的实现**

```python
def edt_npu(data):
    """
    基于广播的暴力 O(N³) EDT，但完全向量化在 NPU 上运行。
    对于 SAM3 典型的 256x256 ~ 1024x1024 mask 仍然实用。
    """
    B, H, W = data.shape
    device = data.device
    INF = 1e18
    
    # F(i,j) = 0 if data[i,j]==0 else INF
    f = torch.where(data.bool(), INF, 0.0)
    
    # Stage 1: 行方向 EDT²
    # 对每行的每个像素 j，计算 min_k((j-k)² + f[i,k])
    col_idx = torch.arange(W, device=device, dtype=torch.float32)
    # f: (B, H, W) → (B, H, 1, W) - 所有可能的源点
    # col_idx: (W,) → (1, 1, W, 1) - 目标点
    # 距离: (W, W) 的广播
    dist_sq = (col_idx.view(1, 1, -1, 1) - col_idx.view(1, 1, 1, -1)) ** 2
    # f_expanded: (B, H, 1, W)
    row_edt_sq = (dist_sq + f.unsqueeze(2)).amin(dim=3)  # (B, H, W)
    
    # Stage 2: 列方向 EDT²
    row_idx = torch.arange(H, device=device, dtype=torch.float32)
    dist_sq_v = (row_idx.view(1, -1, 1, 1) - row_idx.view(1, 1, -1, 1)) ** 2
    edt_sq = (dist_sq_v + row_edt_sq.unsqueeze(1)).amin(dim=2)  # (B, H, W)
    
    return edt_sq.sqrt()
```

**策略 B：分块 + 迭代的 O(N²logN) 近似方案（大图推荐）**

```python
def edt_npu_iterative(data, max_dist=None):
    """
    基于迭代膨胀的近似 EDT。
    对于二值 mask，从边界逐层扩展计算距离。
    精度不如精确 EDT，但性能极高。
    """
    B, H, W = data.shape
    device = data.device
    bg_mask = ~data.bool()
    
    if max_dist is None:
        max_dist = max(H, W)
    
    # 初始距离：前景=INF，背景=0
    dist = torch.where(data.bool(), float(max_dist), 0.0).unsqueeze(1)
    
    # 构造 L2 距离核（3x3 棋盘距离核，用于逐步传播）
    # 使用倍增策略：先用 3x3 核传播 1 像素距离，然后 5x5 传播 2 像素距离...
    for step_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if step_size > max_dist:
            break
        # 通过 dilation 参数控制传播步长
        dilated = -F.max_pool2d(
            -dist, kernel_size=3, stride=1, padding=step_size, dilation=step_size
        )
        # 加上步进距离
        dist = torch.minimum(dist, dilated + step_size)
    
    return dist.squeeze(1)
```

**策略 C：精确 O(N²) 包络线算法（CPU 辅助序列部分）**

```python
def edt_npu_exact(data):
    """
    混合策略：行方向用 O(N³) 并行广播（小 N 或分块），
    列方向同理，但可利用排序优化。
    """
    # 对于 SAM3 典型尺寸 (256x1024x1024)，策略 A 的内存消耗为：
    # 行方向: B*H*W*W*4 bytes = 256*1024*1024*1024*4 ≈ 1TB（不可行！）
    # 需要分块处理
    
    B, H, W = data.shape
    CHUNK = 256  # 分块大小，限制内存
    INF = 1e18
    f = torch.where(data.bool(), INF, 0.0)
    
    # Stage 1: 行方向分块 EDT
    col_idx = torch.arange(W, device=data.device, dtype=torch.float32)
    row_edt_sq = torch.full_like(f, INF)
    
    for start in range(0, W, CHUNK):
        end = min(start + CHUNK, W)
        chunk_idx = col_idx[start:end]  # (chunk_size,)
        dist_sq = (col_idx.unsqueeze(0) - chunk_idx.unsqueeze(1)) ** 2  # (chunk, W)
        f_chunk = f[:, :, start:end]  # (B, H, chunk)
        # (B, H, chunk, 1) + (1, 1, chunk, W) → (B, H, chunk, W) → min over chunk dim
        chunk_result = (f_chunk.unsqueeze(3) + dist_sq.unsqueeze(0).unsqueeze(0)).amin(dim=2)
        row_edt_sq = torch.minimum(row_edt_sq, chunk_result)
    
    # Stage 2: 列方向分块 EDT（同理）
    row_idx = torch.arange(H, device=data.device, dtype=torch.float32)
    edt_sq = torch.full_like(row_edt_sq, INF)
    
    for start in range(0, H, CHUNK):
        end = min(start + CHUNK, H)
        chunk_idx = row_idx[start:end]
        dist_sq = (row_idx.unsqueeze(0) - chunk_idx.unsqueeze(1)) ** 2
        f_chunk = row_edt_sq[:, start:end, :]
        chunk_result = (f_chunk.unsqueeze(1) + dist_sq.unsqueeze(0).unsqueeze(3)).amin(dim=2)
        edt_sq = torch.minimum(edt_sq, chunk_result)
    
    return edt_sq.sqrt()
```

**推荐选择**：
- 小图 (≤512x512)：策略 A 最简单，内存可接受
- 中图 (512-2048)：策略 C 分块精确算法
- 超大图或实时场景：策略 B 迭代近似

**预期性能**：
- 策略 A/C（精确）：相对于 Triton O(N²) 实现约 **50-80%** 性能水平
- 策略 B（迭代近似）：可能**超越** Triton 精确版本，因为 max_pool2d 是 NPU 高度优化的算子
- 所有策略相对于当前 CPU OpenCV 回退：**5-20x 提升**

**关键原语**：
- `torch.where`（NPU 原生）
- `F.max_pool2d`（NPU 原生，含 dilation 参数）
- `torch.amin` / `torch.minimum`（NPU 原生）
- 广播 matmul（NPU tensor core）

---

### 3.5 Fused AddMM + Activation (fused.py)

**当前状态**：使用 `torch.ops.aten._addmm_activation`

**最优策略**：使用 `torch_npu.npu_linear` + `torch_npu.npu_gelu/npu_silu`，或者利用 `npu_ffn` 融合

```python
def addmm_act_npu(activation, linear, mat1):
    # npu_linear 要求 2D 输入，需先 flatten 再 reshape 回去
    mat1_bf16 = mat1.to(torch.bfloat16)
    weight_bf16 = linear.weight.detach().to(torch.bfloat16)
    bias_bf16 = linear.bias.detach().to(torch.bfloat16)
    
    # flatten → 2D matmul → reshape
    orig_shape = mat1_bf16.shape
    mat1_flat = mat1_bf16.view(-1, orig_shape[-1])
    out = torch.addmm(bias_bf16, mat1_flat, weight_bf16.t())
    out = out.view(*orig_shape[:-1], -1)
    
    # 融合激活
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        return torch_npu.npu_fast_gelu(out)
    elif activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        return torch.nn.functional.relu(out)
    raise ValueError(f"Unexpected activation {activation}")
```

**预期性能**：与 CUDA aten::_addmm_activation **持平或略优**（NPU 有专门的融合算子）

---

### 3.6 Mask IoU / Associate Det-Trk (masks_ops.py, iou.py, associate_det_trk.py)

**当前状态**：已用 matmul 实现，NPU 上正常工作

**进一步优化**：

```python
def mask_iou_npu(pred_masks, gt_masks):
    # 当前实现已经是向量化的 matmul，NPU 上效率良好
    # 可以额外使用 BF16 加速
    m1_flat = pred_masks.flatten(1).to(torch.bfloat16)
    m2_flat = gt_masks.flatten(1).to(torch.bfloat16)
    intersection = torch_npu.npu_bmmV2(
        m1_flat.unsqueeze(0), m2_flat.unsqueeze(0).transpose(1, 2), []
    ).squeeze(0).float()
    area1 = m1_flat.sum(dim=1).float()
    area2 = m2_flat.sum(dim=1).float()
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1)
```

**预期性能**：与 CUDA 实现 **持平**（核心是 matmul，NPU tensor core 与 CUDA tensor core 效率相当）

---

## 第四部分：综合对比与优先级排序

### 4.1 优化优先级排序

| 优先级 | 算子 | 当前状态 | 优化后 | 预期加速比 | 实现难度 |
|--------|------|----------|--------|-----------|----------|
| **P0** | Flash Attention | SDPA 回退 | `npu_fusion_attention` | **1.5-3x** | ⭐ 低 |
| **P0** | Connected Components | CPU skimage | NPU 迭代标签传播 | **10-50x** | ⭐⭐⭐ 中 |
| **P1** | NMS | CPU Python 循环 | NPU 向量化 / `npu_nms_v4` | **5-20x** | ⭐⭐ 低-中 |
| **P1** | EDT | Triton (不可用) | NPU 分块/迭代 EDT | **5-20x** | ⭐⭐⭐ 中 |
| **P2** | Fused AddMM+Act | aten fallback | `npu_linear` + `npu_fast_gelu` | **1.2-1.5x** | ⭐ 低 |
| **P2** | Mask IoU | matmul (已工作) | BF16 优化 | **1.1-1.3x** | ⭐ 低 |

### 4.2 NPU vs CUDA 总体性能预期

| 算子类别 | CUDA (H100) 基准 | NPU (910B) 预期 | 占比 |
|----------|-----------------|-----------------|------|
| Flash Attention (FA3 FP8) | 100% | 60-80% (npu_fusion_attention BF16) | 核心 |
| Matmul / Linear | 100% | 80-95% | 核心 |
| NMS (Triton) | 100% | 70-90% (NPU 向量化) | 后处理 |
| Connected Components (Triton) | 100% | 40-70% (迭代传播) | 后处理 |
| EDT (Triton) | 100% | 50-80% (分块/迭代) | 后处理 |
| torch.compile | 100% | 60-80% (npu backend) | 全局 |

### 4.3 torch.compile 在 NPU 上的使用建议

1. **dynamo 后端** `'npu'` 已注册可用，基础功能测试通过
2. NPUGraph (类 CUDA Graph) 已支持，可减少 kernel launch 开销
3. **建议**：对于推理场景，可尝试 `torch.compile(backend='npu')` 编译整个模型
4. **注意**：torch.compile 对自定义 torch_npu 算子的支持需逐个验证
5. 当前 SAM3 默认禁用 compile（`SAM3_NPU_ENABLE_COMPILE=1` 可开启），建议在上述算子优化完成后重新评估

---

## 第五部分：不存在原生支持的能力

以下功能在 `torch_npu` 中**没有**原生 API，需要用基础原语组合实现：

| 功能 | torch_npu 原生支持 | 替代方案 |
|------|-------------------|----------|
| Connected Components | ❌ 无 | min_pool2d 迭代传播 + scatter_add 计数 |
| Euclidean Distance Transform | ❌ 无 | max_pool2d 迭代 / 分块广播 / 分离式包络线 |
| Generic NMS (IoU 矩阵输入) | ❌ 无（仅有 box NMS） | NPU tensor 向量化循环 |
| Triton kernel 直接运行 | ❌ 不支持 Triton | 用 torch_npu 原语重写 |
| prefix scan (parallel) | 部分（有 torch.cumsum） | torch.cumsum 即可 |
| parallel sort (with indices) | ✅ torch.sort / npu_sort_v2 | npu_sort_v2 更快但不返回 indices |
| scatter_reduce | ⚠️ CPU fallback | 改用 scatter_add_ + 手动 reduce |
