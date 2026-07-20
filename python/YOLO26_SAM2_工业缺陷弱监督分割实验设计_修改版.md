# 基于 YOLO26 与 SAM2 的工业缺陷弱监督分割实验设计

## 1. 研究定位

本研究围绕“检测框提示的通用分割模型能否降低工业缺陷分割的标注成本”展开。前期 PPT 已经明确主线：以 YOLO 系列模型负责缺陷候选区域检测，以 SAM2 根据检测框生成像素级掩码，最终形成“detect -> prompt -> segment -> rule statistics”的工业缺陷检测与辅助标注流程。

核心问题：

在 MVTec AD 等工业异常检测数据集上，仅使用 bounding box 作为主要监督信号时，YOLO26 -> SAM2 框提示分割能否获得接近全监督 YOLO-seg 的分割效果，同时显著降低人工 mask 标注成本？

建议论文题目可表述为：

《基于 YOLO26 框提示与 SAM2 的工业表面缺陷弱监督分割方法研究》

或更聚焦：

《面向 MVTec 工业缺陷数据集的 YOLO26-SAM2 弱监督分割方法研究》

## 2. 实验目标与假设

### 2.1 实验目标

1. 建立 MVTec AD 数据集上的 YOLO 检测 baseline。
2. 建立 YOLO-seg 全监督分割 baseline，作为 mask 标注上限参考。
3. 建立 YOLO26 -> SAM2 框提示分割流程，作为本文核心方法。
4. 比较不同方法在检测精度、分割精度、推理速度、标注成本和失败场景上的差异。
5. 形成可复现的训练、测试、可视化和统计代码框架。

### 2.2 实验假设

H1：YOLO 检测框可以为 SAM2 提供有效 prompt，使其在工业表面缺陷图像上生成可用的缺陷 mask。

H2：YOLO26 -> SAM2 的分割 Dice / IoU 低于全监督 YOLO-seg，但在大部分缺陷类别上可以达到接近可用的水平。

H3：与逐像素 mask 标注相比，box 标注可以显著降低标注时间，适合作为工业场景中的低成本辅助标注方案。

H4：YOLO26 -> SAM2 的主要失败场景会集中在小目标、低对比度缺陷、反光纹理、边界模糊、多个缺陷粘连和检测框偏移。

## 3. 数据集选择

### 3.1 主数据集：MVTec AD

MVTec AD 是工业异常检测常用公开数据集，包含多类纹理和物体缺陷图像。其原始组织方式通常为：

- `train/good`：正常样本。
- `test/good`：测试集中正常样本。
- `test/<defect_type>`：测试集中异常样本。
- `ground_truth/<defect_type>`：异常区域的像素级 mask。

本项目建议优先选择 3-5 个类别进行主实验，而不是一次性覆盖全部类别。推荐起步类别：

- `bottle`：轮廓清晰，适合验证框提示分割。
- `hazelnut`：存在局部缺陷，适合分析小目标。
- `metal_nut`：接近“小型金属零件”方向。
- `screw`：小目标和细长结构较多，适合作为困难类别。
- `capsule` 或 `cable`：用于测试复杂纹理/结构。

### 3.2 备选数据集：BTAD

BTAD 可作为补充验证集，用于证明方法不是只对 MVTec 有效。建议先完成 MVTec 主实验，再用 BTAD 做小规模迁移验证。

## 4. 数据划分与预处理

### 4.1 数据划分原则

MVTec AD 的原始训练集多为正常样本，异常样本主要在测试集中。为了训练 YOLO 检测/分割模型，需要从异常测试图像和对应 mask 中构造监督标签。

建议采用两套划分：

#### A. 主实验划分

用于比较 YOLO、YOLO-seg、YOLO26 -> SAM2。

| 子集 | 比例 | 用途 |
|---|---:|---|
| Train | 70% abnormal + 部分 good | 训练 YOLO 检测/YOLO-seg |
| Val | 10% abnormal + 部分 good | 调参、早停、选择 checkpoint |
| Test | 20% abnormal + 部分 good | 最终指标报告 |

#### B. 少标注实验划分

用于验证“低标注成本”。

| 设置 | box 标注比例 | mask 标注比例 | 目的 |
|---|---:|---:|---|
| Full-mask baseline | 100% | 100% | YOLO-seg 上限 |
| Box-only main | 100% | 0% train, 100% test eval | YOLO26 -> SAM2 主方法 |
| Few-mask adapter | 100% | 5% / 10% / 20% | 可选：轻量微调或筛选规则 |

说明：即使训练阶段不使用 mask，测试阶段仍需要 ground truth mask 来计算 IoU、Dice 等分割指标。

### 4.2 标签转换

从 MVTec 的 ground truth mask 生成：

1. 检测框标签：由 mask 的最小外接矩形得到 YOLO detection label。
2. 分割标签：由 mask 轮廓转换为 YOLO segmentation polygon label。
3. 评估掩码：保留原始 mask，用于 IoU、Dice、Pixel-AUROC 等计算。

### 4.3 图像预处理

| 步骤 | 说明 | 是否固定 |
|---|---|---|
| Resize | 统一输入尺寸，如 640 或 1024 | 是 |
| Normalize | 使用模型默认归一化 | 是 |
| Data augmentation | flip、scale、HSV、mosaic 等 | 训练用 |
| Mask binarization | ground truth mask 转为 0/1 | 是 |
| Small region filter | 删除面积小于阈值的噪声区域 | 是 |

建议先用 `imgsz=640` 跑通 baseline，再尝试 `imgsz=1024` 观察小缺陷提升。

## 5. 方法结构

### 5.1 Baseline 1：YOLO Detection

目标：只检测缺陷位置，输出 bounding box。

流程：

1. 从 mask 生成 bbox 标签。
2. 使用 Ultralytics YOLO detection 模型训练。
3. 评估 Precision、Recall、mAP@0.5、mAP@0.5:0.95。
4. 保存预测框，用作 SAM2 prompt。

作用：

- 保底实验。
- 验证检测框是否足够准确。
- 为 YOLO26 -> SAM2 提供输入。

### 5.2 Baseline 2：YOLO-seg Full Supervision

目标：使用完整 mask/polygon 标签训练实例分割模型。

流程：

1. 从 ground truth mask 转换 polygon。
2. 使用 YOLO segmentation 模型训练。
3. 评估 mask mAP、mIoU、Dice。

作用：

- 作为全监督分割性能上限。
- 对比 box-only 方法与 mask-supervised 方法的差距。

### 5.3 Main Method：YOLO26 -> SAM2

目标：用 YOLO 输出 bbox，再将 bbox 作为 SAM2 prompt，生成缺陷 mask。

流程：

1. 输入测试图像。
2. YOLO26 检测缺陷 bbox。
3. 根据置信度筛选 bbox。
4. 将 bbox 输入 SAM2。
5. 得到候选 mask。
6. 后处理 mask：选择最大连通域、过滤小面积区域、与 bbox 约束相交。
7. 输出最终 mask 与面积统计。

核心优势：

- 训练阶段主要需要 box 标签。
- SAM2 提供边界细化能力。
- 可以用于自动生成 pseudo mask，支持辅助标注。

### 5.4 Optional Extension：SAM2 Pseudo-label -> YOLO-seg

作为进一步改进，可使用 YOLO26 -> SAM2 自动生成 pseudo mask，再训练 YOLO-seg。

流程：

1. YOLO26 -> SAM2 生成 pseudo mask。
2. 通过规则过滤低质量 mask。
3. 人工抽样检查并修正一小部分。
4. 用 pseudo mask 训练 YOLO-seg。
5. 比较其与人工 full-mask YOLO-seg 的差距。

该部分适合写作中的“改进方法”或“未来工作”，不建议作为第一阶段必须完成项。

## 6. 控制变量设计

为了保证对比公平，除待比较因素外，其余变量应尽量固定。

| 变量 | 推荐设置 | 说明 |
|---|---|---|
| Dataset split | 固定 train/val/test 文件列表 | 所有方法使用同一划分 |
| Image size | 640，扩展实验用 1024 | 小缺陷可尝试高分辨率 |
| Epoch | 100 baseline，扩展 200 | 所有训练方法一致 |
| Batch size | 8 或 16 | 根据显存调整，但记录清楚 |
| Learning rate | 0.01 或模型默认 | 保持一致 |
| Optimizer | SGD 或 AdamW | 建议先用 Ultralytics 默认 |
| Loss | YOLO 默认检测/分割 loss | 不改 loss 作为公平 baseline |
| Random seed | 42、3407、2026 | 至少固定一个，正式实验可跑三次 |
| Device | 同一 GPU | 记录 GPU 型号 |
| Conf threshold | 0.25 | SAM2 prompt 前固定 |
| NMS IoU | 0.7 | YOLO 默认或固定 |
| SAM2 variant | 固定 SAM2 权重，例如 sam2.1_t.pt 或 sam2.1_b.pt | 记录 SAM2 版本和权重；正式实验中保持不变，避免因模型权重变化影响对比 |
| Postprocess | 固定面积阈值和连通域规则 | 避免测试集人工调参 |
| Train code | 同一训练脚本 | 保存配置文件 |
| Test code | 同一测试脚本 | 保存输出结果 |

建议保存每次实验的：

- `config.yaml`
- `train.log`
- `results.csv`
- `best.pt`
- `predictions/`
- `metrics.json`
- `visualizations/`

## 7. 训练流程与实验顺序

本研究的实验应按照“先 baseline，后主实验，再补上限与消融”的顺序执行。原因是 YOLO26 -> SAM2 主实验依赖 YOLO detection 输出的预测框；如果检测框质量不足，SAM2 后续分割效果也会受到直接影响。因此，YOLO Detection Baseline 是整个流程的前置实验。

### 7.1 阶段 1：数据准备

1. 下载并解压 MVTec AD。
2. 选择实验类别，建议先选择 3 个类别，后续再扩展到 5 个类别。
3. 从 ground truth mask 生成 bbox 标签，用于 YOLO detection 训练。
4. 从 ground truth mask 生成 polygon 标签，用于 YOLO-seg baseline。
5. 固定 train/val/test 划分，所有方法使用同一划分。
6. 检查标签可视化，确认 bbox、polygon、mask 与原图对齐。

### 7.2 阶段 2：YOLO Detection Baseline

目标是先建立检测基准模型，并为后续 YOLO26 -> SAM2 主实验提供预测框。

1. 使用 bbox 标签训练 YOLO26 detection 模型。
2. 记录训练 loss、val mAP、Precision、Recall。
3. 在 test set 上保存预测框、置信度和类别结果。
4. 分析漏检、误检、小目标检测失败等问题。
5. 将预测框保存为后续 SAM2 prompt 输入。

### 7.3 阶段 3：主实验 YOLO26 -> SAM2

目标是验证仅使用 box 监督时，YOLO26 检测框能否作为 SAM2 prompt 生成可用的缺陷 mask。

1. 加载训练好的 YOLO26 detection 模型。
2. 对测试图像预测 bbox。
3. 根据固定 conf threshold 筛选预测框。
4. 将 YOLO26 预测框作为 prompt 输入 SAM2。
5. 对 SAM2 输出 mask 进行统一后处理，包括面积过滤、最大连通域选择和 bbox 约束。
6. 与 ground truth mask 计算 mIoU、Dice、Boundary-F1 和推理时间。
7. 可视化 image + YOLO bbox + SAM2 mask + GT mask。

### 7.4 阶段 4：YOLO-seg Full Supervision Baseline

目标是建立全监督分割上限，用于衡量 YOLO26 -> SAM2 弱监督分割方法与完整 mask 监督方法之间的差距。

1. 使用 polygon/mask 标签训练 YOLO-seg 模型。
2. 记录 box loss、mask loss、cls loss、mAP 等训练指标。
3. 在 test set 上输出预测 mask。
4. 与 ground truth mask 计算 mIoU、Dice、Boundary-F1。
5. 将 YOLO-seg 结果与 YOLO26 -> SAM2 结果进行对比。

### 7.5 阶段 5：GT-box -> SAM2 上限实验

目标是区分“YOLO 检测框误差”和“SAM2 分割能力限制”。

1. 由 ground truth mask 生成真实 bbox。
2. 将真实 bbox 输入 SAM2。
3. 计算 GT-box -> SAM2 的 mIoU、Dice、Boundary-F1。
4. 与 YOLO pred-box -> SAM2 对比。
5. 如果 GT-box -> SAM2 明显优于 pred-box -> SAM2，说明主要瓶颈在 YOLO 检测框质量；如果两者都较低，说明 SAM2 对该类缺陷本身分割困难。

### 7.6 阶段 6：消融实验与结果整理

建议至少完成以下消融：

| 实验 | 改动因素 | 目的 |
|---|---|---|
| YOLO GT-box -> SAM2 | 用真实 bbox prompt | 测 SAM2 本身上限 |
| YOLO pred-box -> SAM2 | 用预测 bbox prompt | 测真实流程表现 |
| Conf threshold | 0.1 / 0.25 / 0.5 | 测 prompt 质量影响 |
| Image size | 640 / 1024 | 测小缺陷收益 |
| SAM2 weight | sam2.1_t / sam2.1_b，可选 | 比较轻量模型与较大模型的精度和速度差异；时间不足时只固定一个 SAM2 权重 |
| Postprocess | 无 / 面积过滤 / 连通域 | 测后处理贡献 |

阶段 6 结束前应完成主结果表、类别分结果表、消融实验表、训练曲线、可视化样例和失败案例分析。

## 8. 评价指标

### 8.1 检测指标

| 指标 | 含义 |
|---|---|
| Precision | 预测为缺陷的框中有多少是真的 |
| Recall | 真实缺陷中有多少被检测到 |
| mAP@0.5 | IoU 阈值 0.5 下平均精度 |
| mAP@0.5:0.95 | 多 IoU 阈值下平均精度 |

### 8.2 分割指标

| 指标 | 公式/含义 |
|---|---|
| IoU | `intersection / union` |
| Dice | `2 * intersection / (pred + gt)` |
| mIoU | 多图像或多类别平均 IoU |
| Boundary-F1 | 评价边界贴合程度 |
| Pixel Accuracy | 像素级准确率，类别不平衡时仅作辅助 |

### 8.3 速度与成本指标

| 指标 | 说明 |
|---|---|
| YOLO inference time | 单图检测时间 |
| SAM2 inference time | 单图分割时间 |
| End-to-end time | YOLO26 + SAM2 总时间 |
| Box annotation time | 单图 box 标注时间 |
| Mask annotation time | 单图 mask 标注时间 |
| Cost reduction ratio | `mask_time / box_time` |

### 8.4 论文中推荐主指标

主表格建议使用：

- Detection：Precision、Recall、mAP@0.5、mAP@0.5:0.95
- Segmentation：mIoU、Dice、Boundary-F1
- Efficiency：FPS 或 ms/image
- Annotation：annotation time/image、cost reduction ratio

## 9. 结果表格模板

### 9.1 主结果表

| Method | Supervision | Prompt | mIoU ↑ | Dice ↑ | Boundary-F1 ↑ | Time ms/img ↓ | Annotation time/img ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| YOLO detection | Box | None | - | - | - | TBD | 30s |
| YOLO-seg | Mask | None | TBD | TBD | TBD | TBD | 20-30min |
| YOLO GT-box -> SAM2 | GT box | Box | TBD | TBD | TBD | TBD | 30s |
| YOLO pred-box -> SAM2 | Box | Pred box | TBD | TBD | TBD | TBD | 30s |

### 9.2 类别分结果表

| Category | Method | Precision | Recall | mAP@0.5 | mIoU | Dice | Failure notes |
|---|---|---:|---:|---:|---:|---:|---|
| bottle | YOLO26 -> SAM2 | TBD | TBD | TBD | TBD | TBD | TBD |
| metal_nut | YOLO26 -> SAM2 | TBD | TBD | TBD | TBD | TBD | TBD |
| screw | YOLO26 -> SAM2 | TBD | TBD | TBD | TBD | TBD | TBD |

### 9.3 消融实验表

| Setting | Changed variable | mIoU ↑ | Dice ↑ | Recall ↑ | Time ↓ | Conclusion |
|---|---|---:|---:|---:|---:|---|
| GT-box -> SAM2 | prompt source | TBD | TBD | - | TBD | SAM2 upper bound |
| Pred-box -> SAM2 | prompt source | TBD | TBD | TBD | TBD | real pipeline |
| imgsz=1024 | resolution | TBD | TBD | TBD | TBD | small defect effect |
| no postprocess | postprocess | TBD | TBD | TBD | TBD | rule effect |

## 10. Loss / Accuracy 曲线方案

### 10.1 训练曲线

YOLO detection：

- train/box_loss
- train/cls_loss
- train/dfl_loss
- val/box_loss
- metrics/precision
- metrics/recall
- metrics/mAP50
- metrics/mAP50-95

YOLO-seg：

- train/box_loss
- train/seg_loss
- train/cls_loss
- val/box_loss
- val/seg_loss
- metrics/mAP50(B)
- metrics/mAP50(M)

### 10.2 分割评估曲线

对于 YOLO26 -> SAM2：

- conf threshold vs Dice
- conf threshold vs mIoU
- image size vs Dice
- inference time vs image size
- bbox IoU vs mask Dice

其中 `bbox IoU vs mask Dice` 很有价值，可以说明检测框质量对 SAM2 分割质量的影响。

## 11. 可视化设计

### 11.1 必做可视化

每个类别至少展示：

1. 原图。
2. Ground truth mask。
3. YOLO 检测框。
4. SAM2 预测 mask。
5. YOLO-seg 预测 mask。
6. 错误区域图：FP 用红色，FN 用蓝色，TP 用绿色。

### 11.2 论文图建议

| 图编号 | 内容 |
|---|---|
| Fig. 1 | 整体框架：image -> YOLO26 -> bbox prompt -> SAM2 -> mask -> statistics |
| Fig. 2 | 数据集样例与标签转换 |
| Fig. 3 | 三类方法可视化对比 |
| Fig. 4 | loss/mAP 曲线 |
| Fig. 5 | bbox 质量与 mask 质量关系 |
| Fig. 6 | 失败案例分析 |

## 12. 问题与改进

### 12.1 可能问题

| 问题 | 原因 | 解决方向 |
|---|---|---|
| YOLO 漏检小缺陷 | 缺陷面积小、对比度低 | 提高 imgsz、类别重采样、切图训练 |
| SAM2 mask 超出缺陷区域 | 框内背景复杂 | bbox 约束、最大连通域、面积过滤 |
| SAM2 无法分割弱纹理缺陷 | 缺陷边界不明显 | 加点提示、融合边缘/热力图 |
| YOLO-seg 效果不稳定 | mask 数量少 | 数据增强、pseudo mask 扩充 |
| 类别不平衡 | good 远多于 defect | 缺陷样本重采样、按类别报告 |
| 推理速度慢 | SAM2 模型较大 | 蒸馏、轻量 SAM、离线辅助标注 |

### 12.2 改进路线

优先级从高到低：

1. 提高 YOLO 检测框质量，因为它直接决定 SAM2 prompt 质量。
2. 对 SAM2 输出做统一后处理，避免人工挑选案例。
3. 增加 GT-box -> SAM2 上限实验，区分“检测问题”和“分割问题”。
4. 引入 pseudo mask 筛选，训练轻量 YOLO-seg。
5. 对失败案例按缺陷类型分类，形成论文分析亮点。

## 13. 与 Literature Review 和 Introduction 的衔接

### 13.1 Introduction 可写逻辑

1. 工业视觉检测需要从图像中定位并量化缺陷。
2. 深度学习方法在工业检测中表现良好，但像素级 mask 标注昂贵。
3. YOLO 系列模型适合高效检测，SAM2 模型具有强泛化分割能力。
4. 因此，使用 YOLO 检测框作为 SAM2 prompt，可能在低标注成本下获得可用分割结果。
5. 本文在 MVTec AD 上系统比较 YOLO detection、YOLO-seg 和 YOLO26 -> SAM2，评估精度、速度、成本和失败边界。

### 13.2 Literature Review 可分三段

1. Industrial anomaly detection and segmentation：介绍 MVTec AD、工业缺陷分割任务、传统/深度方法。
2. YOLO-based detection and segmentation：介绍 YOLO 在实时检测与分割中的优势。
3. Promptable segmentation and SAM2：介绍 SAM/SAM2 的 prompt 分割能力，以及其在弱监督/辅助标注中的潜力。

### 13.3 Contribution 建议写法

本文贡献可以写成：

1. 构建了一个面向工业缺陷的 YOLO26 -> SAM2 弱监督分割流程，仅依赖框级标注生成像素级缺陷掩码。
2. 在 MVTec AD 数据集上系统比较检测、全监督分割和框提示分割方法，量化精度、速度和标注成本之间的权衡。
3. 通过类别分结果和失败案例分析，总结 YOLO26-SAM2 流程在小目标、低对比度和边界模糊缺陷中的适用边界。

## 14. 推荐执行顺序与时间安排

整体时间目标：尽可能在 **2026 年 7 月 18 日前基本完成实验主体工作**；最迟在 **2026 年 7 月 25 日前完成阶段 6**，即完成除论文写作以外的全部实验、消融、结果整理和可视化工作。7 月 18 日至 7 月 25 日作为缓冲期，用于补充失败类别、修正代码、重跑异常实验、整理最终表格和可视化结果。

| 阶段 | 任务内容 | 截止日期 | 交付结果 |
|---|---|---|---|
| 阶段 1：数据准备 | 选择 MVTec AD 实验类别；固定 train/val/test；完成 mask -> bbox 和 mask -> polygon；检查标签可视化 | **7 月 3 日前** | 数据划分文件、bbox 标签、polygon 标签、标签检查图 |
| 阶段 2：YOLO Detection Baseline | 训练 YOLO26 detection；记录 Precision、Recall、mAP@0.5、mAP@0.5:0.95；保存 test set 预测框 | **7 月 6 日前** | detection best.pt、results.csv、预测框、检测可视化 |
| 阶段 3：主实验 YOLO26 -> SAM2 | 使用 YOLO26 预测框作为 SAM2 prompt；生成 mask；计算 mIoU、Dice、Boundary-F1 和推理时间 | **7 月 10 日前** | SAM2 mask、分割指标、主方法可视化 |
| 阶段 4：YOLO-seg Full Supervision Baseline | 使用 full mask / polygon 标签训练 YOLO-seg；作为全监督分割上限参考 | **7 月 13 日前** | YOLO-seg best.pt、mask mAP、mIoU、Dice、可视化 |
| 阶段 5：GT-box -> SAM2 上限实验 | 使用真实 bbox 作为 SAM2 prompt；比较 GT-box 和 YOLO pred-box 的分割差异 | **7 月 16 日前** | GT-box -> SAM2 指标、与 pred-box -> SAM2 对比结果 |
| 阶段 6：消融实验与结果整理 | 完成 conf threshold、image size、postprocess 等消融；整理主结果表、类别分结果表、失败案例和曲线图 | **7 月 18 日前基本完成；7 月 25 日前最终完成** | 主结果表、消融表、类别表、训练曲线、失败案例图、论文结果图 |
| 阶段 7：论文写作与 PPT 整理 | 回填 Introduction、Literature Review、Experiment、Results、Discussion；整理 PPT 框架图和结果图 | **7 月 25 日后继续推进** | 论文初稿、PPT 初稿、最终图表 |

### 14.1 最低完成标准

如果时间紧张，7 月 25 日前至少应完成以下内容：

1. 固定 3 个 MVTec AD 类别。
2. 完成 YOLO26 detection baseline。
3. 完成 YOLO26 -> SAM2 主实验。
4. 完成 GT-box -> SAM2 上限实验。
5. 完成 YOLO-seg full supervision baseline。
6. 至少完成 conf threshold 或 postprocess 中的一组消融实验。
7. 完成主结果表、类别结果表、可视化对比图和失败案例分析。

### 14.2 优先级说明

优先级从高到低为：

1. YOLO Detection Baseline，因为它决定 SAM2 prompt 质量。
2. YOLO26 -> SAM2 主实验，因为这是论文核心方法。
3. GT-box -> SAM2 上限实验，因为它能区分检测误差和分割误差。
4. YOLO-seg Full Supervision Baseline，因为它提供 full-mask 上限参考。
5. 消融实验和失败案例分析，因为它们增强论文说服力。
6. BTAD 迁移验证和 pseudo mask -> YOLO-seg 扩展实验，如时间不足可放入未来工作。

## 15. 参考资料

- MVTec AD official dataset page: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- MVTec AD paper: https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf
- Ultralytics YOLO Python usage: https://docs.ultralytics.com/usage/python/
- Ultralytics YOLO training mode: https://docs.ultralytics.com/modes/train/
- Ultralytics SAM documentation: https://docs.ultralytics.com/models/sam/
- Ultralytics SAM 2 documentation: https://docs.ultralytics.com/models/sam-2/
- Ultralytics SAM predictor API: https://docs.ultralytics.com/reference/models/sam/predict/
