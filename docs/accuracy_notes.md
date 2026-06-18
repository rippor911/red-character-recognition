# Accuracy Notes

本项目的指标是 `final_exact_acc`：先识别 5 个字符和 5 个颜色位置，再只拼接预测为红色的位置。因此它比单字符准确率更严格，也和通用 OCR 文档识别指标不可直接互换。

## 网上常见准确率区间

调研结论：固定长度、字符集有限的验证码/字符识别任务，在训练集和测试集分布接近时，公开论文和工程报告里常见目标是 `95%+`；较强的深度学习模型经常报告接近 `99%` 或更高的整串准确率。不过这些数值强依赖验证码样式、噪声、是否粘连、是否合成数据同分布、评价口径和是否集成。

| 来源 | 任务/模型 | 公开结果 | 对本项目的启发 |
| --- | --- | --- | --- |
| [CAPTCHA recognition based on deep convolutional neural network](https://www.aimspress.com/article/doi/10.3934/mbe.2019292?viewType=HTML) / [PubMed](https://pubmed.ncbi.nlm.nih.gov/31499741/) | 基于 CNN/DenseNet 思路的验证码识别 | 带背景噪声和字符粘连的验证码识别准确率报告为 `99.9%+` | 同分布验证码任务可以期待很高准确率，但该方法使用更强 DenseNet/ResNet 思路，本项目仍保持轻量自定义 CNN |
| [Robust CAPTCHA Recognition with a CNN-RNN-CTC](https://icai.uni-eszterhazy.hu/2026/abstracts/ICAI_2026_abstract_90.pdf) | CNN-RNN-CTC CAPTCHA 识别 | 单模型平均整串准确率 `98.95%`，21 模型集成 `99.70%` | 说明 99% 档通常需要序列建模或集成；本项目暂不使用 CTC/集成，只把它作为上限参照 |
| [Captcha Recognition Based on Deep Learning](https://dl.acm.org/doi/10.1145/3445945.3445961) | 简单和复杂验证码深度学习识别 | 摘要报告两类数据准确率接近 `99%` | 固定格式验证码的实用目标可设在 `95%-99%` |
| [A novel CAPTCHA solver framework using deep skipping connections](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044336/) | 带跳连结构的验证码求解器 | 字符识别层面多数接近或达到 `99%` | 单字符准确率高不等于整串准确率高，本项目应优先看 `final_exact_acc` |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / [PP-OCRv5 技术报告](https://arxiv.org/html/2507.05595v1) | 通用 OCR 文本识别 | 官方强调新版本相对旧版显著提升识别准确率，但通用 OCR 指标和验证码整串指标不同 | 通用 OCR 可作为工程参照，不适合作为本课程模型的直接准确率目标 |

## 本项目优化目标

在没有使用预训练、Transformer、CTC、集成和人工测试集修正的前提下，当前优化目标设为：

```text
first target:  final_exact_acc >= 0.95
stretch target: final_exact_acc 接近 0.98-0.99
```

如果官方验证集达不到 `95%`，优先继续排查：

1. 颜色预测是否成为瓶颈：查看 `color_pattern_acc`。
2. 字符分类是否成为瓶颈：查看 `char_slot_acc`。
3. 是否需要更长训练：当前推荐先跑 `20-30` epoch。
4. 是否需要按图像实际样式调整增强强度或输入尺寸。

当前仓库没有官方 `dataset/`，因此还不能证明已经达到上述目标。放入官方数据后，以 README 中的默认命令或推荐命令运行并记录验证集结果。

## 当前已实现的提分策略

当前实现仍保持“无预训练、无 Transformer、无 CTC、无集成”的约束，主要增强点是：

1. 更稳的轻量 CNN：使用 depthwise separable block、slot pooling、slot 位置嵌入和 MLP 分类头；默认 `feature_dim/head_hidden_dim=384`，并为 5 个位置使用位置专用分类头，整体仍保持约 146 万参数的轻量规模。
2. 颜色专用统计分支：每个 slot 额外提取 RGB 均值以及 `red - max(green, blue)` 的均值/最大值，降低颜色判定对深层字符特征的依赖。
3. 训练策略：轻量几何/亮度增强、AdamW、label smoothing、字符/颜色类别权重、warmup+cosine scheduler、AMP、梯度裁剪、EMA 权重滑动平均和确定性 TTA。
4. 验证集阈值校准：保留原始 `final_exact_acc`，同时在验证集上先扫描全局红色概率阈值，再贪心校准 5 个位置各自的红色阈值，记录 `threshold_final_exact_acc`、`color_thresholds` 和 `threshold_gain`。最终测试推理使用验证集选出的阈值列表，不读取或人工修改测试标签。

这类阈值校准常用于把分类概率转成任务目标所需的离散决策。它不会改变模型接口：

```python
char_logits, color_logits = model(images)
```

## 使用诊断文件继续提分

训练完成后查看：

```text
outputs/training_history.csv
outputs/val_predictions.csv
outputs/val_errors.csv
```

建议按下面顺序定位问题：

1. 如果 `color_pattern_acc` 或 `target_length_acc` 低，先看 `val_errors.csv` 里的 `red_prob_1...red_prob_5` 和 `pred_color_threshold`，通常需要调整每个位置的阈值范围、颜色分支、颜色增强或颜色类别权重。颜色类别权重只按训练子集统计，可用 `--no-color-class-weight` 做对照。
2. 如果 `char_slot_acc` 低，先看 `char_slot_1_acc...char_slot_5_acc` 是否集中坏在某个位置，再考虑加大输入宽度、训练轮数、CNN 宽度或字符类别权重；可通过 `--feature-dim`、`--head-hidden-dim`、`--shared-heads` 和 `--no-char-class-weight` 做容量/分类头/权重对照。
3. 如果 `char_conf_*` 普遍低但颜色正确，说明主要是字符分类欠拟合，优先增加 epoch 或 `head_hidden_dim`。
4. 如果训练集小样本能记住但验证集不上升，优先减弱增强、调高 dropout/weight decay 或检查 train/val 分布。
5. 如果 `training_history.csv` 中 `val_ema_threshold_final_exact_acc` 高于 `val_raw_threshold_final_exact_acc`，说明 EMA 更稳；如果 raw 长期更高，可以临时加 `--no-ema` 做对照实验。
6. 如果验证/测试图像存在轻微水平偏移，默认 TTA 会平均 `0,-2,2` 三个水平平移视图；如果发现 TTA 变慢或无收益，可加 `--no-tta` 或调整 `--tta-shifts` 做对照。
