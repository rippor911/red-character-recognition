# Red Character Recognition

机器学习课程大作业“红色字符识别”的基础模型实现。

## 数据目录

默认读取项目根目录下的数据：

```text
dataset/
├── train/
│   ├── images/
│   └── labels.csv
├── test/
│   └── images/
└── submission_sample.csv
```

`labels.csv` 需要包含：

```text
filename,color,all_label
```

其中 `all_label` 是 5 位完整字符，字符集固定为：

```python
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

`color` 使用 `u/r` 编码，训练时映射为：

```text
u -> 0
r -> 1
```

## 运行命令

完整训练、验证、测试推理并生成提交文件：

```bash
python src/main.py
```

当前最佳实验配置：

```bash
python src/main.py --model convnext_tiny --slot-extractor pool_query --normalization imagenet --image-height 192 --image-width 576 --learning-rate 2e-6 --epochs 1 --batch-size 8 --init-checkpoint checkpoints/convnext_192x576_pool_query_e2/baseline_best.pt --char-loss-weight 1.5 --color-loss-weight 0.5 --no-scheduler
```

Current best validation post-processing:

```bash
python src/main.py --predict-only --checkpoint-path checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt --model convnext_tiny --slot-extractor pool_query --normalization imagenet --image-height 192 --image-width 576 --batch-size 16 --use-confusion-rules
```

当前最佳验证结果记录在 [docs/experiment_log.md](docs/experiment_log.md)：

```text
BaselineCNN E5 calibrated_final_exact_acc=0.9188
ConvNeXt-Tiny 192x576 pool_query polish E1 calibrated_final_exact_acc=0.9874
ConvNeXt-Tiny + validation-tuned confusion rules calibrated_final_exact_acc=0.9898
relative_error_reduction=87.4%
```

使用已经保存的最佳 checkpoint 直接生成提交文件：

```bash
python src/main.py --predict-only --checkpoint-path checkpoints/baseline_best.pt
```

如果省略 `--checkpoint-path`，`--predict-only` 会默认读取 `--checkpoint-dir` 下的 `baseline_best.pt`。

从已有 checkpoint 初始化继续微调：

```bash
python src/main.py --model convnext_tiny --normalization imagenet --init-checkpoint checkpoints/convnext_pool_128x384_e4/baseline_best.pt --learning-rate 5e-5 --epochs 2
```

重新加载 checkpoint 并按固定验证划分输出指标：

```bash
python src/main.py --eval-checkpoint --checkpoint-path checkpoints/baseline_best.pt --skip-test
```

`--eval-checkpoint` 会复用 checkpoint 中保存的输入尺寸、归一化、TTA 与解码先验；不加 `--skip-test` 时会在评估后继续生成 `outputs/submission.csv`。

默认参数：

```text
batch_size=64
epochs=20
learning_rate=1e-3
optimizer=AdamW
image_size=64x256
seed=2026
device=cuda if available else cpu
weight_decay=1e-4
label_smoothing=0.03
char_class_weight=on, per-slot
max_char_class_weight=3.0
color_class_weight=on, per-slot
max_color_class_weight=3.0
feature_dim=384
head_hidden_dim=384
position_specific_heads=on
slot_pooling=avgmax
slot_context=on
normalization=dataset
normalization_samples=2048
train_augmentation=on
cosine_scheduler=on
warmup_epochs=2
amp=cuda only
ema=on
ema_decay=0.999
tta=on
tta_shifts=0,-2,2
tta_scales=1,0.95,1.05
balanced_sampler=on, by color pattern
char_prior_weights=0,0.1,0.25,0.5,1
count_prior_weights=0,0.25,0.5,1,1.5,2
pattern_confidence_weights=0,0.25,0.5,1
```

输出文件：

```text
checkpoints/baseline_best.pt
outputs/submission.csv
outputs/training_history.csv
outputs/val_predictions.csv
outputs/val_errors.csv
```

小样本过拟合检查：

```bash
python src/main.py --debug-overfit --debug-samples 128 --epochs 30 --batch-size 32 --skip-test
```

`--debug-overfit` 会固定取按文件名排序后的前 N 张训练图像，同时把该子集作为评估集，用于检查标签编码、模型输出、损失和反向传播是否正常。

常见验证码/字符识别器准确率调研和本项目优化目标见 [docs/accuracy_notes.md](docs/accuracy_notes.md)。

使用公开 ImageNet 预训练 ConvNeXt-Tiny backbone：

```bash
python src/main.py --model convnext_tiny --normalization imagenet --image-height 96 --image-width 320 --learning-rate 3e-4 --batch-size 32
```

Stronger public ImageNet backbone option:

```bash
python src/main.py --model convnext_small --slot-extractor pool_query --normalization imagenet --image-height 160 --image-width 480 --learning-rate 1e-4 --batch-size 4
```

`--model baseline_cnn` 是默认小 CNN；`--model convnext_tiny` 会使用 TorchVision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` 初始化视觉 backbone。若只想验证结构、不加载预训练权重，可加 `--no-pretrained-backbone`。预训练模型来源和实验过程记录在 [docs/experiment_log.md](docs/experiment_log.md)。

SVTRv2 启发的固定槽位特征重排实验：

```bash
python src/main.py --model convnext_tiny --slot-extractor pool_query --normalization imagenet --image-height 96 --image-width 320 --learning-rate 3e-4 --batch-size 32
```

`pool_query` 会在原有池化 slot 之外加入 5 个 learned slot query，从 ConvNeXt 特征图中抽取固定五字符位置的重排特征；这是对 SVTRv2 Feature Rearranging 思想的任务定制轻量化版本。

## 模型结构

模型名：`BaselineCNN`

结构：

```text
输入图片 [B, 3, 64, 256]
-> 小型 CNN 特征提取
-> AdaptiveAvgPool2d((1, 5)) + AdaptiveMaxPool2d((1, 5))
-> avg/max slot feature 拼接后投影回 384 维
-> 5 个 slot 上的局部 1D 卷积上下文混合
-> 5 个位置的 384 维 slot feature
-> 位置专用字符分类头 Linear(..., 36)
-> 颜色统计分支在反标准化后的 RGB 空间提取每个 slot 的 RGB 均值、red-minus-other 均值/最大值、正向红色响应和红色覆盖率
-> 位置专用颜色分类头 Linear(..., 2)
```

统一接口：

```python
char_logits, color_logits = model(images)
```

输出形状：

```text
char_logits:  [B, 5, 36]
color_logits: [B, 5, 2]
```

## 损失函数

```python
char_loss = CrossEntropyLoss(char_logits.reshape(-1, 36), char_target.reshape(-1))
color_loss = CrossEntropyLoss(color_logits.reshape(-1, 2), color_target.reshape(-1))
loss = char_loss + color_loss
```

两个任务权重均为 1。

正式训练默认从训练子集抽样计算全局输入均值/标准差并用于 train/val/test，避免验证或测试信息泄漏；可用 `--normalization fixed` 切回旧的 `mean=0.5,std=0.5`，或用 `--normalization none` 直接使用 `0-1` 像素。CNN 主干使用标准化输入，颜色统计分支会在模型内部按训练时的 mean/std 还原到 `0-1` RGB 再计算红色特征。对字符分类使用 `label_smoothing=0.03`，并按训练子集里 5 个位置各自的字符频率自动计算类别权重；对颜色分类也按 5 个位置各自的 `u/r` 比例自动计算类别权重。训练 loader 默认按 `color` 模式做均衡采样，可加 `--no-balanced-sampler` 关闭。验证和测试默认使用确定性位移+尺度 TTA，平均 `0,-2,2` 三个位移和 `1,0.95,1.05` 三个尺度的 logits；TTA 的白色填充值会随输入标准化同步调整。若需要更快可设 `--tta-scales 1` 或 `--no-tta`。slot pooling 默认使用 `avgmax`，同时保留均值和最大响应；可用 `--slot-pooling avg` 切回最初的纯 `AdaptiveAvgPool2d((1, 5))`。默认还会在 5 个 slot 特征上做一层局部 1D 卷积上下文混合，可加 `--no-slot-context` 关闭做对照。分类头默认按 5 个位置分别建模，可加 `--shared-heads` 切回共享 head 做对照。每个 epoch 会同时评估 raw/EMA，保存 `calibrated_final_exact_acc` 更高的版本。评估指标仍使用普通交叉熵和准确率，便于横向比较。`--debug-overfit` 会自动关闭 label smoothing、字符/颜色类别权重、dropout、scheduler、EMA、TTA、颜色模式先验、均衡采样和数据增强，便于检查小样本记忆能力。

## 验证指标

每个 epoch 输出：

```text
final_exact_acc
threshold_final_exact_acc
threshold_color_acc
count_final_exact_acc
count_color_acc
pattern_final_exact_acc
calibrated_final_exact_acc
calibrated_color_pattern_acc
calibrated_char_slot_acc
calibrated_length_acc
char_sequence_acc
char_oracle_final_exact_acc
color_oracle_final_exact_acc
color_decode_method
char_decode_method
color_thresholds
char_prior_weight
count_prior_weight
pattern_prior_weight
pattern_confidence_weight
char_prior_final_exact_acc
char_prior_slot_acc
char_slot_acc
color_slot_acc
color_pattern_acc
calibrated_gain
target_length_acc
char_slot_1_acc ... char_slot_5_acc
color_slot_1_acc ... color_slot_5_acc
calibrated_final_exact_acc_len_1 ... calibrated_final_exact_acc_len_5
calibrated_final_exact_acc_pattern_<color>
```

最终答案解码逻辑：

1. 字符默认取 `char_logits.argmax(dim=-1)`；
2. 颜色取 `color_logits.argmax(dim=-1)`；
3. 只保留预测为红色的位置；
4. 从左到右拼接对应字符；
5. 如果没有任何红色预测，则选择红色 logit 最高的位置，避免空答案。

`final_exact_acc` 使用上述 argmax 颜色解码；`threshold_final_exact_acc` 会在验证集上先扫描全局红色概率阈值，再贪心校准 5 个位置各自的红色阈值，并保存最佳 `color_thresholds` 到 checkpoint。训练还会从训练子集统计红色字符数量分布，在验证集上扫描 `count_prior_weight`，得到 `count_final_exact_acc`；它只约束红色数量，不约束具体位置。随后从训练子集统计非空 `r/u` 颜色模式，在验证集上扫描 `pattern_prior_weight`，并额外扫描 `pattern_confidence_weight`，用字符分类头的 slot 置信度辅助选择更可信的红色模式，得到 `pattern_final_exact_acc`。颜色校准后，还会只用训练子集统计 5 个位置各自的字符先验，在验证集上扫描 `char_prior_weight`；只有它提升 `calibrated_final_exact_acc` 时才保存为 `char_prior` 解码。最终 checkpoint 和测试推理使用 `calibrated_final_exact_acc` 对应的更优解码方式；如果数量先验、颜色模式先验、置信度辅助或字符先验没有带来收益，则退回阈值颜色解码和 argmax 字符解码。

训练结束后默认保存验证集诊断文件：

```text
outputs/val_predictions.csv
outputs/val_errors.csv
```

其中包含每张验证图的目标标签、预测标签、argmax/阈值/红色数量先验/模式先验或置信度辅助模式/最终校准颜色模式、字符先验标签、颜色 oracle 标签、字符 oracle 标签、红色概率、红色数量、字符置信度以及最终采用的解码参数。若只想训练不导出诊断，可加 `--no-val-diagnostics`；错误样本数量可用 `--max-error-samples` 控制。

## 提交文件校验

`outputs/submission.csv` 会自动检查：

```text
列名为 id,label
默认共有 5000 行预测
id 不重复
label 只包含 0-9A-Z
label 长度为 1 到 5
```

如需在自建小数据上 smoke test，可以用 `--expected-test-rows` 改写行数检查。

## 已运行检查

形状检查：

```text
torch.Size([2, 5, 36])
torch.Size([2, 5, 2])
default_params=1,909,038
no_slot_context_params=1,759,662
avg_pooling_no_context_params=1,462,830
```

临时小数据 smoke test：

```bash
python src/main.py --data-dir <temp_data> --output-dir <temp_outputs> --checkpoint-dir <temp_checkpoints> --epochs 1 --batch-size 4 --expected-test-rows 2
```

结果摘要：

```text
Device: cuda
Train samples: 9 | val samples: 3
Input normalization: dataset mean=0.xxxx std=0.xxxx white_fill=0.xxxx
Train augmentation: on | AMP: on
Dropout: 0.100 | label_smoothing: 0.030 | scheduler: warmup+cosine warmup_epochs=2
Color class weights: per-slot shape=(5, 2) min=0.3636 max=1.6364 mean=1.0000 r_by_slot=1.091,0.727,1.455,0.909,1.636
Char class weights: per-slot shape=(5, 36) min=0.3770 max=1.1429 mean=1.0000
Slot pooling: avgmax | slot_context: on
Char position prior: on weights=0,0.1,0.25,0.5,1
Red count prior: on weights=0,0.25,0.5,1,1.5,2 counts=1:0.xxx, 2:0.xxx, 3:0.xxx, 4:0.xxx, 5:0.xxx
EMA: on decay=0.99900
TTA shifts: 0,-2,2
TTA scales: 1,0.95,1.05
Balanced sampler: on color_patterns=rruuu:3, ururu:3, ruruu:1, uurru:1, uuurr:1
Color pattern prior: on candidates=5 weights=0,0.25,0.5,1,1.5,2 confidence_weights=0,0.25,0.5,1 top=rruuu:0.286, ururu:0.286, ruruu:0.143, uurru:0.143, uuurr:0.143
Model parameters: 1,909,038
Epoch 01/1 lr=1.00e-03 selected=raw train_loss=4.3416 val_loss=4.0800 final_exact_acc=0.0000 threshold_final_exact_acc=0.0000 calibrated_final_exact_acc=0.0000 decode=threshold color_thresholds=0.500,0.500,0.500,0.500,0.500 count_final_exact_acc=0.0000 count_prior_weight=0.00 pattern_final_exact_acc=0.0000 pattern_prior_weight=0.00 pattern_confidence_weight=0.00 char_prior_weight=0.00 char_decode=argmax char_slot_acc=0.0000 char_sequence_acc=0.0000 color_slot_acc=1.0000 color_pattern_acc=1.0000 char_oracle_final_exact_acc=1.0000 color_oracle_final_exact_acc=0.0000 calibrated_color_pattern_acc=1.0000 calibrated_length_acc=1.0000 calibrated_gain=0.0000 raw_calibrated_final_exact_acc=0.0000 ema_calibrated_final_exact_acc=0.0000
Saved best raw checkpoint
Saved training_history.csv
Saved val_predictions.csv
Saved val_errors.csv
Using color thresholds 0.500,0.500,0.500,0.500,0.500 for test decoding
Using argmax character decoding for test decoding
Using input normalization dataset (mean=0.xxxx, std=0.xxxx) for test decoding
Using TTA shifts 0,-2,2 for test decoding
Using TTA scales 1,0.95,1.05 for test decoding
Saved submission.csv
```

当前工作目录没有官方 `dataset/`，所以尚未记录 50,000 张训练集上的正式验证结果；放入官方数据后运行默认命令即可得到真实指标。

小样本过拟合临时检查：

```bash
python src/main.py --data-dir <temp_data> --output-dir <temp_outputs> --checkpoint-dir <temp_checkpoints> --debug-overfit --debug-samples 8 --epochs 30 --batch-size 4 --skip-test
```

结果摘要：

```text
Input normalization: dataset mean=0.xxxx std=0.xxxx white_fill=0.xxxx
Train augmentation: off | AMP: on
Dropout: 0.000 | label_smoothing: 0.000 | scheduler: off
Color class weights: off
Char class weights: off
Slot pooling: avgmax | slot_context: on
Char position prior: off
Red count prior: off
EMA: off
TTA shifts: 0
TTA scales: 1
Balanced sampler: off
Color pattern prior: off
Epoch 01/30 lr=1.00e-03 selected=raw train_loss=4.1810 debug_train_loss=3.7732 final_exact_acc=0.0000 threshold_final_exact_acc=0.0000 calibrated_final_exact_acc=0.0000 decode=threshold color_thresholds=0.500,0.500,0.500,0.500,0.500 count_final_exact_acc=0.0000 count_prior_weight=0.00 pattern_final_exact_acc=0.0000 pattern_prior_weight=0.00 pattern_confidence_weight=0.00 char_prior_weight=0.00 char_decode=argmax char_slot_acc=0.1250 char_sequence_acc=0.0000 color_slot_acc=0.9250 color_pattern_acc=0.6250 char_oracle_final_exact_acc=0.6250 color_oracle_final_exact_acc=0.0000 calibrated_gain=0.0000
...
Epoch 30/30 lr=1.00e-03 selected=raw train_loss=0.0002 debug_train_loss=0.1044 final_exact_acc=1.0000 threshold_final_exact_acc=1.0000 calibrated_final_exact_acc=1.0000 decode=threshold color_thresholds=0.500,0.500,0.500,0.500,0.500 count_final_exact_acc=1.0000 count_prior_weight=0.00 pattern_final_exact_acc=1.0000 pattern_prior_weight=0.00 pattern_confidence_weight=0.00 char_prior_weight=0.00 char_decode=argmax char_slot_acc=0.9750 char_sequence_acc=0.8750 color_slot_acc=1.0000 color_pattern_acc=1.0000 char_oracle_final_exact_acc=1.0000 color_oracle_final_exact_acc=1.0000 calibrated_gain=0.0000
Saved debug_train_predictions.csv
Saved debug_train_errors.csv
```
