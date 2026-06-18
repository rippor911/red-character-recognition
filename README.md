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
char_class_weight=on
max_char_class_weight=3.0
color_class_weight=on
max_color_class_weight=3.0
feature_dim=384
head_hidden_dim=384
position_specific_heads=on
train_augmentation=on
cosine_scheduler=on
amp=cuda only
ema=on
ema_decay=0.999
tta=on
tta_shifts=0,-2,2
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

## 模型结构

模型名：`BaselineCNN`

结构：

```text
输入图片 [B, 3, 64, 256]
-> 小型 CNN 特征提取
-> AdaptiveAvgPool2d((1, 5))
-> 5 个位置的 384 维 slot feature
-> 位置专用字符分类头 Linear(..., 36)
-> 颜色统计分支提取每个 slot 的 RGB 均值和 red-minus-other 均值/最大值
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

正式训练默认对字符分类使用 `label_smoothing=0.03`，并按训练子集里的字符频率自动计算字符类别权重；对颜色分类也按训练子集里的 `u/r` 位置比例自动计算类别权重。验证和测试默认使用确定性水平平移 TTA，平均 `0,-2,2` 三个视图的 logits。分类头默认按 5 个位置分别建模，可加 `--shared-heads` 切回共享 head 做对照。每个 epoch 会同时评估 raw/EMA，保存 `threshold_final_exact_acc` 更高的版本。评估指标仍使用普通交叉熵和准确率，便于横向比较。`--debug-overfit` 会自动关闭 label smoothing、字符/颜色类别权重、dropout、scheduler、EMA、TTA 和数据增强，便于检查小样本记忆能力。

## 验证指标

每个 epoch 输出：

```text
final_exact_acc
threshold_final_exact_acc
color_threshold
char_slot_acc
color_slot_acc
color_pattern_acc
threshold_gain
target_length_acc
char_slot_1_acc ... char_slot_5_acc
color_slot_1_acc ... color_slot_5_acc
```

最终答案解码逻辑：

1. 字符取 `char_logits.argmax(dim=-1)`；
2. 颜色取 `color_logits.argmax(dim=-1)`；
3. 只保留预测为红色的位置；
4. 从左到右拼接对应字符；
5. 如果没有任何红色预测，则选择红色 logit 最高的位置，避免空答案。

`final_exact_acc` 使用上述 argmax 颜色解码；`threshold_final_exact_acc` 会在验证集上扫描红色概率阈值，并保存最佳 `color_threshold` 到 checkpoint。测试集推理会使用该阈值解码，若没有校准信息则退回 `0.5`。

训练结束后默认保存验证集诊断文件：

```text
outputs/val_predictions.csv
outputs/val_errors.csv
```

其中包含每张验证图的目标标签、预测标签、argmax/阈值颜色模式、红色概率和字符置信度。若只想训练不导出诊断，可加 `--no-val-diagnostics`；错误样本数量可用 `--max-error-samples` 控制。

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
```

临时小数据 smoke test：

```bash
python src/main.py --data-dir <temp_data> --output-dir <temp_outputs> --checkpoint-dir <temp_checkpoints> --epochs 1 --batch-size 4 --expected-test-rows 2
```

结果摘要：

```text
Device: cuda
Train samples: 7 | val samples: 5
Train augmentation: on | AMP: on
Dropout: 0.100 | label_smoothing: 0.030 | scheduler: on
Color class weights: u=0.8000 r=1.2000
Char class weights: min=1.0000 max=1.0000 mean=1.0000
EMA: on decay=0.99900
TTA shifts: 0,-2,2
Model parameters: 1,462,062
Epoch 01/1 lr=1.00e-03 selected=raw train_loss=4.2598 val_loss=3.9604 final_exact_acc=0.0000 threshold_final_exact_acc=0.0000 color_threshold=0.500 char_slot_acc=0.0400 color_slot_acc=1.0000 color_pattern_acc=1.0000 threshold_gain=0.0000 raw_threshold_final_exact_acc=0.0000 ema_threshold_final_exact_acc=0.0000
Saved best raw checkpoint
Saved training_history.csv
Saved val_predictions.csv
Saved val_errors.csv
Using color threshold 0.500 for test decoding
Using TTA shifts 0,-2,2 for test decoding
Saved submission.csv
```

当前工作目录没有官方 `dataset/`，所以尚未记录 50,000 张训练集上的正式验证结果；放入官方数据后运行默认命令即可得到真实指标。

小样本过拟合临时检查：

```bash
python src/main.py --data-dir <temp_data> --output-dir <temp_outputs> --checkpoint-dir <temp_checkpoints> --debug-overfit --debug-samples 8 --epochs 30 --batch-size 4 --skip-test
```

结果摘要：

```text
Train augmentation: off | AMP: on
Dropout: 0.000 | label_smoothing: 0.000 | scheduler: off
Color class weights: off
Char class weights: off
EMA: off
TTA shifts: 0
Epoch 01/30 lr=1.00e-03 selected=raw train_loss=4.1910 debug_train_loss=3.6281 final_exact_acc=0.1250 threshold_final_exact_acc=0.1250 color_threshold=0.500 char_slot_acc=0.1500 color_slot_acc=0.9000 color_pattern_acc=0.5000 threshold_gain=0.0000
...
Epoch 30/30 lr=1.00e-03 selected=raw train_loss=0.0002 debug_train_loss=0.0019 final_exact_acc=1.0000 threshold_final_exact_acc=1.0000 color_threshold=0.500 char_slot_acc=1.0000 color_slot_acc=1.0000 color_pattern_acc=1.0000 threshold_gain=0.0000
Saved debug_train_predictions.csv
Saved debug_train_errors.csv
```
