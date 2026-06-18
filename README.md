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
epochs=5
learning_rate=1e-3
optimizer=AdamW
image_size=64x256
seed=2026
device=cuda if available else cpu
```

输出文件：

```text
checkpoints/baseline_best.pt
outputs/submission.csv
```

小样本过拟合检查：

```bash
python src/main.py --debug-overfit --debug-samples 128 --epochs 30 --batch-size 32 --skip-test
```

`--debug-overfit` 会固定取按文件名排序后的前 N 张训练图像，同时把该子集作为评估集，用于检查标签编码、模型输出、损失和反向传播是否正常。

## 模型结构

模型名：`BaselineCNN`

结构：

```text
输入图片 [B, 3, 64, 256]
-> 小型 CNN 特征提取
-> AdaptiveAvgPool2d((1, 5))
-> 5 个位置的 slot feature
-> 字符分类头 Linear(..., 36)
-> 颜色分类头 Linear(..., 2)
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

## 验证指标

每个 epoch 输出：

```text
final_exact_acc
char_slot_acc
color_slot_acc
color_pattern_acc
```

最终答案解码逻辑：

1. 字符取 `char_logits.argmax(dim=-1)`；
2. 颜色取 `color_logits.argmax(dim=-1)`；
3. 只保留预测为红色的位置；
4. 从左到右拼接对应字符；
5. 如果没有任何红色预测，则选择红色 logit 最高的位置，避免空答案。

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
Model parameters: 988,998
Epoch 01/1 train_loss=4.2600 val_loss=4.3009 final_exact_acc=0.0000 char_slot_acc=0.0000 color_slot_acc=0.4000 color_pattern_acc=0.0000
Saved best checkpoint
Saved submission.csv
```

当前工作目录没有官方 `dataset/`，所以尚未记录 50,000 张训练集上的正式验证结果；放入官方数据后运行默认命令即可得到真实指标。
