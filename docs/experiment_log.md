# Experiment Log

## 2026-06-19 Setup

Objective: build a reproducible baseline, then improve it with public pretrained models and selected SVTRv2-inspired modules while keeping the task academically clean.

Data:

```text
data_dir=C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别
train_images=50000
train_labels=50000
test_images=5000
submission_rows=5000
sampled_image_size=200x60
```

Hardware and libraries:

```text
torch=2.7.1+cu128
torchvision=0.22.1+cu128
cuda_available=True
gpu=NVIDIA GeForce RTX 4060 Laptop GPU
gpu_memory=8.0GB
```

Academic-use notes:

- No test image is manually labeled.
- No leaked or task-specific external CAPTCHA data is used.
- Public pretrained models may be used and must be cited in the report.
- Current small CNN baseline uses no pretrained weights.

Planned stages:

1. Run the current CNN baseline on the official 90/10 split.
2. Download and cache a public ImageNet-pretrained ConvNeXt-Tiny model from TorchVision.
3. Replace only the visual backbone first, keeping the project-specific five-slot char/color heads.
4. Add selected SVTRv2-inspired modules only when they match this fixed five-character task:
   - higher-resolution input,
   - local Conv2-style slot mixing,
   - lightweight fixed-slot feature rearrangement.
5. Record every command, metric, implementation change, and commit.

## 2026-06-19 Baseline CNN E5

Command:

```bash
python src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/baseline_cnn_e5 \
  --checkpoint-dir checkpoints/baseline_cnn_e5 \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda
```

Model:

```text
BaselineCNN
params=1,909,038
pretrained_weights=None
image_size=64x256
slot_pooling=avgmax
slot_context=on
normalization=dataset, per-channel
TTA shifts=0,-2,2
TTA scales=1,0.95,1.05
```

Best validation result:

```text
epoch=5
train_loss=0.5209938836
val_loss=0.1897287615
final_exact_acc=0.9156
calibrated_final_exact_acc=0.9188
char_slot_acc=0.95028
char_sequence_acc=0.7740
color_slot_acc=0.99848
color_pattern_acc=0.9926
calibrated_color_pattern_acc=0.9942
calibrated_length_acc=0.9948
char_oracle_final_exact_acc=0.9948
color_oracle_final_exact_acc=0.9218
color_decode_method=threshold
char_decode_method=argmax
color_thresholds=0.850,0.750,0.850,0.650,0.900
```

Artifacts:

```text
checkpoints/baseline_cnn_e5/baseline_best.pt
outputs/baseline_cnn_e5/training_history.csv
outputs/baseline_cnn_e5/val_predictions.csv
outputs/baseline_cnn_e5/val_errors.csv
outputs/baseline_cnn_e5/submission.csv
```

Notes:

- The baseline is already strong; the main remaining bottleneck is character recognition, not red-color detection.
- `color_slot_acc=0.99848` and `color_pattern_acc=0.9926` show that the color branch is near saturation.
- `char_slot_acc=0.95028` and `char_sequence_acc=0.7740` leave more room for a pretrained visual backbone.
- Because baseline exact accuracy is 91.88%, a literal +15 absolute-point gain is impossible without exceeding 100%. I will use "at least 15% relative error reduction" as the practical target:

```text
baseline_error = 1 - 0.9188 = 0.0812
15_percent_error_reduction_target = 1 - 0.0812 * 0.85 = 0.93098
target_calibrated_final_exact_acc >= 0.9310
```

## 2026-06-19 Pretrained Weight Download

Command:

```bash
python - <<PY
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)
PY
```

Downloaded public pretrained model:

```text
model=TorchVision ConvNeXt-Tiny
weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
source_url=https://download.pytorch.org/models/convnext_tiny-983f1562.pth
cache_file=C:\Users\GJR79\.cache\torch\hub\checkpoints\convnext_tiny-983f1562.pth
cache_file_size=114,419,221 bytes
model_params=28,589,128
license/source=TorchVision official pretrained weights
```

Use plan:

- Use the pretrained ConvNeXt-Tiny only as a visual feature extractor/fine-tuned backbone.
- Keep the task-specific five-slot character head and five-slot red-color head.
- Do not use external task labels or test-set annotations.

## 2026-06-19 ConvNeXt-Tiny Integration

Implementation changes:

- Added `PretrainedConvNeXtSlotModel` in `src/model.py`.
- Added CLI option `--model baseline_cnn|convnext_tiny`.
- Added CLI option `--no-pretrained-backbone`.
- Added `--normalization imagenet` for pretrained ImageNet backbones.
- Kept the same five-slot output contract:

```text
char_logits=[B,5,36]
color_logits=[B,5,2]
```

Smoke command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_smoke \
  --checkpoint-dir checkpoints/convnext_smoke \
  --model convnext_tiny \
  --normalization imagenet \
  --feature-dim 64 \
  --head-hidden-dim 64 \
  --debug-overfit \
  --debug-samples 16 \
  --epochs 1 \
  --batch-size 4 \
  --device cuda \
  --skip-test \
  --no-tta
```

Smoke result:

```text
Model: convnext_tiny pretrained_backbone=on
params=27,969,918
train_loss=4.2474
debug_train_loss=4.1370
calibrated_final_exact_acc=0.0625
char_slot_acc=0.1000
color_slot_acc=0.6375
```

Next experiment:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_tiny_e5 \
  --checkpoint-dir checkpoints/convnext_tiny_e5 \
  --model convnext_tiny \
  --normalization imagenet \
  --image-height 96 \
  --image-width 320 \
  --learning-rate 3e-4 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

## 2026-06-19 ConvNeXt-Tiny Pretrained E2 Early Stop

Training command actually used:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_tiny_e5 \
  --checkpoint-dir checkpoints/convnext_tiny_e5 \
  --model convnext_tiny \
  --normalization imagenet \
  --image-height 96 \
  --image-width 320 \
  --learning-rate 3e-4 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

Note: this command missed `--epochs 5`, so it started with the default 20 epochs. It was manually early-stopped after epoch 2 because the validation target had already been exceeded.

Checkpoint evaluation command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_tiny_e2_eval \
  --checkpoint-path checkpoints/convnext_tiny_e5/baseline_best.pt \
  --eval-checkpoint \
  --expected-test-rows 5000 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

Best validation result:

```text
epoch=2
selected_model=ema
val_loss=0.1143
final_exact_acc=0.9528
calibrated_final_exact_acc=0.9578
char_slot_acc=0.9820
char_sequence_acc=0.9138
color_slot_acc=0.9976
color_pattern_acc=0.9882
calibrated_color_pattern_acc=0.9922
calibrated_length_acc=0.9928
char_oracle_final_exact_acc=0.9928
color_oracle_final_exact_acc=0.9642
color_decode_method=threshold
char_decode_method=argmax
color_thresholds=0.950,0.850,0.950,0.950,0.950
```

Improvement over CNN baseline:

```text
baseline_calibrated_final_exact_acc=0.9188
convnext_calibrated_final_exact_acc=0.9578
absolute_gain=+0.0390
baseline_error=0.0812
convnext_error=0.0422
relative_error_reduction=(0.0812 - 0.0422) / 0.0812 = 48.0%
target_relative_error_reduction=15.0%
target_met=True
```

Takeaways:

- The pretrained visual backbone is the strongest gain so far.
- The color branch is no longer the bottleneck; exact accuracy is mostly limited by remaining character mistakes.
- This is a clean report point: public ImageNet pretrained backbone + task-specific five-slot heads.

Artifacts:

```text
checkpoints/convnext_tiny_e5/baseline_best.pt
outputs/convnext_tiny_e2_eval/val_checkpoint_metrics.csv
outputs/convnext_tiny_e2_eval/val_checkpoint_predictions.csv
outputs/convnext_tiny_e2_eval/val_checkpoint_errors.csv
outputs/convnext_tiny_e2_eval/submission.csv
```
