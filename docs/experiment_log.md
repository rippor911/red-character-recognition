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

## 2026-06-19 ConvNeXt-Tiny Pool E6

Command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_tiny_pool_e6 \
  --checkpoint-dir checkpoints/convnext_tiny_pool_e6 \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 96 \
  --image-width 320 \
  --learning-rate 2e-4 \
  --epochs 6 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

Best validation result:

```text
best_epoch=5
selected_model=ema
val_loss=0.0687181046
final_exact_acc=0.9772
calibrated_final_exact_acc=0.9780
char_slot_acc=0.98924
char_sequence_acc=0.9472
color_slot_acc=0.99920
color_pattern_acc=0.9962
calibrated_color_pattern_acc=0.9968
calibrated_length_acc=0.9968
char_oracle_final_exact_acc=0.9968
color_oracle_final_exact_acc=0.9810
char_decode_method=char_prior
char_prior_weight=1.00
color_decode_method=threshold
color_thresholds=0.850,0.500,0.600,0.500,0.750
```

Learning curve:

```text
epoch 1: calibrated_final_exact_acc=0.9330
epoch 2: calibrated_final_exact_acc=0.9572
epoch 3: calibrated_final_exact_acc=0.9682
epoch 4: calibrated_final_exact_acc=0.9754
epoch 5: calibrated_final_exact_acc=0.9780
epoch 6: calibrated_final_exact_acc=0.9772
```

Comparison:

```text
BaselineCNN_E5=0.9188
ConvNeXt_FRM-lite_E2=0.9580
ConvNeXt_pool_E6=0.9780
gain_vs_baseline=+0.0592
gain_vs_previous_best=+0.0200
baseline_error=0.0812
pool_e6_error=0.0220
relative_error_reduction_vs_baseline=(0.0812 - 0.0220) / 0.0812 = 72.9%
```

Takeaways:

- Longer fine-tuning is much more important than the early FRM-lite gain.
- The best epoch is 5; epoch 6 slightly regressed, so the saved checkpoint uses epoch 5.
- Remaining errors are now small; color accuracy is almost saturated and most remaining room is character recognition on long red patterns.

Artifacts:

```text
checkpoints/convnext_tiny_pool_e6/baseline_best.pt
outputs/convnext_tiny_pool_e6/training_history.csv
outputs/convnext_tiny_pool_e6/val_predictions.csv
outputs/convnext_tiny_pool_e6/val_errors.csv
outputs/convnext_tiny_pool_e6/submission.csv
```

## Updated Current Best

```text
best_model=ConvNeXt-Tiny pretrained + avgmax pooled slots
checkpoint=checkpoints/convnext_tiny_pool_e6/baseline_best.pt
submission=outputs/convnext_tiny_pool_e6/submission.csv
calibrated_final_exact_acc=0.9780
baseline_calibrated_final_exact_acc=0.9188
absolute_gain=+0.0592
relative_error_reduction=72.9%
```

## 2026-06-19 ConvNeXt-Tiny High-Resolution Pool E4

Command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_pool_128x384_e4 \
  --checkpoint-dir checkpoints/convnext_pool_128x384_e4 \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 128 \
  --image-width 384 \
  --learning-rate 1.5e-4 \
  --epochs 4 \
  --batch-size 24 \
  --num-workers 2 \
  --device cuda
```

Best validation result:

```text
best_epoch=4
selected_model=ema
val_loss=0.0484169374
final_exact_acc=0.9808
calibrated_final_exact_acc=0.9820
char_slot_acc=0.99100
char_sequence_acc=0.9558
color_slot_acc=0.99932
color_pattern_acc=0.9968
calibrated_color_pattern_acc=0.9980
calibrated_length_acc=0.9980
char_oracle_final_exact_acc=0.9980
color_oracle_final_exact_acc=0.9840
char_decode_method=argmax
char_prior_weight=0.00
color_decode_method=threshold
color_thresholds=0.500,0.800,0.800,0.500,0.950
```

Learning curve:

```text
epoch 1: calibrated_final_exact_acc=0.9492
epoch 2: calibrated_final_exact_acc=0.9704
epoch 3: calibrated_final_exact_acc=0.9796
epoch 4: calibrated_final_exact_acc=0.9820
```

Comparison:

```text
BaselineCNN_E5=0.9188
ConvNeXt_pool_E6_96x320=0.9780
ConvNeXt_pool_E4_128x384=0.9820
gain_vs_baseline=+0.0632
gain_vs_previous_best=+0.0040
baseline_error=0.0812
highres_error=0.0180
relative_error_reduction_vs_baseline=(0.0812 - 0.0180) / 0.0812 = 77.8%
```

Takeaways:

- Higher feature resolution gives a measurable gain over 96x320.
- Batch size 24 fits RTX 4060 Laptop GPU with roughly 5GB memory used.
- The remaining errors are now close to the oracle limits from character and color branches.

Artifacts:

```text
checkpoints/convnext_pool_128x384_e4/baseline_best.pt
outputs/convnext_pool_128x384_e4/training_history.csv
outputs/convnext_pool_128x384_e4/val_predictions.csv
outputs/convnext_pool_128x384_e4/val_errors.csv
outputs/convnext_pool_128x384_e4/submission.csv
```

## Updated Current Best 2

```text
best_model=ConvNeXt-Tiny pretrained + avgmax pooled slots + 128x384 input
checkpoint=checkpoints/convnext_pool_128x384_e4/baseline_best.pt
submission=outputs/convnext_pool_128x384_e4/submission.csv
calibrated_final_exact_acc=0.9820
baseline_calibrated_final_exact_acc=0.9188
absolute_gain=+0.0632
relative_error_reduction=77.8%
```

## 2026-06-19 Fine-Tuning From Checkpoint Support

Motivation:

- High-resolution E4 was still improving at epoch 4.
- Re-running the whole experiment from scratch is slow.
- Remaining validation errors are mostly character confusions, so low-LR fine-tuning from the best checkpoint is a reasonable next step.

Implementation:

- Added CLI option `--init-checkpoint`.
- It initializes training weights from a saved checkpoint before optimizer creation.
- It is separate from `--checkpoint-path`, which remains for `--predict-only` and `--eval-checkpoint`.

Planned command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_highres_ft_e2 \
  --checkpoint-dir checkpoints/convnext_highres_ft_e2 \
  --init-checkpoint checkpoints/convnext_pool_128x384_e4/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 128 \
  --image-width 384 \
  --learning-rate 5e-5 \
  --epochs 2 \
  --batch-size 24 \
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

## 2026-06-19 SVTRv2-Inspired FRM-Lite Implementation

Motivation:

- SVTRv2 uses feature rearranging to improve text recognition features.
- This dataset always contains exactly five character slots, so a full CTC-oriented FRM is unnecessary.
- I implemented a task-specific fixed-slot feature rearranger instead:

```text
ConvNeXt feature map [B,C,H,W]
-> flatten to visual tokens [B,HW,C]
-> 5 learned slot queries
-> query-key attention over visual tokens
-> 5 rearranged slot features [B,5,D]
-> fuse with pooled slot features when --slot-extractor pool_query
```

Implementation changes:

- Added `FixedSlotFeatureRearranger` in `src/model.py`.
- Added `slot_extractor` modes for ConvNeXt:
  - `pool`: original adaptive pool slots,
  - `query`: learned query slots only,
  - `pool_query`: fused pooled and query slots.
- Added CLI option `--slot-extractor pool|query|pool_query`.

Smoke command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/frm_lite_smoke \
  --checkpoint-dir checkpoints/frm_lite_smoke \
  --model convnext_tiny \
  --slot-extractor pool_query \
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
slot_extractor=pool_query
params=28,077,310
train_loss=4.3033
debug_train_loss=4.1849
calibrated_final_exact_acc=0.0625
char_slot_acc=0.0625
color_slot_acc=0.6250
```

Next ablation:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_frm_lite_e2 \
  --checkpoint-dir checkpoints/convnext_frm_lite_e2 \
  --model convnext_tiny \
  --slot-extractor pool_query \
  --normalization imagenet \
  --image-height 96 \
  --image-width 320 \
  --learning-rate 3e-4 \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

## 2026-06-19 ConvNeXt + FRM-Lite E2

Command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_frm_lite_e2 \
  --checkpoint-dir checkpoints/convnext_frm_lite_e2 \
  --model convnext_tiny \
  --slot-extractor pool_query \
  --normalization imagenet \
  --image-height 96 \
  --image-width 320 \
  --learning-rate 3e-4 \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

Result:

```text
epoch=2
selected_model=ema
params=30,644,830
train_loss=0.4284676435
val_loss=0.1210689162
final_exact_acc=0.9528
calibrated_final_exact_acc=0.9580
char_slot_acc=0.98208
char_sequence_acc=0.9144
color_slot_acc=0.99768
color_pattern_acc=0.9886
calibrated_color_pattern_acc=0.9938
calibrated_length_acc=0.9942
char_oracle_final_exact_acc=0.9942
color_oracle_final_exact_acc=0.9636
color_decode_method=threshold
char_decode_method=argmax
color_thresholds=0.950,0.750,0.750,0.950,0.950
```

Comparison:

```text
BaselineCNN_E5=0.9188
ConvNeXt_pool_E2=0.9578
ConvNeXt_pool_query_E2=0.9580
gain_vs_baseline=+0.0392
gain_vs_convnext_pool=+0.0002
baseline_error=0.0812
frm_lite_error=0.0420
relative_error_reduction_vs_baseline=(0.0812 - 0.0420) / 0.0812 = 48.3%
target_relative_error_reduction=15.0%
target_met=True
```

Takeaways:

- FRM-lite is a small positive ablation over ConvNeXt pool slots at epoch 2.
- The gain is tiny, so the final report should not overclaim it.
- It is still useful as an innovation point: SVTRv2 feature rearranging adapted to fixed five-slot CAPTCHA recognition.
- Cost: memory increased from about 4.6GB to about 7.4GB on RTX 4060 Laptop GPU.

Artifacts:

```text
checkpoints/convnext_frm_lite_e2/baseline_best.pt
outputs/convnext_frm_lite_e2/training_history.csv
outputs/convnext_frm_lite_e2/val_predictions.csv
outputs/convnext_frm_lite_e2/val_errors.csv
outputs/convnext_frm_lite_e2/submission.csv
```

## Current Best

```text
best_model=ConvNeXt-Tiny pretrained + FRM-lite pool_query
checkpoint=checkpoints/convnext_frm_lite_e2/baseline_best.pt
submission=outputs/convnext_frm_lite_e2/submission.csv
calibrated_final_exact_acc=0.9580
baseline_calibrated_final_exact_acc=0.9188
absolute_gain=+0.0392
relative_error_reduction=48.3%
```

Recommended reproduction command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_frm_lite_e2 \
  --checkpoint-dir checkpoints/convnext_frm_lite_e2 \
  --model convnext_tiny \
  --slot-extractor pool_query \
  --normalization imagenet \
  --image-height 96 \
  --image-width 320 \
  --learning-rate 3e-4 \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 2 \
  --device cuda
```

## 2026-06-19 ConvNeXt-Tiny High-Resolution Fine-Tune E2

Motivation:

- The 128x384 high-resolution run improved exact accuracy to 0.9820.
- Most remaining validation errors are character confusions rather than color pattern mistakes.
- I added `--init-checkpoint` so training can initialize from an existing best checkpoint without switching into eval/predict-only mode.

Implementation:

- Added CLI option `--init-checkpoint`.
- The training path loads model weights, prior thresholds, image size, normalization and decode prior metadata from the initialization checkpoint.
- Optimizer, scheduler, EMA and epoch counters still start fresh for the new fine-tuning run.

Command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_highres_ft_e2 \
  --checkpoint-dir checkpoints/convnext_highres_ft_e2 \
  --init-checkpoint checkpoints/convnext_pool_128x384_e4/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 128 \
  --image-width 384 \
  --learning-rate 5e-5 \
  --epochs 2 \
  --batch-size 24 \
  --num-workers 2 \
  --device cuda
```

Best result:

```text
best_epoch=1
selected_model=ema
train_loss=0.2642
val_loss=0.0485
final_exact_acc=0.9818
threshold_final_exact_acc=0.9832
calibrated_final_exact_acc=0.9832
char_slot_acc=0.99160
char_sequence_acc=0.9584
color_slot_acc=0.99940
color_pattern_acc=0.9972
char_oracle_final_exact_acc=0.9986
color_oracle_final_exact_acc=0.9846
color_thresholds=0.500,0.800,0.850,0.650,0.950
```

Comparison:

```text
BaselineCNN_E5=0.9188
ConvNeXt_pool_E6_96x320=0.9780
ConvNeXt_pool_E4_128x384=0.9820
ConvNeXt_highres_finetune_E1=0.9832
gain_vs_baseline=+0.0644
gain_vs_previous_best=+0.0012
baseline_error=0.0812
finetune_error=0.0168
relative_error_reduction_vs_baseline=(0.0812 - 0.0168) / 0.0812 = 79.3%
target_relative_error_reduction=15.0%
target_met=True
```

Takeaways:

- Low-learning-rate fine-tuning from the high-resolution checkpoint is a small but real positive step.
- The improvement mostly comes from character accuracy: `char_slot_acc` rose from 0.99100 to 0.99160.
- Color recognition is already near saturation, so the next experiments should focus more loss/augmentation capacity on confusing character pairs.

Artifacts:

```text
checkpoints/convnext_highres_ft_e2/baseline_best.pt
outputs/convnext_highres_ft_e2/training_history.csv
outputs/convnext_highres_ft_e2/val_predictions.csv
outputs/convnext_highres_ft_e2/val_errors.csv
outputs/convnext_highres_ft_e2/submission.csv
```

## Updated Current Best 3

```text
best_model=ConvNeXt-Tiny pretrained + avgmax pooled slots + 128x384 input + checkpoint fine-tuning
checkpoint=checkpoints/convnext_highres_ft_e2/baseline_best.pt
submission=outputs/convnext_highres_ft_e2/submission.csv
calibrated_final_exact_acc=0.9832
baseline_calibrated_final_exact_acc=0.9188
absolute_gain=+0.0644
relative_error_reduction=79.3%
```

Recommended reproduction command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_highres_ft_e2 \
  --checkpoint-dir checkpoints/convnext_highres_ft_e2 \
  --init-checkpoint checkpoints/convnext_pool_128x384_e4/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 128 \
  --image-width 384 \
  --learning-rate 5e-5 \
  --epochs 2 \
  --batch-size 24 \
  --num-workers 2 \
  --device cuda
```

## 2026-06-19 Character-Weighted High-Resolution Fine-Tune E2

Motivation:

- The previous high-resolution fine-tune reached 0.9832 with only 84 validation errors.
- Error analysis showed 78/84 errors still had full-character mistakes, while only 7/84 had color decoding mistakes.
- The most frequent confusions were `I->1`, `O->0`, `1->I`, `Q->O` and `E->F`, so I shifted more training weight to the character head.

Command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_highres_ft_charweight_e2 \
  --checkpoint-dir checkpoints/convnext_highres_ft_charweight_e2 \
  --init-checkpoint checkpoints/convnext_highres_ft_e2/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 128 \
  --image-width 384 \
  --learning-rate 2e-5 \
  --epochs 2 \
  --batch-size 24 \
  --num-workers 2 \
  --device cuda \
  --char-loss-weight 1.5 \
  --color-loss-weight 0.5
```

Best result:

```text
best_epoch=1
selected_model=raw
train_loss=0.3735
val_loss=0.0510
final_exact_acc=0.9824
threshold_final_exact_acc=0.9836
calibrated_final_exact_acc=0.9836
char_slot_acc=0.99136
char_sequence_acc=0.9576
color_slot_acc=0.99936
color_pattern_acc=0.9970
char_oracle_final_exact_acc=0.9982
color_oracle_final_exact_acc=0.9854
color_thresholds=0.700,0.500,0.950,0.800,0.950
```

Comparison:

```text
BaselineCNN_E5=0.9188
ConvNeXt_highres_finetune_E1=0.9832
ConvNeXt_highres_char_weight_E1=0.9836
gain_vs_baseline=+0.0648
gain_vs_previous_best=+0.0004
baseline_error=0.0812
char_weight_error=0.0164
relative_error_reduction_vs_baseline=(0.0812 - 0.0164) / 0.0812 = 79.8%
target_relative_error_reduction=15.0%
target_met=True
```

Validation error analysis:

```text
errors=82
char_all_wrong=74
color_wrong=9
length_wrong=9
top_confusions=I->1:5, O->0:4, 1->I:4, Q->O:4, P->R:3, U->J:3, X->Y:3
```

Submission check:

```text
path=outputs/convnext_highres_ft_charweight_e2/submission.csv
columns=id,label
rows=5000
unique_ids=5000
labels_match_[0-9A-Z]{1,5}=True
```

Takeaways:

- Character-weighted fine-tuning is a small positive ablation over the previous best.
- The best epoch selected the raw model rather than EMA, so further low-learning-rate runs should not assume EMA always wins.
- Remaining mistakes are still dominated by visually similar characters, making confusion-aware augmentation or higher-resolution crops the next likely improvement direction.

Artifacts:

```text
checkpoints/convnext_highres_ft_charweight_e2/baseline_best.pt
outputs/convnext_highres_ft_charweight_e2/training_history.csv
outputs/convnext_highres_ft_charweight_e2/val_predictions.csv
outputs/convnext_highres_ft_charweight_e2/val_errors.csv
outputs/convnext_highres_ft_charweight_e2/submission.csv
logs/convnext_highres_ft_charweight_e2.out.log
logs/convnext_highres_ft_charweight_e2.err.log
```

## Updated Current Best 4

```text
best_model=ConvNeXt-Tiny pretrained + avgmax pooled slots + 128x384 input + checkpoint fine-tuning + character-weighted loss
checkpoint=checkpoints/convnext_highres_ft_charweight_e2/baseline_best.pt
submission=outputs/convnext_highres_ft_charweight_e2/submission.csv
calibrated_final_exact_acc=0.9836
baseline_calibrated_final_exact_acc=0.9188
absolute_gain=+0.0648
relative_error_reduction=79.8%
```

Recommended reproduction command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_highres_ft_charweight_e2 \
  --checkpoint-dir checkpoints/convnext_highres_ft_charweight_e2 \
  --init-checkpoint checkpoints/convnext_highres_ft_e2/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 128 \
  --image-width 384 \
  --learning-rate 2e-5 \
  --epochs 2 \
  --batch-size 24 \
  --num-workers 2 \
  --device cuda \
  --char-loss-weight 1.5 \
  --color-loss-weight 0.5
```

## 2026-06-19 160x480 Character-Weighted Fine-Tune E2

Motivation:

- The 128x384 character-weighted model still made 82 validation errors.
- Remaining mistakes are dominated by visually similar characters such as `I/1`, `O/0`, `E/F`, and `Q/O`.
- I increased input resolution to 160x480 while keeping the same ConvNeXt-Tiny backbone and character-weighted loss, to preserve more stroke-level detail.

Command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_160x480_charweight_e2 \
  --checkpoint-dir checkpoints/convnext_160x480_charweight_e2 \
  --init-checkpoint checkpoints/convnext_highres_ft_charweight_e2/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 160 \
  --image-width 480 \
  --learning-rate 1e-5 \
  --epochs 2 \
  --batch-size 16 \
  --num-workers 2 \
  --device cuda \
  --char-loss-weight 1.5 \
  --color-loss-weight 0.5
```

Best result:

```text
best_epoch=2
selected_model=ema
train_loss=0.3868
val_loss=0.0448
final_exact_acc=0.9854
threshold_final_exact_acc=0.9860
calibrated_final_exact_acc=0.9860
char_slot_acc=0.99296
char_sequence_acc=0.9652
color_slot_acc=0.99960
color_pattern_acc=0.9980
char_oracle_final_exact_acc=0.9986
color_oracle_final_exact_acc=0.9874
color_thresholds=0.950,0.500,0.750,0.500,0.650
```

Comparison:

```text
BaselineCNN_E5=0.9188
ConvNeXt_highres_char_weight_E1=0.9836
ConvNeXt_160x480_char_weight_E2=0.9860
gain_vs_baseline=+0.0672
gain_vs_previous_best=+0.0024
baseline_error=0.0812
current_error=0.0140
relative_error_reduction_vs_baseline=(0.0812 - 0.0140) / 0.0812 = 82.8%
target_relative_error_reduction=15.0%
target_met=True
```

Validation error analysis:

```text
errors=70
char_all_wrong=63
color_wrong=7
length_wrong=7
top_confusions=I->1:5, O->0:5, E->F:4, L->I:3, Q->O:3, 0->O:2, J->I:2
```

Submission check:

```text
path=outputs/convnext_160x480_charweight_e2/submission.csv
columns=id,label
rows=5000
unique_ids=5000
labels_match_[0-9A-Z]{1,5}=True
```

Takeaways:

- Higher input resolution is the strongest improvement after the pretrained ConvNeXt backbone and is a clear report point.
- The exact-accuracy gain mainly comes from character recognition: `char_slot_acc` improved from 0.99136 to 0.99296 versus the 128x384 character-weighted run.
- The run fits on the local RTX 4060 with batch size 16 and uses about 5.2GB GPU memory during training.
- The remaining 70 validation errors are still mostly character confusions, so future work should target ambiguous glyph pairs rather than color classification.

Artifacts:

```text
checkpoints/convnext_160x480_charweight_e2/baseline_best.pt
outputs/convnext_160x480_charweight_e2/training_history.csv
outputs/convnext_160x480_charweight_e2/val_predictions.csv
outputs/convnext_160x480_charweight_e2/val_errors.csv
outputs/convnext_160x480_charweight_e2/submission.csv
logs/convnext_160x480_charweight_e2.out.log
logs/convnext_160x480_charweight_e2.err.log
```

## Updated Current Best 5

```text
best_model=ConvNeXt-Tiny pretrained + avgmax pooled slots + 160x480 input + checkpoint fine-tuning + character-weighted loss
checkpoint=checkpoints/convnext_160x480_charweight_e2/baseline_best.pt
submission=outputs/convnext_160x480_charweight_e2/submission.csv
calibrated_final_exact_acc=0.9860
baseline_calibrated_final_exact_acc=0.9188
absolute_gain=+0.0672
relative_error_reduction=82.8%
```

Recommended reproduction command:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_160x480_charweight_e2 \
  --checkpoint-dir checkpoints/convnext_160x480_charweight_e2 \
  --init-checkpoint checkpoints/convnext_highres_ft_charweight_e2/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool \
  --normalization imagenet \
  --image-height 160 \
  --image-width 480 \
  --learning-rate 1e-5 \
  --epochs 2 \
  --batch-size 16 \
  --num-workers 2 \
  --device cuda \
  --char-loss-weight 1.5 \
  --color-loss-weight 0.5
```

