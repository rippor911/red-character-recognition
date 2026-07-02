# 0.98500 Submission Reproduction Note

Generated on 2026-07-03 from local repository:

```text
C:\Users\GJR79\Desktop\桌面整理_20260630\01_课程作业\机器学习大作业
```

## Short Answer

The `submission.csv` that is currently in the canonical output location is:

```text
outputs/submission.csv
```

It is byte-identical to:

```text
outputs/convnext_best_confusion_rules_contextual_predict/submission.csv
```

SHA256:

```text
36f131cda8c5987bd960f89f2830cbe3bb9a5e1967b33e795e879e99b2e196e9
```

Validation caveat: this file was generated with validation-tuned contextual confusion rules. It is reproducible locally, but it should be reported as a high-risk post-processing result, not as the clean model-only result.

Important ambiguity: I did not find a saved external leaderboard record containing the literal score `0.98500` in the repository. The local experiment that contains an exact `0.9850` validation metric is a different file:

```text
outputs/convnext_224x672_pool_query_e1/submission.csv
sha256=74e3638294af7b7c6c30b4788b7eb931e0fd6f5a1e9ad9581d5368d43fe3bd99
```

That 224x672 file differs from the current canonical `outputs/submission.csv` in 40 predictions.

## 1. Which Local File Produced The 0.98500 Submission

If "0.98500" refers to the file I most recently handed over as `submission.csv`, the local file is:

```text
outputs/submission.csv
```

Source file:

```text
outputs/convnext_best_confusion_rules_contextual_predict/submission.csv
```

Local validation of the CSV:

```text
columns=id,label
rows=5000
unique_ids=5000
label_regex=[0-9A-Z]{1,5}
label_ok=True
```

If "0.98500" refers to the local validation metric `count_final_exact_acc=0.9850` / `pattern_final_exact_acc=0.9850`, then the file is:

```text
outputs/convnext_224x672_pool_query_e1/submission.csv
```

## 2. Corresponding Git Commit Hash

For the current canonical `outputs/submission.csv`:

```text
contextual_rules_code_commit=1761c7820acdc9de37b7e05469c809214f3d598c
contextual_rules_docs_commit=ef4e85a6b58e40f16efd4d41b2804cba28866af7
current_audit_start_HEAD=ae8ff30fe38634238d27a3e84bc1e686a88c01cb
```

The actual prediction log does not embed `git rev-parse HEAD`, so the exact runtime commit cannot be proven solely from the log. The best local evidence is that `1761c78` added the contextual rule implementation and `ef4e85a` recorded the evaluation/prediction artifacts.

For the separate 224x672 local `0.9850` metric candidate:

```text
docs_record_commit=02329b7266ca26abf7f154c2c77e6a6cc0b22c5f
```

## 3. Whether There Was Uncommitted Code Diff

No historical dirty-state snapshot was saved in the training or prediction logs, so I cannot honestly prove whether there was an uncommitted diff at the exact runtime.

Current audit status before creating this document:

```text
HEAD=ae8ff30fe38634238d27a3e84bc1e686a88c01cb
untracked=docs/video_model_support_matrix.html
tracked_diff=none
staged_diff=none
```

`docs/video_model_support_matrix.html` was already untracked during the audit and was not used for model training or prediction.

## 4. Complete Training Command

The current canonical `outputs/submission.csv` was not produced by a new training run. It was produced by predicting from this checkpoint:

```text
checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt
```

The checkpoint training command was:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_192x576_pool_query_lr2e6_e1 \
  --checkpoint-dir checkpoints/convnext_192x576_pool_query_lr2e6_e1 \
  --init-checkpoint checkpoints/convnext_192x576_pool_query_e2/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool_query \
  --normalization imagenet \
  --image-height 192 \
  --image-width 576 \
  --learning-rate 2e-6 \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda \
  --char-loss-weight 1.5 \
  --color-loss-weight 0.5 \
  --no-scheduler
```

The separate 224x672 candidate command was:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_224x672_pool_query_e1 \
  --checkpoint-dir checkpoints/convnext_224x672_pool_query_e1 \
  --init-checkpoint checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt \
  --model convnext_tiny \
  --slot-extractor pool_query \
  --normalization imagenet \
  --image-height 224 \
  --image-width 672 \
  --learning-rate 1e-6 \
  --epochs 1 \
  --batch-size 6 \
  --num-workers 2 \
  --device cuda \
  --char-loss-weight 1.5 \
  --color-loss-weight 0.5 \
  --no-scheduler
```

## 5. Config And Hyperparameters

For the checkpoint used by the current canonical submission:

```text
model=convnext_tiny
pretrained_backbone=TorchVision public ImageNet weights
slot_extractor=pool_query
slot_pooling=avgmax
slot_context=on
image_size=192x576
normalization=imagenet
dropout=0.1
label_smoothing=0.03
optimizer=AdamW
learning_rate=2e-6
epochs=1
batch_size=8
weight_decay=1e-4
max_grad_norm=5.0
char_loss_weight=1.5
color_loss_weight=0.5
char_class_weights=per-slot enabled
color_class_weights=per-slot enabled
scheduler=off
amp=on
ema=on, decay=0.999
train_augmentation=on
balanced_sampler=on, by color pattern
```

For the final contextual prediction:

```text
checkpoint=checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt
batch_size=16
image_size=192x576
normalization=imagenet
tta_shifts=0,-2,2
tta_scales=1,0.95,1.05
color_decode_method=threshold
color_thresholds=0.950,0.800,0.550,0.900,0.750
char_decode_method=confusion_rules
confusion_rule_set=contextual
contextual_rule_count=41
```

## 6. Random Seed And Train/Val Split

```text
seed=2026
val_ratio=0.1
train_samples=45000
val_samples=5000
split_method=stratified by color column
```

Implementation details from `src/data.py`:

```text
df.groupby("color", sort=True)
random.Random(seed).shuffle(indices) inside each color group
round(group_size * val_ratio) validation samples per group
shuffle final train and val index lists with the same seeded RNG
```

This is deterministic as long as `labels.csv` and pandas grouping behavior remain unchanged.

## 7. Whether Training Continued From A Checkpoint

Yes. The checkpoint used for the canonical submission was:

```text
checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt
```

It continued from:

```text
checkpoints/convnext_192x576_pool_query_e2/baseline_best.pt
```

The final `outputs/submission.csv` step itself was predict-only, not further training.

## 8. Init-Checkpoint Lineage

Direct lineage of the canonical submission checkpoint:

```text
convnext_192x576_pool_query_lr2e6_e1
  init -> convnext_192x576_pool_query_e2
    init -> convnext_192x576_charweight_e2
      init -> convnext_160x480_charweight_e2
        init -> convnext_highres_ft_charweight_e2
```

Key local checkpoint metadata:

```text
checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt
  epoch=1
  init=checkpoints\convnext_192x576_pool_query_e2\baseline_best.pt
  calibrated_final_exact_acc=0.9874

checkpoints/convnext_192x576_pool_query_e2/baseline_best.pt
  epoch=2
  init=checkpoints\convnext_192x576_charweight_e2\baseline_best.pt
  calibrated_final_exact_acc=0.9870

checkpoints/convnext_192x576_charweight_e2/baseline_best.pt
  epoch=2
  init=checkpoints\convnext_160x480_charweight_e2\baseline_best.pt
  calibrated_final_exact_acc=0.9866

checkpoints/convnext_160x480_charweight_e2/baseline_best.pt
  epoch=2
  init=checkpoints\convnext_highres_ft_charweight_e2\baseline_best.pt
  calibrated_final_exact_acc=0.9860
```

The backbone itself is a public TorchVision ConvNeXt-Tiny ImageNet-pretrained model.

## 9. Training Logs And Validation Metrics

Main checkpoint training log:

```text
logs/convnext_192x576_pool_query_lr2e6_e1.out.log
```

Main checkpoint result:

```text
epoch=1
selected_model=ema
train_loss=0.3837
val_loss=0.0375
final_exact_acc=0.9856
threshold_final_exact_acc=0.9874
calibrated_final_exact_acc=0.9874
char_slot_acc=0.9935
char_sequence_acc=0.9676
color_slot_acc=0.9993
color_pattern_acc=0.9968
char_oracle_final_exact_acc=0.9988
color_oracle_final_exact_acc=0.9886
color_thresholds=0.950,0.800,0.550,0.900,0.750
```

Contextual rule evaluation log:

```text
logs/convnext_best_confusion_rules_contextual_eval.out.log
```

Contextual validation result:

```text
loss=0.0375
final_exact_acc=0.9856
calibrated_final_exact_acc=0.9962
confusion_rules_final_exact_acc=0.9962
char_slot_acc=0.9935
color_slot_acc=0.9993
color_pattern_acc=0.9968
validation_errors=19
```

Again: `0.9962` is from validation-tuned contextual rules and should not be treated as a clean generalization estimate.

Separate 224x672 candidate log:

```text
logs/convnext_224x672_pool_query_e1.out.log
```

Separate 224x672 candidate result:

```text
epoch=1
selected_model=raw
train_loss=0.3969
val_loss=0.0412
final_exact_acc=0.9848
threshold_final_exact_acc=0.9858
calibrated_final_exact_acc=0.9858
count_final_exact_acc=0.9850
pattern_final_exact_acc=0.9850
char_slot_acc=0.9926
char_sequence_acc=0.9638
color_slot_acc=0.9995
color_pattern_acc=0.9976
```

## 10. Environment Versions

Current local environment during this audit:

```text
Python=3.13.5 | packaged by Anaconda, Inc.
Python executable=C:\ProgramData\anaconda3\python.exe
OS=Windows-11-10.0.26200-SP0
torch=2.7.1+cu128
torchvision=0.22.1+cu128
torch CUDA build=12.8
CUDA available=True
GPU=NVIDIA GeForce RTX 4060 Laptop GPU
GPU capability=(8, 9)
NVIDIA driver=596.21
nvidia-smi CUDA Version=13.2
GPU memory=8188 MiB
```

No historical environment lockfile was found for the exact original run, so this records the current reproducibility environment.

## 11. Ensemble, TTA, Threshold Calibration, Confusion Rules

For the canonical `outputs/submission.csv`:

```text
ensemble=no
eight_model_vote=no
TTA=yes, 3 horizontal shifts x 3 scales = 9 views
threshold_calibration=yes, per-slot color thresholds selected on validation
red_count_prior=stored in checkpoint but not used in final test decode because color_decode_method=threshold
color_pattern_prior=stored in checkpoint but not used in final test decode because color_decode_method=threshold
confusion_rules=yes
confusion_rule_set=contextual
contextual_rules=41
```

This is not a pure model-only submission. It uses validation-tuned post-processing, so it has overfitting risk.

For the separate 224x672 candidate:

```text
ensemble=no
eight_model_vote=no
TTA=yes, 9 views
threshold_calibration=yes
confusion_rules=no
final_color_decode_method=threshold
```

## 12. Final Submission Generation Command

Command that generated the source file for the current canonical submission:

```bash
python -u src/main.py \
  --data-dir "C:\Users\GJR79\xwechat_files\wxid_y2flsengm4t722_bc12\msg\file\2026-06\红色字符识别" \
  --output-dir outputs/convnext_best_confusion_rules_contextual_predict \
  --checkpoint-path checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt \
  --predict-only \
  --model convnext_tiny \
  --slot-extractor pool_query \
  --normalization imagenet \
  --image-height 192 \
  --image-width 576 \
  --batch-size 16 \
  --num-workers 2 \
  --device cuda \
  --use-confusion-rules \
  --confusion-rule-set contextual
```

The canonical file is the same bytes as the source prediction file:

```text
outputs/submission.csv
outputs/convnext_best_confusion_rules_contextual_predict/submission.csv
sha256=36f131cda8c5987bd960f89f2830cbe3bb9a5e1967b33e795e879e99b2e196e9
```

Equivalent copy command, if rebuilding the canonical path:

```powershell
Copy-Item outputs\convnext_best_confusion_rules_contextual_predict\submission.csv outputs\submission.csv -Force
```

Clean-model recommendation for reporting:

```text
Use checkpoints/convnext_192x576_pool_query_lr2e6_e1/baseline_best.pt as the main reproducible model.
Report contextual confusion rules separately as validation-tuned post-processing.
Do not describe the contextual-rule result as an unbiased validation or test estimate.
```
