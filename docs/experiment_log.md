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

