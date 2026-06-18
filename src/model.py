import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: tuple[int, int] | None = None):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(nn.MaxPool2d(pool))
        super().__init__(*layers)


class BaselineCNN(nn.Module):
    def __init__(self, feature_dim: int = 256, num_slots: int = 5, num_chars: int = 36):
        super().__init__()
        self.num_slots = num_slots
        self.num_chars = num_chars
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 2)),
            ConvBlock(128, feature_dim, pool=(2, 1)),
            ConvBlock(feature_dim, feature_dim),
        )
        self.slot_pool = nn.AdaptiveAvgPool2d((1, num_slots))
        self.char_head = nn.Linear(feature_dim, num_chars)
        self.color_head = nn.Linear(feature_dim, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        slot_features = self.slot_pool(features).squeeze(2).permute(0, 2, 1).contiguous()
        char_logits = self.char_head(slot_features)
        color_logits = self.color_head(slot_features)
        return char_logits, color_logits
