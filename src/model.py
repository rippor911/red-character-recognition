import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: tuple[int, int] | None = None,
        dropout: float = 0.0,
    ):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        if pool is not None:
            layers.append(nn.MaxPool2d(pool))
        super().__init__(*layers)


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: tuple[int, int] | None = None,
        dropout: float = 0.0,
    ):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        if pool is not None:
            layers.append(nn.MaxPool2d(pool))
        super().__init__(*layers)


class SlotHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SlotContextBlock(nn.Module):
    def __init__(self, feature_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.depthwise = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size=3,
            padding=1,
            groups=feature_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, slot_features: torch.Tensor) -> torch.Tensor:
        context = self.norm(slot_features).transpose(1, 2).contiguous()
        context = self.depthwise(context)
        context = self.activation(context)
        context = self.pointwise(context).transpose(1, 2).contiguous()
        return slot_features + self.dropout(context)


class BaselineCNN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 384,
        head_hidden_dim: int = 384,
        num_slots: int = 5,
        num_chars: int = 36,
        dropout: float = 0.1,
        position_specific_heads: bool = True,
        slot_pooling: str = "avgmax",
        use_slot_context: bool = True,
    ):
        super().__init__()
        if slot_pooling not in {"avg", "max", "avgmax"}:
            raise ValueError(f"slot_pooling must be one of avg, max, avgmax; got {slot_pooling!r}")
        self.num_slots = num_slots
        self.num_chars = num_chars
        self.position_specific_heads = position_specific_heads
        self.slot_pooling = slot_pooling
        self.use_slot_context = use_slot_context
        self.backbone = nn.Sequential(
            ConvBlock(3, 48, pool=(2, 2)),
            ConvBlock(48, 64),
            DepthwiseSeparableBlock(64, 96, pool=(2, 2), dropout=dropout * 0.5),
            DepthwiseSeparableBlock(96, 160, pool=(2, 2), dropout=dropout * 0.5),
            DepthwiseSeparableBlock(160, feature_dim, pool=(2, 1), dropout=dropout),
            DepthwiseSeparableBlock(feature_dim, feature_dim, dropout=dropout),
        )
        self.slot_avg_pool = nn.AdaptiveAvgPool2d((1, num_slots))
        self.slot_max_pool = nn.AdaptiveMaxPool2d((1, num_slots))
        pooled_feature_dim = feature_dim * 2 if slot_pooling == "avgmax" else feature_dim
        self.slot_projection = (
            nn.Sequential(
                nn.LayerNorm(pooled_feature_dim),
                nn.Linear(pooled_feature_dim, feature_dim),
                nn.GELU(),
            )
            if pooled_feature_dim != feature_dim
            else nn.Identity()
        )
        self.raw_slot_pool = nn.AdaptiveAvgPool2d((1, num_slots))
        self.red_slot_avg_pool = nn.AdaptiveAvgPool2d((1, num_slots))
        self.red_slot_max_pool = nn.AdaptiveMaxPool2d((1, num_slots))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_slots, feature_dim))
        self.color_stat_proj = nn.Sequential(
            nn.Linear(5, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        if position_specific_heads:
            self.char_heads = nn.ModuleList(
                SlotHead(feature_dim, head_hidden_dim, num_chars, dropout)
                for _ in range(num_slots)
            )
            self.color_heads = nn.ModuleList(
                SlotHead(feature_dim, head_hidden_dim // 2, 2, dropout)
                for _ in range(num_slots)
            )
        else:
            self.char_head = SlotHead(feature_dim, head_hidden_dim, num_chars, dropout)
            self.color_head = SlotHead(feature_dim, head_hidden_dim // 2, 2, dropout)
        self.extra_color_stat_proj = nn.Linear(2, feature_dim, bias=False)
        nn.init.zeros_(self.extra_color_stat_proj.weight)
        self.slot_context = SlotContextBlock(feature_dim, dropout) if use_slot_context else nn.Identity()

    def apply_slot_heads(
        self,
        slot_features: torch.Tensor,
        heads: nn.ModuleList,
    ) -> torch.Tensor:
        return torch.stack(
            [head(slot_features[:, slot_idx, :]) for slot_idx, head in enumerate(heads)],
            dim=1,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        avg_slot_features = self.slot_avg_pool(features).squeeze(2).permute(0, 2, 1).contiguous()
        if self.slot_pooling == "avg":
            slot_features = avg_slot_features
        else:
            max_slot_features = self.slot_max_pool(features).squeeze(2).permute(0, 2, 1).contiguous()
            if self.slot_pooling == "max":
                slot_features = max_slot_features
            else:
                slot_features = torch.cat([avg_slot_features, max_slot_features], dim=-1)
        slot_features = self.slot_projection(slot_features)
        slot_features = slot_features + self.position_embedding
        slot_features = self.slot_context(slot_features)

        raw_slots = self.raw_slot_pool(images).squeeze(2).permute(0, 2, 1).contiguous()
        red_minus_other = images[:, 0:1] - torch.maximum(images[:, 1:2], images[:, 2:3])
        red_avg = self.red_slot_avg_pool(red_minus_other).squeeze(2).permute(0, 2, 1).contiguous()
        red_max = self.red_slot_max_pool(red_minus_other).squeeze(2).permute(0, 2, 1).contiguous()
        red_positive = red_minus_other.clamp_min(0.0)
        red_positive_avg = self.red_slot_avg_pool(red_positive).squeeze(2).permute(0, 2, 1).contiguous()
        red_coverage = self.red_slot_avg_pool((red_minus_other > 0.15).to(images.dtype)).squeeze(2).permute(0, 2, 1).contiguous()
        color_stats = torch.cat([raw_slots, red_avg, red_max], dim=-1)
        extra_color_stats = torch.cat([red_positive_avg, red_coverage], dim=-1)
        color_features = slot_features + self.color_stat_proj(color_stats) + self.extra_color_stat_proj(extra_color_stats)
        if self.position_specific_heads:
            char_logits = self.apply_slot_heads(slot_features, self.char_heads)
            color_logits = self.apply_slot_heads(color_features, self.color_heads)
        else:
            char_logits = self.char_head(slot_features)
            color_logits = self.color_head(color_features)
        return char_logits, color_logits
