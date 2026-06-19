import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_small,
    convnext_tiny,
)


CONVNEXT_BACKBONES = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
}


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


class FixedSlotFeatureRearranger(nn.Module):
    def __init__(self, in_dim: int, feature_dim: int, num_slots: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_slots, feature_dim) * 0.02)
        self.key = nn.Linear(in_dim, feature_dim)
        self.value = nn.Linear(in_dim, feature_dim)
        self.out_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = feature_dim ** -0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = features.flatten(2).transpose(1, 2).contiguous()
        keys = self.key(tokens)
        values = self.value(tokens)
        queries = self.query.expand(features.size(0), -1, -1)
        attention = torch.matmul(queries, keys.transpose(1, 2)) * self.scale
        attention = attention.softmax(dim=-1)
        slot_features = torch.matmul(attention, values)
        return self.out_norm(self.dropout(slot_features))


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

    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            getattr(self, "input_mean", 0.5),
            dtype=images.dtype,
            device=images.device,
        ).flatten()
        std = torch.as_tensor(
            getattr(self, "input_std", 0.5),
            dtype=images.dtype,
            device=images.device,
        ).flatten().clamp_min(1e-6)
        if mean.numel() == 1:
            mean = mean.expand(images.size(1))
        if std.numel() == 1:
            std = std.expand(images.size(1))
        mean = mean[: images.size(1)].view(1, -1, 1, 1)
        std = std[: images.size(1)].view(1, -1, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

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

        color_images = self.denormalize_images(images)
        raw_slots = self.raw_slot_pool(color_images).squeeze(2).permute(0, 2, 1).contiguous()
        red_minus_other = color_images[:, 0:1] - torch.maximum(color_images[:, 1:2], color_images[:, 2:3])
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


class PretrainedConvNeXtSlotModel(nn.Module):
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
        pretrained: bool = True,
        slot_extractor: str = "pool",
        backbone_name: str = "convnext_tiny",
    ):
        super().__init__()
        if slot_pooling not in {"avg", "max", "avgmax"}:
            raise ValueError(f"slot_pooling must be one of avg, max, avgmax; got {slot_pooling!r}")
        if slot_extractor not in {"pool", "query", "pool_query"}:
            raise ValueError(f"slot_extractor must be one of pool, query, pool_query; got {slot_extractor!r}")
        if backbone_name not in CONVNEXT_BACKBONES:
            supported = ", ".join(sorted(CONVNEXT_BACKBONES))
            raise ValueError(f"backbone_name must be one of {supported}; got {backbone_name!r}")
        self.num_slots = num_slots
        self.num_chars = num_chars
        self.position_specific_heads = position_specific_heads
        self.slot_pooling = slot_pooling
        self.use_slot_context = use_slot_context
        self.pretrained = pretrained
        self.slot_extractor = slot_extractor
        self.backbone_name = backbone_name

        backbone_factory, default_weights, backbone_dim = CONVNEXT_BACKBONES[backbone_name]
        weights = default_weights if pretrained else None
        convnext = backbone_factory(weights=weights)
        self.backbone = convnext.features

        self.slot_avg_pool = nn.AdaptiveAvgPool2d((1, num_slots))
        self.slot_max_pool = nn.AdaptiveMaxPool2d((1, num_slots))
        pooled_feature_dim = backbone_dim * 2 if slot_pooling == "avgmax" else backbone_dim
        self.slot_projection = nn.Sequential(
            nn.LayerNorm(pooled_feature_dim),
            nn.Linear(pooled_feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.slot_rearranger = FixedSlotFeatureRearranger(backbone_dim, feature_dim, num_slots, dropout)
        self.slot_fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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

    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            getattr(self, "input_mean", (0.485, 0.456, 0.406)),
            dtype=images.dtype,
            device=images.device,
        ).flatten()
        std = torch.as_tensor(
            getattr(self, "input_std", (0.229, 0.224, 0.225)),
            dtype=images.dtype,
            device=images.device,
        ).flatten().clamp_min(1e-6)
        if mean.numel() == 1:
            mean = mean.expand(images.size(1))
        if std.numel() == 1:
            std = std.expand(images.size(1))
        mean = mean[: images.size(1)].view(1, -1, 1, 1)
        std = std[: images.size(1)].view(1, -1, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

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
        pooled_slot_features = self.slot_projection(slot_features)
        if self.slot_extractor == "pool":
            slot_features = pooled_slot_features
        else:
            query_slot_features = self.slot_rearranger(features)
            if self.slot_extractor == "query":
                slot_features = query_slot_features
            else:
                slot_features = self.slot_fusion(torch.cat([pooled_slot_features, query_slot_features], dim=-1))
        slot_features = slot_features + self.position_embedding
        slot_features = self.slot_context(slot_features)

        color_images = self.denormalize_images(images)
        raw_slots = self.raw_slot_pool(color_images).squeeze(2).permute(0, 2, 1).contiguous()
        red_minus_other = color_images[:, 0:1] - torch.maximum(color_images[:, 1:2], color_images[:, 2:3])
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
