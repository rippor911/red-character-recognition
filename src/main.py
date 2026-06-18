import argparse
import random
from dataclasses import dataclass, field
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import (
    CHARSET,
    RedCharacterTestDataset,
    RedCharacterTrainDataset,
    color_indices_from_scores,
    color_indices_from_pattern_prior,
    decode_batch_final,
    decode_batch_with_pattern_prior,
    decode_batch_with_threshold,
    load_image_tensor,
    split_train_val,
    validate_submission_frame,
    validate_test_frame,
    validate_train_frame,
)
from model import BaselineCNN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "dataset"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints"


@dataclass
class TrainConfig:
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    checkpoint_dir: Path = field(default_factory=lambda: DEFAULT_CHECKPOINT_DIR)
    image_size: tuple[int, int] = (64, 256)
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.03
    char_loss_weight: float = 1.0
    use_char_class_weight: bool = True
    max_char_class_weight: float = 3.0
    color_loss_weight: float = 1.0
    use_color_class_weight: bool = True
    max_color_class_weight: float = 3.0
    max_grad_norm: float = 5.0
    feature_dim: int = 384
    dropout: float = 0.1
    head_hidden_dim: int = 384
    position_specific_heads: bool = True
    slot_pooling: str = "avgmax"
    normalization: str = "dataset"
    normalization_samples: int = 2048
    use_augmentation: bool = True
    use_amp: bool = True
    use_scheduler: bool = True
    warmup_epochs: int = 2
    use_ema: bool = True
    ema_decay: float = 0.999
    use_tta: bool = True
    tta_shifts: tuple[int, ...] = (0, -2, 2)
    tta_scales: tuple[float, ...] = (1.0, 0.95, 1.05)
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 19
    use_pattern_prior: bool = True
    pattern_prior_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    pattern_confidence_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    use_balanced_sampler: bool = True
    save_val_diagnostics: bool = True
    max_error_samples: int = 200
    val_ratio: float = 0.1
    seed: int = 2026
    num_workers: int = 0
    device: str = "auto"
    debug_overfit: bool = False
    debug_samples: int = 128
    skip_test: bool = False
    expected_test_rows: Optional[int] = 5000


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train baseline CNN for red-character recognition.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--char-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-char-class-weight", action="store_true")
    parser.add_argument("--max-char-class-weight", type=float, default=3.0)
    parser.add_argument("--color-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-color-class-weight", action="store_true")
    parser.add_argument("--max-color-class-weight", type=float, default=3.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden-dim", type=int, default=384)
    parser.add_argument("--shared-heads", action="store_true")
    parser.add_argument("--slot-pooling", choices=["avg", "max", "avgmax"], default="avgmax")
    parser.add_argument("--normalization", choices=["dataset", "fixed", "none"], default="dataset")
    parser.add_argument("--normalization-samples", type=int, default=2048)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta-shifts", type=str, default="0,-2,2")
    parser.add_argument("--tta-scales", type=str, default="1,0.95,1.05")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    parser.add_argument("--no-pattern-prior", action="store_true")
    parser.add_argument("--pattern-prior-weights", type=str, default="0,0.25,0.5,1,1.5,2")
    parser.add_argument("--pattern-confidence-weights", type=str, default="0,0.25,0.5,1")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--no-val-diagnostics", action="store_true")
    parser.add_argument("--max-error-samples", type=int, default=200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--debug-overfit", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=128)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--expected-test-rows", type=int, default=5000)
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        char_loss_weight=args.char_loss_weight,
        use_char_class_weight=not args.no_char_class_weight,
        max_char_class_weight=args.max_char_class_weight,
        color_loss_weight=args.color_loss_weight,
        use_color_class_weight=not args.no_color_class_weight,
        max_color_class_weight=args.max_color_class_weight,
        max_grad_norm=args.max_grad_norm,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        head_hidden_dim=args.head_hidden_dim,
        position_specific_heads=not args.shared_heads,
        slot_pooling=args.slot_pooling,
        normalization=args.normalization,
        normalization_samples=args.normalization_samples,
        use_augmentation=not args.no_augment,
        use_amp=not args.no_amp,
        use_scheduler=not args.no_scheduler,
        warmup_epochs=args.warmup_epochs,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        use_tta=not args.no_tta,
        tta_shifts=parse_tta_shifts(args.tta_shifts),
        tta_scales=parse_tta_scales(args.tta_scales),
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        use_pattern_prior=not args.no_pattern_prior,
        pattern_prior_weights=parse_float_sequence(args.pattern_prior_weights),
        pattern_confidence_weights=parse_float_sequence(args.pattern_confidence_weights),
        use_balanced_sampler=not args.no_balanced_sampler,
        save_val_diagnostics=not args.no_val_diagnostics,
        max_error_samples=args.max_error_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        debug_overfit=args.debug_overfit,
        debug_samples=args.debug_samples,
        skip_test=args.skip_test,
        expected_test_rows=args.expected_test_rows,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> AbstractContextManager:
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_epochs = max(1, epochs)
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = max(0.0, min(1.0, progress))
        return float(0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_tta_shifts(value: str) -> tuple[int, ...]:
    shifts: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        shifts.append(int(item))
    if 0 not in shifts:
        shifts.insert(0, 0)
    deduped: list[int] = []
    for shift in shifts:
        if shift not in deduped:
            deduped.append(shift)
    return tuple(deduped) if deduped else (0,)


def parse_tta_scales(value: str) -> tuple[float, ...]:
    scales: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        scale = float(item)
        if scale <= 0:
            raise ValueError("tta scale values must be positive")
        scales.append(scale)
    if 1.0 not in scales:
        scales.insert(0, 1.0)
    deduped: list[float] = []
    for scale in scales:
        if scale not in deduped:
            deduped.append(scale)
    return tuple(deduped) if deduped else (1.0,)


def parse_float_sequence(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    deduped: list[float] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped) if deduped else (0.0,)


def translate_images(images: torch.Tensor, shift_x: int, fill_value: float = 1.0) -> torch.Tensor:
    if shift_x == 0:
        return images
    width = images.shape[-1]
    if abs(shift_x) >= width:
        return torch.full_like(images, fill_value)
    shifted = torch.full_like(images, fill_value)
    if shift_x > 0:
        src_x0, src_x1 = 0, width - shift_x
        dst_x0, dst_x1 = shift_x, width
    else:
        src_x0, src_x1 = -shift_x, width
        dst_x0, dst_x1 = 0, width + shift_x
    shifted[:, :, :, dst_x0:dst_x1] = images[:, :, :, src_x0:src_x1]
    return shifted


def scale_images(images: torch.Tensor, scale: float) -> torch.Tensor:
    if abs(float(scale) - 1.0) < 1e-6:
        return images
    height, width = images.shape[-2:]
    target_height = max(8, int(round(height * float(scale))))
    target_width = max(8, int(round(width * float(scale))))
    return F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def forward_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    tta_shifts: Sequence[int],
    tta_scales: Sequence[float] = (1.0,),
    fill_value: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tta_shifts) <= 1 and len(tta_scales) <= 1:
        return model(images)

    char_sum: Optional[torch.Tensor] = None
    color_sum: Optional[torch.Tensor] = None
    view_count = 0
    for scale in tta_scales:
        scaled_images = scale_images(images, scale)
        for shift_x in tta_shifts:
            view = translate_images(scaled_images, shift_x=shift_x, fill_value=fill_value)
            char_logits, color_logits = model(view)
            char_sum = char_logits if char_sum is None else char_sum + char_logits
            color_sum = color_logits if color_sum is None else color_sum + color_logits
            view_count += 1

    assert char_sum is not None and color_sum is not None
    return char_sum / float(view_count), color_sum / float(view_count)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            value = value.detach()
            shadow_value = self.shadow[key]
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                shadow_value.copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.state_dict(), strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)


def load_train_data(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    labels_path = Path(data_dir) / "train" / "labels.csv"
    train_df = pd.read_csv(labels_path, dtype={"filename": str, "color": str, "all_label": str})
    validate_train_frame(train_df)
    return train_df


def load_test_data(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    sample_path = Path(data_dir) / "submission_sample.csv"
    test_df = pd.read_csv(sample_path, dtype={"id": str})
    validate_test_frame(test_df)
    return test_df


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle and sampler is None else None,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def compute_loss(
    char_logits: torch.Tensor,
    color_logits: torch.Tensor,
    char_target: torch.Tensor,
    color_target: torch.Tensor,
    label_smoothing: float = 0.0,
    char_loss_weight: float = 1.0,
    color_loss_weight: float = 1.0,
    char_class_weights: Optional[torch.Tensor] = None,
    color_class_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if char_class_weights is not None and char_class_weights.ndim == 2:
        char_loss = torch.stack(
            [
                F.cross_entropy(
                    char_logits[:, slot_idx, :],
                    char_target[:, slot_idx],
                    label_smoothing=label_smoothing,
                    weight=char_class_weights[slot_idx],
                )
                for slot_idx in range(char_logits.size(1))
            ]
        ).mean()
    else:
        char_loss = F.cross_entropy(
            char_logits.reshape(-1, len(CHARSET)),
            char_target.reshape(-1),
            label_smoothing=label_smoothing,
            weight=char_class_weights,
        )

    if color_class_weights is not None and color_class_weights.ndim == 2:
        color_loss = torch.stack(
            [
                F.cross_entropy(
                    color_logits[:, slot_idx, :],
                    color_target[:, slot_idx],
                    weight=color_class_weights[slot_idx],
                )
                for slot_idx in range(color_logits.size(1))
            ]
        ).mean()
    else:
        color_loss = F.cross_entropy(
            color_logits.reshape(-1, 2),
            color_target.reshape(-1),
            weight=color_class_weights,
        )
    loss = char_loss * char_loss_weight + color_loss * color_loss_weight
    return loss, char_loss, color_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
    label_smoothing: float,
    char_loss_weight: float,
    color_loss_weight: float,
    max_grad_norm: float,
    ema: Optional[ModelEMA] = None,
    char_class_weights: Optional[torch.Tensor] = None,
    color_class_weights: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_char_loss = 0.0
    total_color_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        char_target = batch["char_target"].to(device)
        color_target = batch["color_target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            char_logits, color_logits = model(images)
            loss, char_loss, color_loss = compute_loss(
                char_logits,
                color_logits,
                char_target,
                color_target,
                label_smoothing=label_smoothing,
                char_loss_weight=char_loss_weight,
                color_loss_weight=color_loss_weight,
                char_class_weights=char_class_weights,
                color_class_weights=color_class_weights,
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_char_loss += char_loss.item() * batch_size
        total_color_loss += color_loss.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "char_loss": total_char_loss / total_samples,
        "color_loss": total_color_loss / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_steps: int = 19,
    tta_shifts: Sequence[int] = (0,),
    tta_scales: Sequence[float] = (1.0,),
    tta_fill_value: float = 1.0,
    pattern_candidates: Optional[Sequence[str]] = None,
    pattern_log_priors: Optional[Sequence[float]] = None,
    pattern_prior_weights: Sequence[float] = (0.0,),
    pattern_confidence_weights: Sequence[float] = (0.0,),
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_char_loss = 0.0
    total_color_loss = 0.0
    total_samples = 0
    final_correct = 0
    char_slot_correct = 0
    color_slot_correct = 0
    color_pattern_correct = 0
    total_slots = 0
    target_length_correct = 0
    char_correct_by_slot = torch.zeros(5, dtype=torch.long)
    color_correct_by_slot = torch.zeros(5, dtype=torch.long)
    total_by_slot = torch.zeros(5, dtype=torch.long)
    all_pred_chars: list[torch.Tensor] = []
    all_red_probs: list[torch.Tensor] = []
    all_char_confidence: list[torch.Tensor] = []
    all_target_chars: list[torch.Tensor] = []
    all_target_colors: list[torch.Tensor] = []
    all_target_final: list[str] = []

    for batch in loader:
        images = batch["image"].to(device)
        char_target = batch["char_target"].to(device)
        color_target = batch["color_target"].to(device)
        char_logits, color_logits = forward_with_tta(model, images, tta_shifts, tta_scales, fill_value=tta_fill_value)
        loss, char_loss, color_loss = compute_loss(char_logits, color_logits, char_target, color_target)

        pred_chars = char_logits.argmax(dim=-1)
        pred_colors = color_logits.argmax(dim=-1)
        red_probs = torch.softmax(color_logits, dim=-1)[..., 1]
        char_confidence = torch.softmax(char_logits, dim=-1).max(dim=-1).values
        pred_final = decode_batch_final(pred_chars, pred_colors, red_probs, fallback_if_empty=True)
        target_final = decode_batch_final(char_target, color_target, fallback_if_empty=False)

        batch_size = images.size(0)
        total_samples += batch_size
        total_slots += batch_size * pred_chars.size(1)
        total_loss += loss.item() * batch_size
        total_char_loss += char_loss.item() * batch_size
        total_color_loss += color_loss.item() * batch_size
        final_correct += sum(pred == target for pred, target in zip(pred_final, target_final))
        char_slot_correct += (pred_chars == char_target).sum().item()
        color_slot_correct += (pred_colors == color_target).sum().item()
        color_pattern_correct += (pred_colors == color_target).all(dim=1).sum().item()
        target_length_correct += sum(len(pred) == len(target) for pred, target in zip(pred_final, target_final))
        slot_count = pred_chars.size(1)
        char_correct_by_slot[:slot_count] += (pred_chars == char_target).detach().cpu().sum(dim=0)
        color_correct_by_slot[:slot_count] += (pred_colors == color_target).detach().cpu().sum(dim=0)
        total_by_slot[:slot_count] += batch_size
        all_pred_chars.append(pred_chars.detach().cpu())
        all_red_probs.append(red_probs.detach().cpu())
        all_char_confidence.append(char_confidence.detach().cpu())
        all_target_chars.append(char_target.detach().cpu())
        all_target_colors.append(color_target.detach().cpu())
        all_target_final.extend(target_final)

    pred_chars_all = torch.cat(all_pred_chars, dim=0)
    red_probs_all = torch.cat(all_red_probs, dim=0)
    char_confidence_all = torch.cat(all_char_confidence, dim=0)
    target_chars_all = torch.cat(all_target_chars, dim=0)
    target_colors_all = torch.cat(all_target_colors, dim=0)
    best_threshold = 0.5
    best_threshold_correct = final_correct
    threshold_steps = max(2, threshold_steps)
    threshold_min = max(0.0, min(1.0, threshold_min))
    threshold_max = max(0.0, min(1.0, threshold_max))
    if threshold_min > threshold_max:
        threshold_min, threshold_max = threshold_max, threshold_min
    threshold_candidates = np.linspace(threshold_min, threshold_max, threshold_steps).tolist()
    threshold_candidates.append(0.5)
    threshold_candidates = sorted({round(float(item), 6) for item in threshold_candidates})
    best_thresholds = [0.5] * 5
    for threshold in threshold_candidates:
        threshold_final = decode_batch_with_threshold(
            pred_chars_all,
            red_probs_all,
            threshold=threshold,
            fallback_if_empty=True,
        )
        threshold_correct = sum(
            pred == target for pred, target in zip(threshold_final, all_target_final)
        )
        if threshold_correct > best_threshold_correct or (
            threshold_correct == best_threshold_correct
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold_correct = threshold_correct
            best_threshold = threshold
            best_thresholds = [threshold] * 5

    for slot in range(5):
        current_threshold = best_thresholds[slot]
        for threshold in threshold_candidates:
            candidate_thresholds = best_thresholds.copy()
            candidate_thresholds[slot] = threshold
            threshold_final = decode_batch_with_threshold(
                pred_chars_all,
                red_probs_all,
                threshold=candidate_thresholds,
                fallback_if_empty=True,
            )
            threshold_correct = sum(
                pred == target for pred, target in zip(threshold_final, all_target_final)
            )
            current_distance = abs(current_threshold - 0.5)
            candidate_distance = abs(threshold - 0.5)
            if threshold_correct > best_threshold_correct or (
                threshold_correct == best_threshold_correct
                and candidate_distance < current_distance
            ):
                best_threshold_correct = threshold_correct
                best_thresholds = candidate_thresholds
                current_threshold = threshold

    best_threshold = float(sum(best_thresholds) / len(best_thresholds))
    best_threshold_colors = color_indices_from_scores(red_probs_all, best_thresholds)
    best_threshold_color_correct = (best_threshold_colors == target_colors_all).all(dim=1).sum().item()
    best_pattern_correct = -1
    best_pattern_color_correct = 0
    best_pattern_prior_weight = 0.0
    best_pattern_confidence_weight = 0.0
    use_pattern_candidates = bool(pattern_candidates)
    if use_pattern_candidates:
        candidate_weights = list(pattern_prior_weights) or [0.0]
        if 0.0 not in candidate_weights:
            candidate_weights.insert(0, 0.0)
        candidate_confidence_weights = list(pattern_confidence_weights) or [0.0]
        if 0.0 not in candidate_confidence_weights:
            candidate_confidence_weights.insert(0, 0.0)
        for prior_weight in candidate_weights:
            for confidence_weight in candidate_confidence_weights:
                pattern_final = decode_batch_with_pattern_prior(
                    pred_chars_all,
                    red_probs_all,
                    patterns=pattern_candidates,
                    pattern_log_priors=pattern_log_priors,
                    prior_weight=float(prior_weight),
                    char_confidence=char_confidence_all,
                    confidence_weight=float(confidence_weight),
                    fallback_if_empty=True,
                )
                pattern_correct = sum(
                    pred == target for pred, target in zip(pattern_final, all_target_final)
                )
                pattern_colors = color_indices_from_pattern_prior(
                    red_probs_all,
                    patterns=pattern_candidates,
                    pattern_log_priors=pattern_log_priors,
                    prior_weight=float(prior_weight),
                    char_confidence=char_confidence_all,
                    confidence_weight=float(confidence_weight),
                )
                pattern_color_correct = (pattern_colors == target_colors_all).all(dim=1).sum().item()
                current_complexity = abs(best_pattern_confidence_weight) + abs(best_pattern_prior_weight)
                candidate_complexity = abs(float(confidence_weight)) + abs(float(prior_weight))
                if pattern_correct > best_pattern_correct or (
                    pattern_correct == best_pattern_correct
                    and candidate_complexity < current_complexity
                ):
                    best_pattern_correct = pattern_correct
                    best_pattern_color_correct = int(pattern_color_correct)
                    best_pattern_prior_weight = float(prior_weight)
                    best_pattern_confidence_weight = float(confidence_weight)
    else:
        best_pattern_correct = best_threshold_correct
        best_pattern_color_correct = int(best_threshold_color_correct)

    calibrated_correct = best_threshold_correct
    color_decode_method = "threshold"
    if use_pattern_candidates and best_pattern_correct > best_threshold_correct:
        calibrated_correct = best_pattern_correct
        color_decode_method = "pattern_confidence" if best_pattern_confidence_weight != 0.0 else "pattern_prior"
    if color_decode_method in {"pattern_prior", "pattern_confidence"}:
        calibrated_final = decode_batch_with_pattern_prior(
            pred_chars_all,
            red_probs_all,
            patterns=pattern_candidates or (),
            pattern_log_priors=pattern_log_priors,
            prior_weight=best_pattern_prior_weight,
            char_confidence=char_confidence_all,
            confidence_weight=best_pattern_confidence_weight,
            fallback_if_empty=True,
        )
        calibrated_colors = color_indices_from_pattern_prior(
            red_probs_all,
            patterns=pattern_candidates or (),
            pattern_log_priors=pattern_log_priors,
            prior_weight=best_pattern_prior_weight,
            char_confidence=char_confidence_all,
            confidence_weight=best_pattern_confidence_weight,
        )
    else:
        calibrated_final = decode_batch_with_threshold(
            pred_chars_all,
            red_probs_all,
            threshold=best_thresholds,
            fallback_if_empty=True,
        )
        calibrated_colors = best_threshold_colors

    calibrated_length_correct = sum(
        len(pred) == len(target) for pred, target in zip(calibrated_final, all_target_final)
    )
    calibrated_correct_flags = [
        int(pred == target) for pred, target in zip(calibrated_final, all_target_final)
    ]
    target_lengths = target_colors_all.sum(dim=1).long().tolist()
    target_patterns = [
        color_indices_to_pattern(row)
        for row in target_colors_all.long().tolist()
    ]
    calibrated_pattern_correct = (calibrated_colors == target_colors_all).all(dim=1).sum().item()
    char_sequence_correct = (pred_chars_all == target_chars_all).all(dim=1).sum().item()
    color_oracle_final = decode_batch_final(
        pred_chars_all,
        target_colors_all,
        red_scores=None,
        fallback_if_empty=False,
    )
    char_oracle_final = decode_batch_final(
        target_chars_all,
        calibrated_colors,
        red_scores=red_probs_all,
        fallback_if_empty=True,
    )
    color_oracle_correct = sum(
        pred == target for pred, target in zip(color_oracle_final, all_target_final)
    )
    char_oracle_correct = sum(
        pred == target for pred, target in zip(char_oracle_final, all_target_final)
    )

    metrics = {
        "loss": total_loss / total_samples,
        "char_loss": total_char_loss / total_samples,
        "color_loss": total_color_loss / total_samples,
        "final_exact_acc": final_correct / total_samples,
        "char_slot_acc": char_slot_correct / total_slots,
        "char_sequence_acc": char_sequence_correct / total_samples,
        "color_slot_acc": color_slot_correct / total_slots,
        "color_pattern_acc": color_pattern_correct / total_samples,
        "target_length_acc": target_length_correct / total_samples,
        "threshold_final_exact_acc": best_threshold_correct / total_samples,
        "threshold_color_acc": best_threshold_color_correct / total_samples,
        "color_threshold": best_threshold,
        "color_thresholds": best_thresholds,
        "pattern_final_exact_acc": best_pattern_correct / total_samples,
        "pattern_color_acc": best_pattern_color_correct / total_samples,
        "pattern_prior_weight": best_pattern_prior_weight,
        "pattern_confidence_weight": best_pattern_confidence_weight,
        "calibrated_final_exact_acc": calibrated_correct / total_samples,
        "char_oracle_final_exact_acc": char_oracle_correct / total_samples,
        "color_oracle_final_exact_acc": color_oracle_correct / total_samples,
        "calibrated_color_pattern_acc": calibrated_pattern_correct / total_samples,
        "calibrated_length_acc": calibrated_length_correct / total_samples,
        "calibrated_gain": (calibrated_correct - final_correct) / total_samples,
        "color_decode_method": color_decode_method,
        "threshold_gain": (best_threshold_correct - final_correct) / total_samples,
    }
    for slot in range(5):
        denominator = max(1, int(total_by_slot[slot].item()))
        metrics[f"char_slot_{slot + 1}_acc"] = int(char_correct_by_slot[slot].item()) / denominator
        metrics[f"color_slot_{slot + 1}_acc"] = int(color_correct_by_slot[slot].item()) / denominator
    for target_length in range(1, 6):
        flags = [
            correct
            for correct, length in zip(calibrated_correct_flags, target_lengths)
            if int(length) == target_length
        ]
        metrics[f"samples_len_{target_length}"] = len(flags)
        metrics[f"calibrated_final_exact_acc_len_{target_length}"] = (
            sum(flags) / len(flags) if flags else 0.0
        )
    for pattern in sorted(set(target_patterns)):
        flags = [
            correct
            for correct, target_pattern in zip(calibrated_correct_flags, target_patterns)
            if target_pattern == pattern
        ]
        metrics[f"samples_pattern_{pattern}"] = len(flags)
        metrics[f"calibrated_final_exact_acc_pattern_{pattern}"] = (
            sum(flags) / len(flags) if flags else 0.0
        )
    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def eval_metric_is_better(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
    candidate_score = float(candidate["calibrated_final_exact_acc"])
    incumbent_score = float(incumbent["calibrated_final_exact_acc"])
    return candidate_score > incumbent_score or (
        candidate_score == incumbent_score and float(candidate["loss"]) < float(incumbent["loss"])
    )


def threshold_to_list(threshold: float | Sequence[float], num_slots: int = 5) -> list[float]:
    if isinstance(threshold, (list, tuple)):
        values = [float(item) for item in threshold]
        if len(values) != num_slots:
            raise ValueError(f"threshold must contain {num_slots} values, got {len(values)}")
        return values
    return [float(threshold)] * num_slots


def format_thresholds(threshold: float | Sequence[float]) -> str:
    return ",".join(f"{value:.3f}" for value in threshold_to_list(threshold))


def compute_dataset_normalization(
    df: pd.DataFrame,
    image_dir: Path,
    image_size: tuple[int, int],
    max_samples: int,
    seed: int,
) -> tuple[float, float]:
    if df.empty:
        return 0.5, 0.5
    sample_count = len(df) if max_samples <= 0 else min(len(df), max_samples)
    if sample_count < len(df):
        sampled_df = df.sample(n=sample_count, random_state=seed).reset_index(drop=True)
    else:
        sampled_df = df.sort_values("filename").reset_index(drop=True)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for filename in sampled_df["filename"].astype(str):
        image = load_image_tensor(
            image_dir / filename,
            image_size=image_size,
            augment=False,
            normalize_mean=0.0,
            normalize_std=1.0,
        )
        total_sum += float(image.sum().item())
        total_sq_sum += float((image * image).sum().item())
        total_count += int(image.numel())

    mean = total_sum / max(1, total_count)
    variance = max(1e-6, total_sq_sum / max(1, total_count) - mean * mean)
    std = max(0.05, float(np.sqrt(variance)))
    return float(mean), std


def resolve_input_normalization(
    config: TrainConfig,
    train_split: pd.DataFrame,
    train_image_dir: Path,
) -> tuple[float, float, float]:
    if config.normalization == "none":
        mean, std = 0.0, 1.0
    elif config.normalization == "fixed":
        mean, std = 0.5, 0.5
    else:
        mean, std = compute_dataset_normalization(
            train_split,
            image_dir=train_image_dir,
            image_size=config.image_size,
            max_samples=config.normalization_samples,
            seed=config.seed,
        )
    fill_value = (1.0 - mean) / max(std, 1e-6)
    return mean, std, fill_value


def compute_color_class_weights(
    df: pd.DataFrame,
    device: torch.device,
    max_weight: float,
    smoothing: float = 1.0,
) -> torch.Tensor:
    counts = torch.zeros(5, 2, dtype=torch.float32)
    for color in df["color"].astype(str):
        normalized = color.strip().lower()
        for slot_idx, value in enumerate(normalized[:5]):
            if value == "u":
                counts[slot_idx, 0] += 1
            elif value == "r":
                counts[slot_idx, 1] += 1
    counts = counts + max(0.0, smoothing)
    weights = counts.sum(dim=1, keepdim=True) / (2.0 * counts)
    if max_weight > 0:
        weights = torch.clamp(weights, max=max_weight)
    weights = weights / weights.mean(dim=1, keepdim=True)
    return weights.to(device)


def compute_char_class_weights(
    df: pd.DataFrame,
    device: torch.device,
    max_weight: float,
    smoothing: float = 1.0,
) -> torch.Tensor:
    char_to_idx = {char: idx for idx, char in enumerate(CHARSET)}
    counts = torch.zeros(5, len(CHARSET), dtype=torch.float32)
    for label in df["all_label"].astype(str):
        for slot_idx, char in enumerate(label.strip().upper()[:5]):
            if char in char_to_idx:
                counts[slot_idx, char_to_idx[char]] += 1
    counts = counts + max(0.0, smoothing)
    weights = counts.sum(dim=1, keepdim=True) / (len(CHARSET) * counts)
    if max_weight > 0:
        weights = torch.clamp(weights, max=max_weight)
    weights = weights / weights.mean(dim=1, keepdim=True)
    return weights.to(device)


def summarize_weight_tensor(weights: torch.Tensor) -> str:
    values = weights.detach().cpu()
    return (
        f"shape={tuple(values.shape)} "
        f"min={values.min().item():.4f} "
        f"max={values.max().item():.4f} "
        f"mean={values.mean().item():.4f}"
    )


def normalize_color_pattern(color: str) -> str:
    pattern = str(color).strip().lower()
    if len(pattern) != 5 or any(char not in {"r", "u"} for char in pattern):
        return "unknown"
    return pattern


def build_color_pattern_sampler(
    df: pd.DataFrame,
    seed: int,
) -> tuple[WeightedRandomSampler, dict[str, int]]:
    patterns = [normalize_color_pattern(color) for color in df["color"].astype(str)]
    counts = pd.Series(patterns).value_counts().sort_index().to_dict()
    sample_weights = torch.tensor(
        [1.0 / max(1, int(counts[pattern])) for pattern in patterns],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler, {str(key): int(value) for key, value in counts.items()}


def format_count_summary(counts: dict[str, int], limit: int = 5) -> str:
    if not counts:
        return "none"
    pieces = [
        f"{pattern}:{count}"
        for pattern, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
    if len(counts) > limit:
        pieces.append("...")
    return ", ".join(pieces)


def build_color_pattern_prior(df: pd.DataFrame, smoothing: float = 1.0) -> tuple[list[str], list[float]]:
    counts: dict[str, int] = {}
    for color in df["color"].astype(str):
        pattern = normalize_color_pattern(color)
        if pattern == "unknown":
            continue
        if "r" not in pattern:
            continue
        counts[pattern] = counts.get(pattern, 0) + 1

    if not counts:
        for mask in range(1, 1 << 5):
            pattern = "".join("r" if mask & (1 << slot) else "u" for slot in range(5))
            counts[pattern] = 0

    patterns = sorted(counts, key=lambda pattern: (-counts[pattern], pattern))
    smoothing = max(0.0, smoothing)
    total = float(sum(counts.values())) + smoothing * len(patterns)
    if total <= 0.0:
        total = float(len(patterns))
        smoothing = 1.0
    log_priors = [
        float(np.log((counts[pattern] + smoothing) / total))
        for pattern in patterns
    ]
    return patterns, log_priors


def format_pattern_summary(patterns: Sequence[str], log_priors: Sequence[float], limit: int = 5) -> str:
    if not patterns:
        return "none"
    pieces = []
    for pattern, log_prior in list(zip(patterns, log_priors))[:limit]:
        pieces.append(f"{pattern}:{float(np.exp(log_prior)):.3f}")
    if len(patterns) > limit:
        pieces.append("...")
    return ", ".join(pieces)


def char_indices_to_string(indices: list[int]) -> str:
    return "".join(CHARSET[int(index)] for index in indices)


def color_indices_to_pattern(indices: list[int]) -> str:
    return "".join("r" if int(index) == 1 else "u" for index in indices)


@torch.no_grad()
def save_validation_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    color_threshold: float | Sequence[float],
    max_error_samples: int,
    tta_shifts: Sequence[int] = (0,),
    tta_scales: Sequence[float] = (1.0,),
    tta_fill_value: float = 1.0,
    color_decode_method: str = "threshold",
    color_pattern_candidates: Sequence[str] = (),
    color_pattern_log_priors: Sequence[float] = (),
    pattern_prior_weight: float = 0.0,
    pattern_confidence_weight: float = 0.0,
) -> tuple[Path, Path]:
    model.eval()
    rows: list[dict[str, object]] = []
    use_pattern_prior = color_decode_method in {"pattern_prior", "pattern_confidence"} and bool(color_pattern_candidates)

    for batch in loader:
        images = batch["image"].to(device)
        char_target = batch["char_target"].to(device)
        color_target = batch["color_target"].to(device)
        filenames = list(batch["filename"])

        char_logits, color_logits = forward_with_tta(model, images, tta_shifts, tta_scales, fill_value=tta_fill_value)
        pred_chars = char_logits.argmax(dim=-1)
        pred_colors = color_logits.argmax(dim=-1)
        red_probs = torch.softmax(color_logits, dim=-1)[..., 1]
        char_conf = torch.softmax(char_logits, dim=-1).max(dim=-1).values
        threshold_colors = color_indices_from_scores(red_probs, color_threshold)
        if bool(color_pattern_candidates):
            pattern_colors = color_indices_from_pattern_prior(
                red_probs,
                patterns=color_pattern_candidates,
                pattern_log_priors=color_pattern_log_priors,
                prior_weight=pattern_prior_weight,
                char_confidence=char_conf,
                confidence_weight=pattern_confidence_weight,
            )
        else:
            pattern_colors = threshold_colors

        pred_final = decode_batch_final(pred_chars, pred_colors, red_probs, fallback_if_empty=True)
        pred_threshold_final = decode_batch_with_threshold(
            pred_chars,
            red_probs,
            threshold=color_threshold,
            fallback_if_empty=True,
        )
        if bool(color_pattern_candidates):
            pred_pattern_final = decode_batch_with_pattern_prior(
                pred_chars,
                red_probs,
                patterns=color_pattern_candidates,
                pattern_log_priors=color_pattern_log_priors,
                prior_weight=pattern_prior_weight,
                char_confidence=char_conf,
                confidence_weight=pattern_confidence_weight,
                fallback_if_empty=True,
            )
        else:
            pred_pattern_final = pred_threshold_final
        calibrated_colors = pattern_colors if use_pattern_prior else threshold_colors
        pred_calibrated_final = pred_pattern_final if use_pattern_prior else pred_threshold_final
        pred_color_oracle_final = decode_batch_final(
            pred_chars,
            color_target,
            fallback_if_empty=False,
        )
        pred_char_oracle_final = decode_batch_final(
            char_target,
            calibrated_colors,
            red_scores=red_probs,
            fallback_if_empty=True,
        )
        target_final = decode_batch_final(char_target, color_target, fallback_if_empty=False)

        pred_chars_rows = pred_chars.detach().cpu().tolist()
        pred_colors_rows = pred_colors.detach().cpu().tolist()
        threshold_colors_rows = threshold_colors.detach().cpu().tolist()
        pattern_colors_rows = pattern_colors.detach().cpu().tolist()
        calibrated_colors_rows = calibrated_colors.detach().cpu().tolist()
        red_prob_rows = red_probs.detach().cpu().tolist()
        char_conf_rows = char_conf.detach().cpu().tolist()
        target_chars_rows = char_target.detach().cpu().tolist()
        target_colors_rows = color_target.detach().cpu().tolist()

        for row_index, filename in enumerate(filenames):
            target_all_label = char_indices_to_string(target_chars_rows[row_index])
            pred_all_label = char_indices_to_string(pred_chars_rows[row_index])
            target_color = color_indices_to_pattern(target_colors_rows[row_index])
            pred_color = color_indices_to_pattern(pred_colors_rows[row_index])
            threshold_color = color_indices_to_pattern(threshold_colors_rows[row_index])
            pattern_color = color_indices_to_pattern(pattern_colors_rows[row_index])
            calibrated_color = color_indices_to_pattern(calibrated_colors_rows[row_index])
            target_red_count = target_color.count("r")
            calibrated_red_count = calibrated_color.count("r")
            row: dict[str, object] = {
                "filename": filename,
                "target_all_label": target_all_label,
                "target_color": target_color,
                "target_red_count": target_red_count,
                "target_label": target_final[row_index],
                "pred_all_label": pred_all_label,
                "pred_color_argmax": pred_color,
                "pred_color_threshold": threshold_color,
                "pred_color_pattern_prior": pattern_color,
                "pred_color_calibrated": calibrated_color,
                "color_decode_method": color_decode_method,
                "pattern_prior_weight": pattern_prior_weight,
                "pattern_confidence_weight": pattern_confidence_weight,
                "pred_red_count_calibrated": calibrated_red_count,
                "pred_label_argmax": pred_final[row_index],
                "pred_label_threshold": pred_threshold_final[row_index],
                "pred_label_pattern_prior": pred_pattern_final[row_index],
                "pred_label_calibrated": pred_calibrated_final[row_index],
                "pred_label_color_oracle": pred_color_oracle_final[row_index],
                "pred_label_char_oracle": pred_char_oracle_final[row_index],
                "argmax_correct": pred_final[row_index] == target_final[row_index],
                "threshold_correct": pred_threshold_final[row_index] == target_final[row_index],
                "pattern_prior_correct": pred_pattern_final[row_index] == target_final[row_index],
                "calibrated_correct": pred_calibrated_final[row_index] == target_final[row_index],
                "color_oracle_correct": pred_color_oracle_final[row_index] == target_final[row_index],
                "char_oracle_correct": pred_char_oracle_final[row_index] == target_final[row_index],
                "calibrated_length_correct": len(pred_calibrated_final[row_index]) == len(target_final[row_index]),
                "char_all_correct": pred_all_label == target_all_label,
                "color_argmax_correct": pred_color == target_color,
                "color_threshold_correct": threshold_color == target_color,
                "color_pattern_prior_correct": pattern_color == target_color,
                "color_calibrated_correct": calibrated_color == target_color,
            }
            for slot in range(5):
                row[f"red_prob_{slot + 1}"] = round(float(red_prob_rows[row_index][slot]), 6)
                row[f"char_conf_{slot + 1}"] = round(float(char_conf_rows[row_index][slot]), 6)
            rows.append(row)

    predictions_path = output_dir / f"{split_name}_predictions.csv"
    errors_path = output_dir / f"{split_name}_errors.csv"
    predictions = pd.DataFrame(rows)
    errors = predictions.loc[~predictions["calibrated_correct"]].copy()
    if max_error_samples >= 0:
        errors = errors.head(max_error_samples)
    predictions.to_csv(predictions_path, index=False)
    errors.to_csv(errors_path, index=False)
    print(f"Saved {split_name} predictions to {predictions_path}")
    print(f"Saved {split_name} errors to {errors_path}")
    return predictions_path, errors_path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_model(train_df: pd.DataFrame, config: Optional[TrainConfig] = None) -> BaselineCNN:
    config = config or TrainConfig()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_image_dir = config.data_dir / "train" / "images"

    if config.debug_overfit:
        work_df = train_df.sort_values("filename").head(config.debug_samples).reset_index(drop=True)
        train_split = work_df
        val_split = work_df.copy()
        eval_name = "debug_train"
    else:
        train_split, val_split = split_train_val(train_df, val_ratio=config.val_ratio, seed=config.seed)
        eval_name = "val"

    train_augment = config.use_augmentation and not config.debug_overfit
    balanced_sampler_enabled = config.use_balanced_sampler and not config.debug_overfit
    pattern_prior_enabled = config.use_pattern_prior and not config.debug_overfit
    pattern_candidates: list[str] = []
    pattern_log_priors: list[float] = []
    if pattern_prior_enabled:
        pattern_candidates, pattern_log_priors = build_color_pattern_prior(train_split)
    char_class_weights = None
    if config.use_char_class_weight and not config.debug_overfit:
        char_class_weights = compute_char_class_weights(
            train_split,
            device=device,
            max_weight=config.max_char_class_weight,
        )
    color_class_weights = None
    if config.use_color_class_weight and not config.debug_overfit:
        color_class_weights = compute_color_class_weights(
            train_split,
            device=device,
            max_weight=config.max_color_class_weight,
        )
    input_mean, input_std, tta_fill_value = resolve_input_normalization(config, train_split, train_image_dir)
    train_dataset = RedCharacterTrainDataset(
        train_split,
        train_image_dir,
        config.image_size,
        augment=train_augment,
        normalize_mean=input_mean,
        normalize_std=input_std,
    )
    val_dataset = RedCharacterTrainDataset(
        val_split,
        train_image_dir,
        config.image_size,
        normalize_mean=input_mean,
        normalize_std=input_std,
    )
    train_sampler = None
    sampler_pattern_counts: dict[str, int] = {}
    if balanced_sampler_enabled:
        train_sampler, sampler_pattern_counts = build_color_pattern_sampler(train_split, seed=config.seed)
    train_loader = build_loader(
        train_dataset,
        config.batch_size,
        shuffle=train_sampler is None,
        seed=config.seed,
        num_workers=config.num_workers,
        device=device,
        sampler=train_sampler,
    )
    val_loader = build_loader(
        val_dataset, config.batch_size, shuffle=False, seed=config.seed, num_workers=config.num_workers, device=device
    )

    model_dropout = 0.0 if config.debug_overfit else config.dropout
    train_label_smoothing = 0.0 if config.debug_overfit else config.label_smoothing
    train_scheduler_enabled = config.use_scheduler and not config.debug_overfit
    eval_tta_shifts = config.tta_shifts if config.use_tta and not config.debug_overfit else (0,)
    eval_tta_scales = config.tta_scales if config.use_tta and not config.debug_overfit else (1.0,)

    model = BaselineCNN(
        num_chars=len(CHARSET),
        feature_dim=config.feature_dim,
        dropout=model_dropout,
        head_hidden_dim=config.head_hidden_dim,
        position_specific_heads=config.position_specific_heads,
        slot_pooling=config.slot_pooling,
    ).to(device)
    model.image_size = config.image_size
    model.input_mean = float(input_mean)
    model.input_std = float(input_std)
    model.tta_fill_value = float(tta_fill_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (
        build_scheduler(optimizer, epochs=config.epochs, warmup_epochs=config.warmup_epochs)
        if train_scheduler_enabled
        else None
    )
    use_amp = config.use_amp and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)
    ema_enabled = config.use_ema and not config.debug_overfit
    ema = ModelEMA(model, decay=config.ema_decay) if ema_enabled else None
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.checkpoint_dir / "baseline_best.pt"
    best_score = -1.0
    best_loss = float("inf")
    best_metrics: dict[str, object] = {}
    history: list[dict[str, object]] = []

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | {eval_name} samples: {len(val_dataset)}")
    print(
        f"Input normalization: {config.normalization} "
        f"mean={input_mean:.4f} std={input_std:.4f} white_fill={tta_fill_value:.4f}"
    )
    print(f"Train augmentation: {'on' if train_augment else 'off'} | AMP: {'on' if use_amp else 'off'}")
    print(
        f"Dropout: {model_dropout:.3f} | label_smoothing: {train_label_smoothing:.3f} "
        f"| scheduler: {'warmup+cosine' if scheduler is not None else 'off'}"
        + (f" warmup_epochs={config.warmup_epochs}" if scheduler is not None else "")
    )
    if color_class_weights is None:
        print("Color class weights: off")
    else:
        color_weights = color_class_weights.detach().cpu()
        red_weights = ",".join(f"{value:.3f}" for value in color_weights[:, 1].tolist())
        print(f"Color class weights: per-slot {summarize_weight_tensor(color_class_weights)} r_by_slot={red_weights}")
    if char_class_weights is None:
        print("Char class weights: off")
    else:
        print(f"Char class weights: per-slot {summarize_weight_tensor(char_class_weights)}")
    print(f"Slot pooling: {config.slot_pooling}")
    print(f"EMA: {'on' if ema is not None else 'off'}" + (f" decay={config.ema_decay:.5f}" if ema else ""))
    print(f"TTA shifts: {','.join(str(item) for item in eval_tta_shifts)}")
    print(f"TTA scales: {','.join(f'{item:g}' for item in eval_tta_scales)}")
    if train_sampler is None:
        print("Balanced sampler: off")
    else:
        print(f"Balanced sampler: on color_patterns={format_count_summary(sampler_pattern_counts)}")
    if pattern_prior_enabled:
        print(
            "Color pattern prior: "
            f"on candidates={len(pattern_candidates)} "
            f"weights={','.join(f'{weight:g}' for weight in config.pattern_prior_weights)} "
            f"confidence_weights={','.join(f'{weight:g}' for weight in config.pattern_confidence_weights)} "
            f"top={format_pattern_summary(pattern_candidates, pattern_log_priors)}"
        )
    else:
        print("Color pattern prior: off")
    print(f"Model parameters: {count_parameters(model):,}")

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
            label_smoothing=train_label_smoothing,
            char_loss_weight=config.char_loss_weight,
            color_loss_weight=config.color_loss_weight,
            max_grad_norm=config.max_grad_norm,
            ema=ema,
            char_class_weights=char_class_weights,
            color_class_weights=color_class_weights,
        )
        raw_eval_metrics = evaluate(
            model,
            val_loader,
            device,
            threshold_min=config.threshold_min,
            threshold_max=config.threshold_max,
            threshold_steps=config.threshold_steps,
            tta_shifts=eval_tta_shifts,
            tta_scales=eval_tta_scales,
            tta_fill_value=tta_fill_value,
            pattern_candidates=pattern_candidates,
            pattern_log_priors=pattern_log_priors,
            pattern_prior_weights=config.pattern_prior_weights,
            pattern_confidence_weights=config.pattern_confidence_weights,
        )
        eval_metrics = raw_eval_metrics
        eval_source = "raw"
        ema_eval_metrics: Optional[dict[str, float]] = None
        if ema is not None:
            with ema.apply_to(model):
                ema_eval_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    threshold_min=config.threshold_min,
                    threshold_max=config.threshold_max,
                    threshold_steps=config.threshold_steps,
                    tta_shifts=eval_tta_shifts,
                    tta_scales=eval_tta_scales,
                    tta_fill_value=tta_fill_value,
                    pattern_candidates=pattern_candidates,
                    pattern_log_priors=pattern_log_priors,
                    pattern_prior_weights=config.pattern_prior_weights,
                    pattern_confidence_weights=config.pattern_confidence_weights,
                )
            if eval_metric_is_better(ema_eval_metrics, raw_eval_metrics):
                eval_metrics = ema_eval_metrics
                eval_source = "ema"
        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"lr={current_lr:.2e} "
            f"selected={eval_source} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"{eval_name}_loss={eval_metrics['loss']:.4f} "
            f"final_exact_acc={eval_metrics['final_exact_acc']:.4f} "
            f"threshold_final_exact_acc={eval_metrics['threshold_final_exact_acc']:.4f} "
            f"calibrated_final_exact_acc={eval_metrics['calibrated_final_exact_acc']:.4f} "
            f"decode={eval_metrics['color_decode_method']} "
            f"color_thresholds={format_thresholds(eval_metrics['color_thresholds'])} "
            f"pattern_final_exact_acc={eval_metrics['pattern_final_exact_acc']:.4f} "
            f"pattern_prior_weight={eval_metrics['pattern_prior_weight']:.2f} "
            f"pattern_confidence_weight={eval_metrics['pattern_confidence_weight']:.2f} "
            f"char_slot_acc={eval_metrics['char_slot_acc']:.4f} "
            f"char_sequence_acc={eval_metrics['char_sequence_acc']:.4f} "
            f"color_slot_acc={eval_metrics['color_slot_acc']:.4f} "
            f"color_pattern_acc={eval_metrics['color_pattern_acc']:.4f} "
            f"char_oracle_final_exact_acc={eval_metrics['char_oracle_final_exact_acc']:.4f} "
            f"color_oracle_final_exact_acc={eval_metrics['color_oracle_final_exact_acc']:.4f} "
            f"calibrated_color_pattern_acc={eval_metrics['calibrated_color_pattern_acc']:.4f} "
            f"calibrated_length_acc={eval_metrics['calibrated_length_acc']:.4f} "
            f"calibrated_gain={eval_metrics['calibrated_gain']:.4f}"
            + (
                f" raw_calibrated_final_exact_acc={raw_eval_metrics['calibrated_final_exact_acc']:.4f} "
                f"ema_calibrated_final_exact_acc={ema_eval_metrics['calibrated_final_exact_acc']:.4f}"
                if ema_eval_metrics is not None
                else ""
            )
        )
        history_row: dict[str, object] = {
            "epoch": epoch,
            "lr": current_lr,
            "selected_model": eval_source,
            "input_normalization": config.normalization,
            "input_mean": input_mean,
            "input_std": input_std,
            "tta_fill_value": tta_fill_value,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"{eval_name}_{key}": value for key, value in eval_metrics.items()},
            f"{eval_name}_raw_calibrated_final_exact_acc": raw_eval_metrics["calibrated_final_exact_acc"],
        }
        if ema_eval_metrics is not None:
            history_row[f"{eval_name}_ema_calibrated_final_exact_acc"] = ema_eval_metrics[
                "calibrated_final_exact_acc"
            ]
        history.append(history_row)
        selection_score = float(eval_metrics["calibrated_final_exact_acc"])
        is_better = selection_score > best_score or (
            selection_score == best_score and float(eval_metrics["loss"]) < best_loss
        )
        if is_better:
            best_score = selection_score
            best_loss = eval_metrics["loss"]
            best_metrics = eval_metrics
            best_state_dict = ema.state_dict() if eval_source == "ema" and ema is not None else model.state_dict()
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "charset": CHARSET,
                    "image_size": config.image_size,
                    "model_config": {
                        "feature_dim": config.feature_dim,
                        "head_hidden_dim": config.head_hidden_dim,
                        "num_chars": len(CHARSET),
                        "position_specific_heads": config.position_specific_heads,
                        "slot_pooling": config.slot_pooling,
                    },
                    "metrics": best_metrics,
                    "color_threshold": eval_metrics["color_threshold"],
                    "color_thresholds": eval_metrics["color_thresholds"],
                    "input_normalization": config.normalization,
                    "input_mean": input_mean,
                    "input_std": input_std,
                    "tta_fill_value": tta_fill_value,
                    "color_decode_method": eval_metrics["color_decode_method"],
                    "color_pattern_candidates": pattern_candidates,
                    "color_pattern_log_priors": pattern_log_priors,
                    "pattern_prior_weight": eval_metrics["pattern_prior_weight"],
                    "pattern_confidence_weight": eval_metrics["pattern_confidence_weight"],
                    "model_source": eval_source,
                    "ema_decay": config.ema_decay if ema is not None else None,
                    "scheduler": "warmup+cosine" if scheduler is not None else None,
                    "warmup_epochs": config.warmup_epochs if scheduler is not None else 0,
                    "tta_shifts": list(eval_tta_shifts),
                    "tta_scales": list(eval_tta_scales),
                    "balanced_sampler": balanced_sampler_enabled,
                    "char_class_weights": (
                        char_class_weights.detach().cpu().tolist()
                        if char_class_weights is not None
                        else None
                    ),
                    "color_class_weights": (
                        color_class_weights.detach().cpu().tolist()
                        if color_class_weights is not None
                        else None
                    ),
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved best {eval_source} checkpoint to {best_path}")
        if scheduler is not None:
            scheduler.step()

    history_path = config.output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.best_checkpoint_path = best_path
    model.best_metrics = checkpoint.get("metrics", {})
    model.color_threshold = float(checkpoint.get("color_threshold", model.best_metrics.get("color_threshold", 0.5)))
    model.color_thresholds = tuple(
        checkpoint.get(
            "color_thresholds",
            model.best_metrics.get("color_thresholds", threshold_to_list(model.color_threshold)),
        )
    )
    model.color_decode_method = str(checkpoint.get("color_decode_method", model.best_metrics.get("color_decode_method", "threshold")))
    model.color_pattern_candidates = tuple(checkpoint.get("color_pattern_candidates", ()))
    model.color_pattern_log_priors = tuple(float(item) for item in checkpoint.get("color_pattern_log_priors", ()))
    model.pattern_prior_weight = float(checkpoint.get("pattern_prior_weight", model.best_metrics.get("pattern_prior_weight", 0.0)))
    model.pattern_confidence_weight = float(
        checkpoint.get("pattern_confidence_weight", model.best_metrics.get("pattern_confidence_weight", 0.0))
    )
    model.input_normalization = str(checkpoint.get("input_normalization", config.normalization))
    model.input_mean = float(checkpoint.get("input_mean", input_mean))
    model.input_std = float(checkpoint.get("input_std", input_std))
    model.tta_fill_value = float(checkpoint.get("tta_fill_value", tta_fill_value))
    model.model_source = str(checkpoint.get("model_source", "raw"))
    model.tta_shifts = tuple(checkpoint.get("tta_shifts", eval_tta_shifts))
    model.tta_scales = tuple(checkpoint.get("tta_scales", eval_tta_scales))
    if config.save_val_diagnostics:
        save_validation_diagnostics(
            model=model,
            loader=val_loader,
            device=device,
            output_dir=config.output_dir,
            split_name=eval_name,
            color_threshold=model.color_thresholds,
            max_error_samples=config.max_error_samples,
            tta_shifts=model.tta_shifts,
            tta_scales=model.tta_scales,
            tta_fill_value=model.tta_fill_value,
            color_decode_method=model.color_decode_method,
            color_pattern_candidates=model.color_pattern_candidates,
            color_pattern_log_priors=model.color_pattern_log_priors,
            pattern_prior_weight=model.pattern_prior_weight,
            pattern_confidence_weight=model.pattern_confidence_weight,
        )
    return model


@torch.no_grad()
def predict_test(model: BaselineCNN, test_df: pd.DataFrame, config: Optional[TrainConfig] = None) -> list[str]:
    config = config or TrainConfig()
    device = next(model.parameters()).device
    image_size = getattr(model, "image_size", config.image_size)
    has_input_normalization = hasattr(model, "input_mean") and hasattr(model, "input_std")
    input_mean = float(getattr(model, "input_mean", 0.5))
    input_std = float(getattr(model, "input_std", 0.5))
    input_normalization = str(getattr(model, "input_normalization", "fixed" if not has_input_normalization else config.normalization))
    test_dataset = RedCharacterTestDataset(
        test_df,
        config.data_dir / "test" / "images",
        image_size,
        normalize_mean=input_mean,
        normalize_std=input_std,
    )
    test_loader = build_loader(
        test_dataset, config.batch_size, shuffle=False, seed=config.seed, num_workers=config.num_workers, device=device
    )
    model.eval()
    predictions: list[str] = []
    color_threshold = tuple(
        getattr(
            model,
            "color_thresholds",
            threshold_to_list(float(getattr(model, "color_threshold", 0.5))),
        )
    )
    color_decode_method = str(getattr(model, "color_decode_method", "threshold"))
    color_pattern_candidates = tuple(getattr(model, "color_pattern_candidates", ()))
    color_pattern_log_priors = tuple(getattr(model, "color_pattern_log_priors", ()))
    pattern_prior_weight = float(getattr(model, "pattern_prior_weight", 0.0))
    pattern_confidence_weight = float(getattr(model, "pattern_confidence_weight", 0.0))
    tta_shifts = tuple(getattr(model, "tta_shifts", config.tta_shifts if config.use_tta else (0,)))
    tta_scales = tuple(getattr(model, "tta_scales", config.tta_scales if config.use_tta else (1.0,)))
    tta_fill_value = float(getattr(model, "tta_fill_value", (1.0 - input_mean) / max(input_std, 1e-6)))
    use_pattern_prior = color_decode_method in {"pattern_prior", "pattern_confidence"} and bool(color_pattern_candidates)
    if use_pattern_prior:
        print(
            "Using color pattern prior for test decoding "
            f"(method={color_decode_method}, candidates={len(color_pattern_candidates)}, "
            f"prior_weight={pattern_prior_weight:.2f}, confidence_weight={pattern_confidence_weight:.2f})"
        )
    else:
        print(f"Using color thresholds {format_thresholds(color_threshold)} for test decoding")
    print(
        f"Using input normalization {input_normalization} "
        f"(mean={input_mean:.4f}, std={input_std:.4f}) for test decoding"
    )
    print(f"Using TTA shifts {','.join(str(item) for item in tta_shifts)} for test decoding")
    print(f"Using TTA scales {','.join(f'{item:g}' for item in tta_scales)} for test decoding")
    for batch in test_loader:
        images = batch["image"].to(device)
        char_logits, color_logits = forward_with_tta(
            model,
            images,
            tta_shifts,
            tta_scales,
            fill_value=tta_fill_value,
        )
        pred_chars = char_logits.argmax(dim=-1)
        red_probs = torch.softmax(color_logits, dim=-1)[..., 1]
        char_confidence = torch.softmax(char_logits, dim=-1).max(dim=-1).values
        if use_pattern_prior:
            predictions.extend(
                decode_batch_with_pattern_prior(
                    pred_chars,
                    red_probs,
                    patterns=color_pattern_candidates,
                    pattern_log_priors=color_pattern_log_priors,
                    prior_weight=pattern_prior_weight,
                    char_confidence=char_confidence,
                    confidence_weight=pattern_confidence_weight,
                    fallback_if_empty=True,
                )
            )
        else:
            predictions.extend(
                decode_batch_with_threshold(
                    pred_chars,
                    red_probs,
                    threshold=color_threshold,
                    fallback_if_empty=True,
                )
            )
    return predictions


def save_submission(
    test_df: pd.DataFrame,
    predictions: list[str],
    config: Optional[TrainConfig] = None,
) -> Path:
    config = config or TrainConfig()
    if len(test_df) != len(predictions):
        raise ValueError(f"prediction count {len(predictions)} does not match test rows {len(test_df)}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = config.output_dir / "submission.csv"
    submission = pd.DataFrame({"id": test_df["id"].astype(str), "label": predictions})
    validate_submission_frame(submission, expected_rows=config.expected_test_rows)
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    return submission_path


def main():
    config = parse_args()
    train_df = load_train_data(config.data_dir)
    model = train_model(train_df, config)
    if config.skip_test:
        print("Skipped test inference by request.")
        return
    test_df = load_test_data(config.data_dir)
    predictions = predict_test(model, test_df, config)
    save_submission(test_df, predictions, config)


if __name__ == "__main__":
    main()
