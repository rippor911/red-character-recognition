import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data import (
    CHARSET,
    RedCharacterTestDataset,
    RedCharacterTrainDataset,
    decode_batch_final,
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
    epochs: int = 5
    learning_rate: float = 1e-3
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


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
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )


def compute_loss(
    char_logits: torch.Tensor,
    color_logits: torch.Tensor,
    char_target: torch.Tensor,
    color_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    char_loss = F.cross_entropy(char_logits.reshape(-1, len(CHARSET)), char_target.reshape(-1))
    color_loss = F.cross_entropy(color_logits.reshape(-1, 2), color_target.reshape(-1))
    loss = char_loss + color_loss
    return loss, char_loss, color_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
        char_logits, color_logits = model(images)
        loss, char_loss, color_loss = compute_loss(char_logits, color_logits, char_target, color_target)
        loss.backward()
        optimizer.step()

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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
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

    for batch in loader:
        images = batch["image"].to(device)
        char_target = batch["char_target"].to(device)
        color_target = batch["color_target"].to(device)
        char_logits, color_logits = model(images)
        loss, char_loss, color_loss = compute_loss(char_logits, color_logits, char_target, color_target)

        pred_chars = char_logits.argmax(dim=-1)
        pred_colors = color_logits.argmax(dim=-1)
        pred_final = decode_batch_final(pred_chars, pred_colors, color_logits[..., 1], fallback_if_empty=True)
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

    return {
        "loss": total_loss / total_samples,
        "char_loss": total_char_loss / total_samples,
        "color_loss": total_color_loss / total_samples,
        "final_exact_acc": final_correct / total_samples,
        "char_slot_acc": char_slot_correct / total_slots,
        "color_slot_acc": color_slot_correct / total_slots,
        "color_pattern_acc": color_pattern_correct / total_samples,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


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

    train_dataset = RedCharacterTrainDataset(train_split, train_image_dir, config.image_size)
    val_dataset = RedCharacterTrainDataset(val_split, train_image_dir, config.image_size)
    train_loader = build_loader(
        train_dataset, config.batch_size, shuffle=True, seed=config.seed, num_workers=config.num_workers, device=device
    )
    val_loader = build_loader(
        val_dataset, config.batch_size, shuffle=False, seed=config.seed, num_workers=config.num_workers, device=device
    )

    model = BaselineCNN(num_chars=len(CHARSET)).to(device)
    model.image_size = config.image_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.checkpoint_dir / "baseline_best.pt"
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | {eval_name} samples: {len(val_dataset)}")
    print(f"Model parameters: {count_parameters(model):,}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"{eval_name}_loss={eval_metrics['loss']:.4f} "
            f"final_exact_acc={eval_metrics['final_exact_acc']:.4f} "
            f"char_slot_acc={eval_metrics['char_slot_acc']:.4f} "
            f"color_slot_acc={eval_metrics['color_slot_acc']:.4f} "
            f"color_pattern_acc={eval_metrics['color_pattern_acc']:.4f}"
        )
        if eval_metrics["final_exact_acc"] > best_score:
            best_score = eval_metrics["final_exact_acc"]
            best_metrics = eval_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "charset": CHARSET,
                    "image_size": config.image_size,
                    "metrics": best_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.best_checkpoint_path = best_path
    model.best_metrics = checkpoint.get("metrics", {})
    return model


@torch.no_grad()
def predict_test(model: BaselineCNN, test_df: pd.DataFrame, config: Optional[TrainConfig] = None) -> list[str]:
    config = config or TrainConfig()
    device = next(model.parameters()).device
    image_size = getattr(model, "image_size", config.image_size)
    test_dataset = RedCharacterTestDataset(test_df, config.data_dir / "test" / "images", image_size)
    test_loader = build_loader(
        test_dataset, config.batch_size, shuffle=False, seed=config.seed, num_workers=config.num_workers, device=device
    )
    model.eval()
    predictions: list[str] = []
    for batch in test_loader:
        images = batch["image"].to(device)
        char_logits, color_logits = model(images)
        pred_chars = char_logits.argmax(dim=-1)
        pred_colors = color_logits.argmax(dim=-1)
        predictions.extend(decode_batch_final(pred_chars, pred_colors, color_logits[..., 1], fallback_if_empty=True))
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
