import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from data import (
    CHARSET,
    RedCharacterTestDataset,
    RedCharacterTrainDataset,
    split_train_val,
    validate_submission_frame,
)
from main import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    TrainConfig,
    build_char_position_prior,
    build_color_pattern_prior,
    build_loader,
    build_red_count_prior,
    compute_white_fill_value,
    evaluate,
    load_model_from_checkpoint,
    load_test_data,
    load_train_data,
    normalization_for_storage,
    predict_test,
    resolve_device,
    save_submission,
    save_validation_diagnostics,
    set_seed,
    threshold_to_list,
)


class LogitEnsemble(nn.Module):
    def __init__(self, models: Sequence[nn.Module], weights: Sequence[float] | None = None):
        super().__init__()
        if not models:
            raise ValueError("at least one model is required")
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError("weights length must match models length")
        weight_tensor = torch.tensor([float(weight) for weight in weights], dtype=torch.float32)
        if torch.any(weight_tensor < 0):
            raise ValueError("ensemble weights must be non-negative")
        if float(weight_tensor.sum()) <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.models = nn.ModuleList(models)
        self.register_buffer("weights", weight_tensor / weight_tensor.sum())

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        char_sum: torch.Tensor | None = None
        color_sum: torch.Tensor | None = None
        for model, weight in zip(self.models, self.weights):
            char_logits, color_logits = model(images)
            char_sum = char_logits * weight if char_sum is None else char_sum + char_logits * weight
            color_sum = color_logits * weight if color_sum is None else color_sum + color_logits * weight
        assert char_sum is not None and color_sum is not None
        return char_sum, color_sum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and predict with a checkpoint logit ensemble.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "ensemble")
    parser.add_argument("--checkpoint-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=19)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-val-diagnostics", action="store_true")
    parser.add_argument("--expected-test-rows", type=int, default=5000)
    parser.add_argument("--max-error-samples", type=int, default=200)
    return parser.parse_args()


def parse_weights(value: str, count: int) -> list[float]:
    if not value.strip():
        return [1.0] * count
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError(f"expected {count} weights, got {len(weights)}")
    return weights


def require_shared_metadata(models: Sequence[nn.Module]) -> tuple[tuple[int, int], tuple[float, ...], tuple[float, ...]]:
    first = models[0]
    image_size = tuple(getattr(first, "image_size"))
    input_mean = tuple(normalization_for_storage(getattr(first, "input_mean", 0.5)))
    input_std = tuple(normalization_for_storage(getattr(first, "input_std", 0.5)))
    for idx, model in enumerate(models[1:], start=2):
        candidate_size = tuple(getattr(model, "image_size"))
        candidate_mean = tuple(normalization_for_storage(getattr(model, "input_mean", 0.5)))
        candidate_std = tuple(normalization_for_storage(getattr(model, "input_std", 0.5)))
        if candidate_size != image_size:
            raise ValueError(f"model {idx} image_size {candidate_size} does not match {image_size}")
        if candidate_mean != input_mean or candidate_std != input_std:
            raise ValueError(f"model {idx} normalization does not match the first model")
    return image_size, input_mean, input_std


def set_ensemble_metadata(
    ensemble: nn.Module,
    models: Sequence[nn.Module],
    metrics: dict[str, object],
    train_split: pd.DataFrame,
    use_tta: bool,
) -> None:
    first = models[0]
    ensemble.image_size = tuple(getattr(first, "image_size"))
    ensemble.input_normalization = str(getattr(first, "input_normalization", "fixed"))
    ensemble.input_mean = normalization_for_storage(getattr(first, "input_mean", 0.5))
    ensemble.input_std = normalization_for_storage(getattr(first, "input_std", 0.5))
    ensemble.tta_fill_value = normalization_for_storage(
        getattr(first, "tta_fill_value", compute_white_fill_value(ensemble.input_mean, ensemble.input_std))
    )
    ensemble.tta_shifts = tuple(getattr(first, "tta_shifts", (0,))) if use_tta else (0,)
    ensemble.tta_scales = tuple(getattr(first, "tta_scales", (1.0,))) if use_tta else (1.0,)
    ensemble.color_threshold = float(metrics.get("color_threshold", 0.5))
    ensemble.color_thresholds = threshold_to_list(metrics.get("color_thresholds", ensemble.color_threshold))
    ensemble.color_decode_method = str(metrics.get("color_decode_method", "threshold"))
    ensemble.char_decode_method = str(metrics.get("char_decode_method", "argmax"))
    ensemble.char_log_priors = tuple(tuple(float(value) for value in row) for row in build_char_position_prior(train_split))
    ensemble.char_prior_weight = float(metrics.get("char_prior_weight", 0.0))
    ensemble.count_log_priors = tuple(float(value) for value in build_red_count_prior(train_split))
    ensemble.count_prior_weight = float(metrics.get("count_prior_weight", 0.0))
    patterns, pattern_priors = build_color_pattern_prior(train_split)
    ensemble.color_pattern_candidates = tuple(patterns)
    ensemble.color_pattern_log_priors = tuple(float(value) for value in pattern_priors)
    ensemble.pattern_prior_weight = float(metrics.get("pattern_prior_weight", 0.0))
    ensemble.pattern_confidence_weight = float(metrics.get("pattern_confidence_weight", 0.0))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    base_config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        val_ratio=args.val_ratio,
        seed=args.seed,
        expected_test_rows=args.expected_test_rows,
        use_tta=not args.no_tta,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        max_error_samples=args.max_error_samples,
    )
    models = [load_model_from_checkpoint(path, base_config).eval() for path in args.checkpoint_paths]
    image_size, input_mean, input_std = require_shared_metadata(models)
    weights = parse_weights(args.weights, len(models))
    ensemble = LogitEnsemble(models, weights=weights).to(device).eval()

    train_df = load_train_data(args.data_dir)
    train_split, val_split = split_train_val(train_df, val_ratio=args.val_ratio, seed=args.seed)
    val_dataset = RedCharacterTrainDataset(
        val_split,
        args.data_dir / "train" / "images",
        image_size,
        augment=False,
        normalize_mean=input_mean,
        normalize_std=input_std,
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )
    count_priors = build_red_count_prior(train_split)
    patterns, pattern_priors = build_color_pattern_prior(train_split)
    char_priors = build_char_position_prior(train_split)
    first = models[0]
    tta_shifts = tuple(getattr(first, "tta_shifts", (0,))) if not args.no_tta else (0,)
    tta_scales = tuple(getattr(first, "tta_scales", (1.0,))) if not args.no_tta else (1.0,)
    tta_fill_value = normalization_for_storage(
        getattr(first, "tta_fill_value", compute_white_fill_value(input_mean, input_std))
    )
    print(f"Ensemble checkpoints: {len(models)}")
    print(f"Image size: {image_size} | batch_size={args.batch_size}")
    print(f"Weights: {','.join(f'{weight:g}' for weight in weights)}")
    print(f"TTA shifts: {','.join(str(item) for item in tta_shifts)}")
    print(f"TTA scales: {','.join(f'{item:g}' for item in tta_scales)}")
    metrics = evaluate(
        ensemble,
        val_loader,
        device,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        tta_shifts=tta_shifts,
        tta_scales=tta_scales,
        tta_fill_value=tta_fill_value,
        char_log_priors=char_priors,
        char_prior_weights=base_config.char_prior_weights,
        count_log_priors=count_priors,
        count_prior_weights=base_config.count_prior_weights,
        pattern_candidates=patterns,
        pattern_log_priors=pattern_priors,
        pattern_prior_weights=base_config.pattern_prior_weights,
        pattern_confidence_weights=base_config.pattern_confidence_weights,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "ensemble_val_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(
        "Ensemble validation "
        f"calibrated_final_exact_acc={metrics['calibrated_final_exact_acc']:.4f} "
        f"char_slot_acc={metrics['char_slot_acc']:.4f} "
        f"color_slot_acc={metrics['color_slot_acc']:.4f} "
        f"decode={metrics['color_decode_method']} "
        f"thresholds={','.join(f'{v:.3f}' for v in threshold_to_list(metrics['color_thresholds']))}"
    )
    print(f"Saved ensemble metrics to {metrics_path}")

    set_ensemble_metadata(ensemble, models, metrics, train_split, use_tta=not args.no_tta)
    if args.no_val_diagnostics:
        print("Skipped validation diagnostics by request.")
    else:
        save_validation_diagnostics(
            ensemble,
            val_loader,
            device,
            output_dir=args.output_dir,
            split_name="val_ensemble",
            color_threshold=ensemble.color_thresholds,
            max_error_samples=args.max_error_samples,
            tta_shifts=ensemble.tta_shifts,
            tta_scales=ensemble.tta_scales,
            tta_fill_value=ensemble.tta_fill_value,
            color_decode_method=ensemble.color_decode_method,
            char_decode_method=ensemble.char_decode_method,
            char_log_priors=ensemble.char_log_priors,
            char_prior_weight=ensemble.char_prior_weight,
            count_log_priors=ensemble.count_log_priors,
            count_prior_weight=ensemble.count_prior_weight,
            color_pattern_candidates=ensemble.color_pattern_candidates,
            color_pattern_log_priors=ensemble.color_pattern_log_priors,
            pattern_prior_weight=ensemble.pattern_prior_weight,
            pattern_confidence_weight=ensemble.pattern_confidence_weight,
        )
    if args.skip_test:
        print("Skipped test inference by request.")
        return

    test_df = load_test_data(args.data_dir)
    predictions = predict_test(ensemble, test_df, base_config)
    submission = pd.DataFrame({"id": test_df["id"].astype(str), "label": predictions})
    validate_submission_frame(submission, expected_rows=args.expected_test_rows)
    save_submission(test_df, predictions, base_config)


if __name__ == "__main__":
    main()
