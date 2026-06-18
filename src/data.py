import random
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARSET)}
COLOR_TO_IDX = {"u": 0, "r": 1}
IDX_TO_COLOR = {0: "u", 1: "r"}
NUM_SLOTS = 5

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_TRANSFORM_AFFINE = getattr(getattr(Image, "Transform", Image), "AFFINE")
_LABEL_RE = re.compile(rf"^[{CHARSET}]{{1,{NUM_SLOTS}}}$")


def encode_chars(label: str) -> list[int]:
    label = str(label).strip().upper()
    if len(label) != NUM_SLOTS:
        raise ValueError(f"all_label must have length {NUM_SLOTS}, got {label!r}")
    try:
        return [CHAR_TO_IDX[char] for char in label]
    except KeyError as exc:
        raise ValueError(f"unknown character {exc.args[0]!r} in label {label!r}") from exc


def encode_colors(color: str) -> list[int]:
    color = str(color).strip().lower()
    if len(color) != NUM_SLOTS:
        raise ValueError(f"color must have length {NUM_SLOTS}, got {color!r}")
    try:
        return [COLOR_TO_IDX[item] for item in color]
    except KeyError as exc:
        raise ValueError(f"unknown color marker {exc.args[0]!r} in color {color!r}") from exc


def apply_train_augmentation(image: Image.Image) -> Image.Image:
    if random.random() < 0.75:
        angle = random.uniform(-4.0, 4.0)
        image = image.rotate(angle, resample=_RESAMPLE_BILINEAR, fillcolor=(255, 255, 255))
    if random.random() < 0.75:
        shift_x = random.uniform(-3.0, 3.0)
        shift_y = random.uniform(-2.0, 2.0)
        image = image.transform(
            image.size,
            _TRANSFORM_AFFINE,
            (1.0, 0.0, shift_x, 0.0, 1.0, shift_y),
            resample=_RESAMPLE_BILINEAR,
            fillcolor=(255, 255, 255),
        )
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.15))
    if random.random() < 0.25:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
    return image


def load_image_tensor(image_path: Path, image_size: tuple[int, int], augment: bool = False) -> torch.Tensor:
    height, width = image_size
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if augment:
            image = apply_train_augmentation(image)
        image = image.resize((width, height), _RESAMPLE_BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - 0.5) / 0.5


def decode_final_label(
    char_indices: Sequence[int],
    color_indices: Sequence[int],
    red_scores: Optional[Sequence[float]] = None,
    fallback_if_empty: bool = True,
) -> str:
    red_positions = [idx for idx, color in enumerate(color_indices) if int(color) == 1]
    if not red_positions and fallback_if_empty:
        if red_scores is None:
            red_positions = [0]
        else:
            red_positions = [max(range(NUM_SLOTS), key=lambda idx: float(red_scores[idx]))]
    return "".join(IDX_TO_CHAR[int(char_indices[idx])] for idx in red_positions)


def decode_batch_final(
    char_indices: torch.Tensor,
    color_indices: torch.Tensor,
    red_scores: Optional[torch.Tensor] = None,
    fallback_if_empty: bool = True,
) -> list[str]:
    char_rows = char_indices.detach().cpu().tolist()
    color_rows = color_indices.detach().cpu().tolist()
    score_rows = red_scores.detach().cpu().tolist() if red_scores is not None else [None] * len(char_rows)
    return [
        decode_final_label(chars, colors, scores, fallback_if_empty=fallback_if_empty)
        for chars, colors, scores in zip(char_rows, color_rows, score_rows)
    ]


def color_indices_from_scores(red_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return (red_scores >= threshold).long()


def decode_batch_with_threshold(
    char_indices: torch.Tensor,
    red_scores: torch.Tensor,
    threshold: float,
    fallback_if_empty: bool = True,
) -> list[str]:
    color_indices = color_indices_from_scores(red_scores, threshold)
    return decode_batch_final(
        char_indices,
        color_indices,
        red_scores=red_scores,
        fallback_if_empty=fallback_if_empty,
    )


def validate_train_frame(df: pd.DataFrame) -> None:
    required = {"filename", "color", "all_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"train labels are missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("train labels are empty")


def validate_test_frame(df: pd.DataFrame) -> None:
    if "id" not in df.columns:
        raise ValueError("submission_sample.csv must contain an 'id' column")
    if df.empty:
        raise ValueError("submission_sample.csv is empty")


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    seed: int = 2026,
    stratify_col: str = "color",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(df) < 2:
        raise ValueError("at least two training samples are required for train/val split")

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    if stratify_col in df.columns:
        groups: Iterable[tuple[object, pd.DataFrame]] = df.groupby(stratify_col, sort=True)
        for _, group in groups:
            indices = list(group.index)
            rng.shuffle(indices)
            if len(indices) == 1:
                train_indices.extend(indices)
                continue
            val_count = max(1, round(len(indices) * val_ratio))
            val_count = min(val_count, len(indices) - 1)
            val_indices.extend(indices[:val_count])
            train_indices.extend(indices[val_count:])
    else:
        indices = list(df.index)
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * val_ratio))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

    if not val_indices:
        all_indices = list(df.index)
        rng.shuffle(all_indices)
        val_indices = [all_indices[0]]
        train_indices = all_indices[1:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
    )


class RedCharacterTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, image_size: tuple[int, int], augment: bool = False):
        validate_train_frame(df)
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        filename = str(row["filename"])
        image = load_image_tensor(self.image_dir / filename, self.image_size, augment=self.augment)
        char_target = torch.tensor(encode_chars(str(row["all_label"])), dtype=torch.long)
        color_target = torch.tensor(encode_colors(str(row["color"])), dtype=torch.long)
        return {
            "image": image,
            "char_target": char_target,
            "color_target": color_target,
            "filename": filename,
        }


class RedCharacterTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, image_size: tuple[int, int]):
        validate_test_frame(df)
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        filename = str(self.df.iloc[index]["id"])
        image = load_image_tensor(self.image_dir / filename, self.image_size, augment=False)
        return {"image": image, "filename": filename}


def validate_submission_frame(submission: pd.DataFrame, expected_rows: Optional[int] = 5000) -> None:
    if list(submission.columns) != ["id", "label"]:
        raise ValueError("submission columns must be exactly: id,label")
    if expected_rows is not None and len(submission) != expected_rows:
        raise ValueError(f"submission must contain {expected_rows} rows, got {len(submission)}")
    if submission["id"].duplicated().any():
        raise ValueError("submission id column contains duplicated values")

    labels = submission["label"].astype(str)
    invalid_mask = ~labels.map(lambda item: bool(_LABEL_RE.fullmatch(item)))
    if invalid_mask.any():
        first_bad = labels[invalid_mask].iloc[0]
        raise ValueError(f"submission label contains invalid value: {first_bad!r}")
