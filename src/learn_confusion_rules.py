import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Rule:
    source: str
    target: str
    threshold: float
    slot: int | None
    pattern: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn character confusion rules from prediction CSV files.")
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-gain", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=80)
    parser.add_argument("--threshold-margin", type=float, default=0.005)
    parser.add_argument(
        "--variants",
        type=str,
        default="slot_pattern,slot,global",
        help="Comma-separated candidate variants: slot_pattern,slot,global.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    required = {
        "filename",
        "target_all_label",
        "target_label",
        "pred_all_label_calibrated",
        "pred_color_calibrated",
        "pred_label_calibrated",
    }
    required.update({f"char_conf_{idx}" for idx in range(1, 6)})
    required.update({f"red_prob_{idx}" for idx in range(1, 6)})
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df.reset_index(drop=True)


def decode(chars: str, pattern: str, red_probs: Iterable[float]) -> str:
    selected = [char for char, color in zip(chars, pattern) if color == "r"]
    if selected:
        return "".join(selected)
    probs = list(red_probs)
    best_slot = max(range(5), key=lambda idx: probs[idx])
    return chars[best_slot]


def row_red_probs(row: pd.Series) -> list[float]:
    return [float(row[f"red_prob_{idx}"]) for idx in range(1, 6)]


def apply_rules_to_chars(row: pd.Series, rules: list[Rule], char_col: str = "pred_all_label_calibrated") -> str:
    chars = list(str(row[char_col]))
    pattern = str(row["pred_color_calibrated"]).lower()
    for rule in rules:
        if rule.pattern is not None and pattern != rule.pattern:
            continue
        slots = [rule.slot] if rule.slot is not None else range(5)
        for slot in slots:
            if chars[slot] != rule.source:
                continue
            if float(row[f"char_conf_{slot + 1}"]) <= rule.threshold:
                chars[slot] = rule.target
    return "".join(chars)


def apply_single_rule_to_chars(row: pd.Series, rule: Rule, char_col: str = "pred_all_label_calibrated") -> str:
    chars = list(str(row[char_col]))
    pattern = str(row["pred_color_calibrated"]).lower()
    if rule.pattern is not None and pattern != rule.pattern:
        return "".join(chars)
    slots = [rule.slot] if rule.slot is not None else range(5)
    for slot in slots:
        if chars[slot] != rule.source:
            continue
        if float(row[f"char_conf_{slot + 1}"]) <= rule.threshold:
            chars[slot] = rule.target
    return "".join(chars)


def evaluate_rules(df: pd.DataFrame, rules: list[Rule]) -> dict[str, float | int]:
    correct = 0
    base_correct = 0
    fixed = 0
    broken = 0
    changed = 0
    for _, row in df.iterrows():
        base_label = str(row["pred_label_calibrated"])
        target = str(row["target_label"])
        base_ok = base_label == target
        base_correct += int(base_ok)
        adjusted_chars = apply_rules_to_chars(row, rules)
        pattern = str(row["pred_color_calibrated"]).lower()
        adjusted_label = decode(adjusted_chars, pattern, row_red_probs(row))
        adjusted_ok = adjusted_label == target
        correct += int(adjusted_ok)
        changed += int(adjusted_label != base_label)
        fixed += int((not base_ok) and adjusted_ok)
        broken += int(base_ok and (not adjusted_ok))
    total = len(df)
    return {
        "rows": total,
        "base_acc": base_correct / total,
        "rule_acc": correct / total,
        "base_errors": total - base_correct,
        "rule_errors": total - correct,
        "fixed": fixed,
        "broken": broken,
        "changed": changed,
        "net_gain": fixed - broken,
    }


def rule_key_variants(
    source: str,
    target: str,
    slot: int,
    pattern: str,
    variants: set[str],
) -> list[tuple[str, str, int | None, str | None]]:
    keys: list[tuple[str, str, int | None, str | None]] = []
    if "slot_pattern" in variants:
        keys.append((source, target, slot, pattern))
    if "slot" in variants:
        keys.append((source, target, slot, None))
    if "global" in variants:
        keys.append((source, target, None, None))
    return keys


def generate_candidates(
    df: pd.DataFrame,
    variants: set[str],
    threshold_margin: float,
) -> list[Rule]:
    groups: dict[tuple[str, str, int | None, str | None], list[float]] = {}
    for _, row in df.iterrows():
        pred_chars = str(row["pred_all_label_calibrated"])
        target_chars = str(row["target_all_label"])
        pattern = str(row["pred_color_calibrated"]).lower()
        if len(pred_chars) != 5 or len(target_chars) != 5 or len(pattern) != 5:
            continue
        for slot, (pred_char, target_char) in enumerate(zip(pred_chars, target_chars)):
            if pred_char == target_char:
                continue
            if pred_char not in CHARSET or target_char not in CHARSET:
                continue
            conf = float(row[f"char_conf_{slot + 1}"])
            for key in rule_key_variants(pred_char, target_char, slot, pattern, variants):
                groups.setdefault(key, []).append(conf)

    candidates: list[Rule] = []
    for (source, target, slot, pattern), confidences in groups.items():
        thresholds = sorted(set(round(min(0.999, conf + threshold_margin), 3) for conf in confidences))
        for threshold in thresholds:
            candidates.append(Rule(source, target, threshold, slot, pattern))
    return candidates


def score_candidate(df: pd.DataFrame, candidate: Rule) -> dict[str, int | float]:
    fixed = 0
    broken = 0
    changed = 0
    for _, row in df.iterrows():
        base_label = str(row["pred_label_calibrated"])
        target = str(row["target_label"])
        adjusted_chars = apply_single_rule_to_chars(row, candidate)
        pattern = str(row["pred_color_calibrated"]).lower()
        adjusted_label = decode(adjusted_chars, pattern, row_red_probs(row))
        if adjusted_label == base_label:
            continue
        changed += 1
        fixed += int(base_label != target and adjusted_label == target)
        broken += int(base_label == target and adjusted_label != target)
    return {
        "gain": fixed - broken,
        "fixed_delta": fixed,
        "broken_delta": broken,
        "changed_delta": changed,
    }


def learn_rules(
    train_df: pd.DataFrame,
    candidates: list[Rule],
    min_gain: int,
    max_rules: int,
) -> tuple[list[Rule], list[dict[str, object]]]:
    scored: list[tuple[Rule, dict[str, int | float]]] = []
    for candidate in candidates:
        score = score_candidate(train_df, candidate)
        if int(score["gain"]) >= min_gain:
            scored.append((candidate, score))
    scored.sort(
        key=lambda item: (
            int(item[1]["gain"]),
            int(item[1]["fixed_delta"]),
            -int(item[1]["broken_delta"]),
            -int(item[1]["changed_delta"]),
        ),
        reverse=True,
    )
    selected = []
    trace = []
    used_keys: set[tuple[str, int | None, str | None]] = set()
    for rule, score in scored:
        if len(selected) >= max_rules:
            break
        key = (rule.source, rule.slot, rule.pattern)
        if key in used_keys:
            continue
        selected.append(rule)
        used_keys.add(key)
        trace.append({"step": len(selected), **rule_to_dict(rule), **score})
    return selected, trace


def rule_to_dict(rule: Rule) -> dict[str, object]:
    return {
        "source": rule.source,
        "target": rule.target,
        "threshold": rule.threshold,
        "slot": "" if rule.slot is None else rule.slot,
        "pattern": "" if rule.pattern is None else rule.pattern,
    }


def format_rule_tuple(rule: Rule) -> str:
    slot = "None" if rule.slot is None else str(rule.slot)
    if rule.pattern is None:
        return f'("{rule.source}", "{rule.target}", {rule.threshold:.3f}, {slot}),'
    return f'("{rule.source}", "{rule.target}", {rule.threshold:.3f}, {slot}, "{rule.pattern}"),'


def main() -> None:
    args = parse_args()
    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    train_df = read_predictions(args.train_predictions)
    val_df = read_predictions(args.val_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(train_df, variants=variants, threshold_margin=args.threshold_margin)
    rules, trace = learn_rules(
        train_df=train_df,
        candidates=candidates,
        min_gain=args.min_gain,
        max_rules=args.max_rules,
    )
    train_metrics = evaluate_rules(train_df, rules)
    val_metrics = evaluate_rules(val_df, rules)

    pd.DataFrame([train_metrics | {"split": "train"}, val_metrics | {"split": "val"}]).to_csv(
        args.output_dir / "metrics.csv",
        index=False,
    )
    pd.DataFrame([rule_to_dict(rule) for rule in rules]).to_csv(args.output_dir / "learned_rules.csv", index=False)
    pd.DataFrame(trace).to_csv(args.output_dir / "selection_trace.csv", index=False)
    with (args.output_dir / "rules_tuple.txt").open("w", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(format_rule_tuple(rule) + "\n")

    print(f"Candidates: {len(candidates)}")
    print(f"Selected rules: {len(rules)}")
    print(
        "Train "
        f"base_acc={train_metrics['base_acc']:.4f} "
        f"rule_acc={train_metrics['rule_acc']:.4f} "
        f"net_gain={train_metrics['net_gain']}"
    )
    print(
        "Val "
        f"base_acc={val_metrics['base_acc']:.4f} "
        f"rule_acc={val_metrics['rule_acc']:.4f} "
        f"net_gain={val_metrics['net_gain']} "
        f"fixed={val_metrics['fixed']} broken={val_metrics['broken']}"
    )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
