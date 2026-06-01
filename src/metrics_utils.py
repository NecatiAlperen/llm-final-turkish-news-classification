from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
    }


def metrics_from_predictions(
    labels: np.ndarray, preds: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
    }


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "nan ± nan"
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def parse_mean_std(value: str) -> tuple[float | None, float | None]:
    match = re.match(r"([\d.]+)\s*±\s*([\d.]+)", str(value).strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, None


def aggregate_seed_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = ("accuracy", "macro_f1", "weighted_f1")
    numeric_keys = (
        "train_time_s",
        "inference_ms_per_sample",
        "model_size_mb",
        "gpu_peak_memory_mb",
    )

    summary: dict[str, Any] = {"phase": "fine_tuned_aggregated", "seed": "mean±std"}

    for key in metric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = format_mean_std(vals)
        summary[f"{key}_values"] = vals

    for key in numeric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = format_mean_std(vals)

    return summary
