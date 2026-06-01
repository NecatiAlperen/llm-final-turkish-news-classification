"""
Çıktı dizinleri, CSV kayıt ve loss eğrileri.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    FAILED_EXAMPLES_CSV,
    OUTPUTS_DIR,
    PLOTS_DIR,
    RESULTS_CSV,
)

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "model",
    "seed",
    "phase",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "train_time_s",
    "inference_ms_per_sample",
    "model_size_mb",
    "gpu_peak_memory_mb",
]


def ensure_output_dirs() -> None:
    """outputs/ alt dizinlerini oluşturur."""
    for path in (OUTPUTS_DIR, OUTPUTS_DIR / "models", PLOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Çıktı dizinleri hazır: %s", OUTPUTS_DIR)


def append_results_row(row: dict[str, Any], csv_path: Path = RESULTS_CSV) -> None:
    """results.csv'ye tek satır ekler (dosya yoksa başlık yazar)."""
    df_new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)


def write_results_table(rows: list[dict[str, Any]], csv_path: Path = RESULTS_CSV) -> None:
    """Tüm sonuç tablosunu yazar."""
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df[RESULT_COLUMNS].to_csv(csv_path, index=False)
    logger.info("Sonuçlar kaydedildi: %s (%d satır)", csv_path, len(df))


def save_failed_examples(
    examples: list[dict[str, Any]],
    csv_path: Path = FAILED_EXAMPLES_CSV,
    min_count: int = 5,
) -> None:
    """Hatalı tahminleri CSV'ye yazar."""
    if len(examples) < min_count:
        logger.warning(
            "Yalnızca %d hatalı örnek bulundu (minimum %d).",
            len(examples),
            min_count,
        )
    df = pd.DataFrame(examples)
    df.to_csv(csv_path, index=False)
    logger.info("Hatalı örnekler: %s (%d satır)", csv_path, len(df))


def plot_loss_curves(
    log_history: list[dict[str, Any]],
    model_short: str,
    seed: int,
    plots_dir: Path = PLOTS_DIR,
) -> Path | None:
    """Trainer log_history'den loss eğrisi kaydeder."""
    train_loss = [
        (e["step"], e["loss"])
        for e in log_history
        if "loss" in e and "eval_loss" not in e
    ]
    eval_loss = [
        (e.get("epoch", e.get("step", 0)), e["eval_loss"])
        for e in log_history
        if "eval_loss" in e
    ]

    if not train_loss and not eval_loss:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    if train_loss:
        steps, losses = zip(*train_loss)
        ax.plot(steps, losses, label="train_loss", alpha=0.8)

    if eval_loss:
        epochs, losses = zip(*eval_loss)
        ax.plot(epochs, losses, label="eval_loss", marker="o")

    ax.set_xlabel("Step / Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_short} — seed {seed}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = plots_dir / f"loss_{model_short}_seed{seed}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Loss eğrisi: %s", out_path)
    return out_path


def model_short_name(model_id: str) -> str:
    """Dosya adları için kısa model adı."""
    return model_id.split("/")[-1].replace("-", "_")
