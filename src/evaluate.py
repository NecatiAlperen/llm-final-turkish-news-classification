from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BATCH_SIZE, EVAL_RESULTS_CSV, MAX_LENGTH, MODELS, MODELS_DIR, SEEDS
from data_utils import prepare_datasets
from io_utils import ensure_output_dirs, model_short_name, write_results_table
from metrics_utils import metrics_from_predictions
from train import (
    measure_inference_ms_per_sample,
    model_size_on_disk_mb,
    peak_gpu_memory_mb,
    reset_gpu_memory_stats,
    tokenize_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_saved_model(
    model_id: str,
    seed: int,
    splits: dict,
    text_col: str,
    id2label: dict,
) -> dict | None:
    short = model_short_name(model_id)
    model_dir = MODELS_DIR / short / f"seed_{seed}"

    if not model_dir.exists():
        logger.warning("Model bulunamadı: %s", model_dir)
        return None

    logger.info("Değerlendiriliyor: %s seed=%d (%s)", model_id, seed, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenized = tokenize_splits(
        {"test": splits["test"]}, tokenizer, text_col, max_length=MAX_LENGTH
    )["test"]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    reset_gpu_memory_stats()
    model.eval()
    all_preds, all_labels = [], []
    t0 = time.perf_counter()

    loader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    eval_time = time.perf_counter() - t0
    metrics = metrics_from_predictions(np.array(all_labels), np.array(all_preds))
    inference_ms = measure_inference_ms_per_sample(model, tokenized, data_collator)

    return {
        "model": model_id,
        "seed": seed,
        "phase": "eval_reload",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "train_time_s": None,
        "inference_ms_per_sample": inference_ms,
        "model_size_mb": model_size_on_disk_mb(model_dir),
        "gpu_peak_memory_mb": peak_gpu_memory_mb(),
        "eval_wall_time_s": eval_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kayıtlı modelleri test setinde değerlendir")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(EVAL_RESULTS_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    models = [args.model] if args.model else MODELS
    seeds = [args.seed] if args.seed is not None else SEEDS

    splits, text_col, _, id2label, _, _ = prepare_datasets(seed=SEEDS[0])

    rows: list[dict] = []
    for model_id in models:
        for seed in seeds:
            row = evaluate_saved_model(model_id, seed, splits, text_col, id2label)
            if row:
                rows.append(row)
                logger.info(
                    "%s seed=%d | acc=%.4f macro_f1=%.4f",
                    model_id,
                    seed,
                    row["accuracy"],
                    row["macro_f1"],
                )

    if not rows:
        logger.error("Hiç model değerlendirilemedi. Önce train.py çalıştırın.")
        return

    out_path = Path(args.output)
    write_results_table(rows, csv_path=out_path)
    logger.info("Değerlendirme sonuçları: %s", out_path)


if __name__ == "__main__":
    main()
