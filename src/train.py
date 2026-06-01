"""
Türkçe haber kategorisi sınıflandırma — fine-tuning ve baseline eğitim scripti.

Kullanım (proje kökünden):
    python src/train.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from hf_compat import build_trainer, build_training_arguments  # noqa: E402

# src/ modül yolu
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    EVALUATION_STRATEGY,
    FP16,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_LENGTH,
    MIN_FAILED_EXAMPLES,
    MODELS,
    MODELS_DIR,
    NUM_EPOCHS,
    SAVE_STRATEGY,
    SEEDS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from data_utils import prepare_datasets  # noqa: E402
from io_utils import (  # noqa: E402
    ensure_output_dirs,
    model_short_name,
    plot_loss_curves,
    save_failed_examples,
    write_results_table,
)
from metrics_utils import aggregate_seed_results, compute_metrics, metrics_from_predictions  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reset_gpu_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return None


def free_gpu_memory(*objects) -> None:
    """Model/trainer arası bellek temizliği (T4 OOM önleme)."""
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_size_mb(model: torch.nn.Module) -> float:
    """Parametre + buffer boyutu (MB)."""
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nbytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return nbytes / (1024**2)


def model_size_on_disk_mb(model_dir: Path) -> float:
    """Kaydedilmiş model dosyalarının toplam boyutu (MB)."""
    if not model_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return total / (1024**2)


def tokenize_splits(
    splits: dict,
    tokenizer,
    text_col: str,
    max_length: int = MAX_LENGTH,
) -> dict:
    def _tok(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = {}
    for name, ds in splits.items():
        remove_cols = [c for c in ds.column_names if c not in ("labels",)]
        tokenized[name] = ds.map(_tok, batched=True, remove_columns=remove_cols)
    return tokenized


def measure_inference_ms_per_sample(
    model,
    dataset: Dataset,
    data_collator,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Test seti üzerinde ms/örnek inference süresi."""
    model.eval()
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    n_samples = len(dataset)
    total_time = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

    return (total_time / n_samples) * 1000.0


def run_baseline(
    model_id: str,
    tokenized_test: Dataset,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> dict:
    """
    Fine-tuning öncesi baseline: rastgele başlatılmış sınıflandırma başlığı
    (eğitimsiz / zero-shot öncesi referans).
    """
    logger.info("=== BASELINE (eğitimsiz başlık): %s ===", model_id)
    set_seed(42)
    device = get_device()
    reset_gpu_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    size_mb = model_size_mb(model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    t0 = time.perf_counter()
    model.eval()
    all_preds, all_labels = [], []

    loader = torch.utils.data.DataLoader(
        tokenized_test,
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
    metrics = metrics_from_predictions(
        np.array(all_labels), np.array(all_preds)
    )
    inference_ms = (eval_time / len(tokenized_test)) * 1000.0
    gpu_mb = peak_gpu_memory_mb()

    logger.info(
        "Baseline %s | acc=%.4f macro_f1=%.4f weighted_f1=%.4f",
        model_id,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    )

    return {
        "model": model_id,
        "seed": "N/A",
        "phase": "baseline",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "train_time_s": 0.0,
        "inference_ms_per_sample": inference_ms,
        "model_size_mb": size_mb,
        "gpu_peak_memory_mb": gpu_mb,
        "_model": model,
        "_tokenizer": tokenizer,
        "_data_collator": data_collator,
    }


def run_finetuning(
    model_id: str,
    seed: int,
    splits: dict,
    text_col: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> dict:
    """Tek model + seed için fine-tuning."""
    short = model_short_name(model_id)
    logger.info("=== FINE-TUNING: %s | seed=%d ===", model_id, seed)

    set_seed(seed)
    device = get_device()
    reset_gpu_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized = tokenize_splits(splits, tokenizer, text_col)
    # Hatalı örnekler için ham metni sakla
    test_with_text = splits["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output_dir = MODELS_DIR / short / f"seed_{seed}" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=False,
        fp16=FP16 and torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    trainer = build_trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_start = time.perf_counter()
    train_result = trainer.train()
    train_time_s = time.perf_counter() - train_start

    plot_loss_curves(trainer.state.log_history, short, seed)

    # Test değerlendirmesi
    test_metrics = trainer.evaluate(tokenized["test"])
    gpu_mb = peak_gpu_memory_mb()

    # Model kaydet
    save_dir = MODELS_DIR / short / f"seed_{seed}"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    disk_mb = model_size_on_disk_mb(save_dir)

    inference_ms = measure_inference_ms_per_sample(
        trainer.model, tokenized["test"], data_collator
    )

    logger.info(
        "%s seed=%d | acc=%.4f macro_f1=%.4f | train=%.1fs infer=%.2f ms/sample",
        model_id,
        seed,
        test_metrics.get("eval_accuracy", 0),
        test_metrics.get("eval_macro_f1", 0),
        train_time_s,
        inference_ms,
    )

    # Hatalı örnekler (ham metinli test seti ile)
    failed = collect_failed_examples_with_text(
        trainer.model,
        tokenized["test"],
        test_with_text,
        text_col,
        data_collator,
        id2label,
        model_id,
        seed,
        "fine_tuned",
    )

    result = {
        "model": model_id,
        "seed": seed,
        "phase": "fine_tuned",
        "accuracy": test_metrics.get("eval_accuracy"),
        "macro_f1": test_metrics.get("eval_macro_f1"),
        "weighted_f1": test_metrics.get("eval_weighted_f1"),
        "train_time_s": train_time_s,
        "inference_ms_per_sample": inference_ms,
        "model_size_mb": disk_mb,
        "gpu_peak_memory_mb": gpu_mb,
        "train_loss": train_result.training_loss,
        "failed_examples": failed,
    }

    free_gpu_memory(trainer, model, tokenizer)
    return result


def collect_failed_examples_with_text(
    model,
    tokenized_test: Dataset,
    raw_test: Dataset,
    text_col: str,
    data_collator,
    id2label: dict[int, str],
    model_id: str,
    seed: int,
    phase: str,
    max_examples: int = 20,
) -> list[dict]:
    """Tokenize edilmiş test + ham metin ile hatalı örnek toplama."""
    model.eval()
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        tokenized_test,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False,
    )
    failed: list[dict] = []
    texts = raw_test[text_col]
    labels = raw_test["labels"]
    idx = 0

    with torch.no_grad():
        for batch in loader:
            batch_labels = batch["labels"].numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            preds = model(**inputs).logits.argmax(dim=-1).cpu().numpy()

            for true_id, pred_id in zip(batch_labels, preds):
                if true_id != pred_id and len(failed) < max_examples:
                    failed.append(
                        {
                            "model": model_id,
                            "seed": seed,
                            "phase": phase,
                            "text": str(texts[idx])[:500],
                            "true_label": id2label[int(true_id)],
                            "predicted_label": id2label[int(pred_id)],
                        }
                    )
                idx += 1

    return failed


def main() -> None:
    ensure_output_dirs()
    device = get_device()
    logger.info("Cihaz: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    all_rows: list[dict] = []
    all_failed: list[dict] = []

    # Veri bir kez yüklenir (bölme seed'i ilk SEEDS değeri ile sabit)
    splits, text_col, _label_col, id2label, label2id, num_labels = prepare_datasets(
        seed=SEEDS[0]
    )

    for model_id in MODELS:
        # --- Baseline (eğitimsiz) ---
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenized_test_only = tokenize_splits(
            {"test": splits["test"]}, tokenizer, text_col
        )["test"]

        baseline_row = run_baseline(
            model_id,
            tokenized_test_only,
            num_labels,
            id2label,
            label2id,
        )
        row_clean = {k: v for k, v in baseline_row.items() if not k.startswith("_")}
        all_rows.append(row_clean)
        free_gpu_memory(
            baseline_row.get("_model"),
            baseline_row.get("_tokenizer"),
            tokenizer,
        )

        # --- Fine-tuning: her seed ---
        seed_rows: list[dict] = []
        for seed in SEEDS:
            try:
                result = run_finetuning(
                    model_id,
                    seed,
                    splits,
                    text_col,
                    num_labels,
                    id2label,
                    label2id,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        "OOM: %s seed=%d — batch_size=8 ile yeniden deneyin (config.py).",
                        model_id,
                        seed,
                    )
                raise

            failed = result.pop("failed_examples", [])
            all_failed.extend(failed)

            row = {k: v for k, v in result.items() if k != "train_loss"}
            all_rows.append(row)
            seed_rows.append(row)

        # Özet satır (ortalama ± std)
        agg = aggregate_seed_results(seed_rows)
        summary_row = {
            "model": model_id,
            "seed": agg["seed"],
            "phase": agg["phase"],
            "accuracy": agg["accuracy"],
            "macro_f1": agg["macro_f1"],
            "weighted_f1": agg["weighted_f1"],
            "train_time_s": agg["train_time_s"],
            "inference_ms_per_sample": agg["inference_ms_per_sample"],
            "model_size_mb": agg["model_size_mb"],
            "gpu_peak_memory_mb": agg["gpu_peak_memory_mb"],
        }
        all_rows.append(summary_row)
        logger.info(
            "Özet %s | accuracy %s | macro_f1 %s",
            model_id,
            agg["accuracy"],
            agg["macro_f1"],
        )

    write_results_table(all_rows)

    # En az MIN_FAILED_EXAMPLES hatalı örnek
    if len(all_failed) < MIN_FAILED_EXAMPLES:
        logger.warning(
            "Toplam %d hatalı örnek; minimum için ek örnek toplanamadı.",
            len(all_failed),
        )
    save_failed_examples(all_failed, min_count=MIN_FAILED_EXAMPLES)

    logger.info("Eğitim tamamlandı. Sonuçlar: outputs/results.csv")


if __name__ == "__main__":
    main()
