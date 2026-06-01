from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from transformers import AutoTokenizer

from config import (
    DATASET_NAME,
    FAST_BENCHMARK_MODE,
    MAX_LENGTH,
    MAX_TEST_SAMPLES,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
    MODELS,
    NUM_EPOCHS,
    SEEDS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from data_utils import (
    apply_fast_benchmark_limits,
    detect_columns,
    encode_labels,
    load_and_merge_dataset,
    stratified_split,
)
from hf_compat import build_training_arguments


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "HATA"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=" * 60)
    print("Kurulum kontrolü")
    print("=" * 60)
    all_ok = True

    cuda = torch.cuda.is_available()
    all_ok &= check("CUDA", cuda, torch.cuda.get_device_name(0) if cuda else "GPU yok")

    try:
        build_training_arguments(
            output_dir="/tmp/t",
            eval_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=2,
            num_train_epochs=1,
        )
        all_ok &= check("TrainingArguments", True)
    except TypeError as e:
        all_ok &= check("TrainingArguments", False, str(e))

    print("\n--- Dataset ---")
    try:
        merged = load_and_merge_dataset()
        text_col, label_col = detect_columns(merged)
        encoded, id2label, label2id = encode_labels(merged, label_col)
        splits = stratified_split(encoded, seed=SEEDS[0])
        splits = apply_fast_benchmark_limits(splits, seed=SEEDS[0])
        n = len(encoded)

        all_ok &= check("Dataset", True, DATASET_NAME)
        all_ok &= check("Metin kolonu", text_col == "content", text_col)
        all_ok &= check("Etiket kolonu", label_col == "category", label_col)
        all_ok &= check("Sınıf sayısı", len(id2label) == 10, str(len(id2label)))

        n_train, n_val, n_test = len(splits["train"]), len(splits["validation"]), len(splits["test"])
        if FAST_BENCHMARK_MODE:
            all_ok &= check(
                "Alt küme",
                n_train <= MAX_TRAIN_SAMPLES
                and n_val <= MAX_VAL_SAMPLES
                and n_test <= MAX_TEST_SAMPLES,
                f"train={n_train} val={n_val} test={n_test}",
            )
        else:
            r_train, r_val, r_test = n_train / n, n_val / n, n_test / n
            all_ok &= check(
                "Bölme",
                abs(r_train - TRAIN_RATIO) < 0.002
                and abs(r_val - VAL_RATIO) < 0.002
                and abs(r_test - TEST_RATIO) < 0.002,
                f"train={n_train} val={n_val} test={n_test}",
            )

        ex = splits["train"][0]
        all_ok &= check("Örnek", ex["labels"] in id2label, id2label[ex["labels"]])

    except Exception as e:
        all_ok = check("Dataset", False, str(e)) and False
        return 1

    print("\n--- Modeller ---")
    for model_id in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            enc = tok("Test.", truncation=True, max_length=MAX_LENGTH)
            all_ok &= check(model_id, "input_ids" in enc)
            del tok
        except Exception as e:
            all_ok &= check(model_id, False, str(e))

    all_ok &= check("Seed", len(SEEDS) >= 1, str(SEEDS))
    all_ok &= check("Epoch", NUM_EPOCHS >= 1, str(NUM_EPOCHS))

    print(f"\nDeney: {len(MODELS)} model × {len(SEEDS)} seed")
    if FAST_BENCHMARK_MODE:
        print(f"Fast mod: max_length={MAX_LENGTH}, epoch={NUM_EPOCHS}")

    print("\n" + "=" * 60)
    print("Tamam" if all_ok else "Hata var")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
