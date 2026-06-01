"""
Eğitim öncesi kontrol: dataset, bölme, modeller, GPU, transformers API.

Kaggle'da train.py'den ÖNCE çalıştırın:
    python src/verify_setup.py
"""

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
    MAX_LENGTH,
    MODELS,
    SEEDS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from data_utils import (
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
    print("ÖN KONTROL — Türkçe haber sınıflandırma")
    print("=" * 60)
    all_ok = True

    # 1) GPU
    cuda = torch.cuda.is_available()
    all_ok &= check("CUDA / GPU", cuda, torch.cuda.get_device_name(0) if cuda else "GPU yok — Kaggle Accelerator ayarlayın")
    if not cuda:
        print("\n>>> Settings → Accelerator → GPU T4, Internet ON, kernel restart\n")

    # 2) Transformers TrainingArguments API
    try:
        build_training_arguments(
            output_dir="/tmp/t",
            eval_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=2,
            num_train_epochs=1,
        )
        all_ok &= check("TrainingArguments API", True, "eval_strategy uyumlu")
    except TypeError as e:
        all_ok &= check("TrainingArguments API", False, str(e))

    # 3) Dataset yükleme
    print("\n--- Dataset (Hugging Face) ---")
    try:
        merged = load_and_merge_dataset()
        text_col, label_col = detect_columns(merged)
        encoded, id2label, label2id = encode_labels(merged, label_col)
        splits = stratified_split(encoded, seed=SEEDS[0])
        n = len(encoded)

        all_ok &= check("Dataset indirildi", True, DATASET_NAME)
        all_ok &= check("Metin kolonu", text_col == "content", f"bulunan: {text_col}")
        all_ok &= check("Etiket kolonu", label_col == "category", f"bulunan: {label_col}")
        all_ok &= check("Sınıf sayısı", len(id2label) == 10, f"{len(id2label)} sınıf")

        n_train, n_val, n_test = len(splits["train"]), len(splits["validation"]), len(splits["test"])
        r_train, r_val, r_test = n_train / n, n_val / n, n_test / n
        all_ok &= check(
            "Bölme oranları (~80/10/10)",
            abs(r_train - TRAIN_RATIO) < 0.002
            and abs(r_val - VAL_RATIO) < 0.002
            and abs(r_test - TEST_RATIO) < 0.002,
            f"train={n_train} ({r_train:.1%}) val={n_val} ({r_val:.1%}) test={n_test} ({r_test:.1%})",
        )

        ex = splits["train"][0]
        all_ok &= check(
            "Örnek metin",
            len(str(ex[text_col])) > 10,
            str(ex[text_col])[:80] + "...",
        )
        all_ok &= check(
            "Örnek etiket",
            ex["labels"] in id2label,
            f"{ex['labels']} -> {id2label[ex['labels']]}",
        )

    except Exception as e:
        all_ok = check("Dataset", False, str(e)) and False
        print("\n>>> Internet ON mu? Dataset adı doğru mu?\n")
        return 1

    # 4) Modeller (tokenizer indirme — hızlı test)
    print("\n--- Modeller (tokenizer) ---")
    for model_id in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            enc = tok("Örnek Türkçe haber metni.", truncation=True, max_length=MAX_LENGTH)
            all_ok &= check(f"Tokenizer: {model_id}", "input_ids" in enc)
            del tok
        except Exception as e:
            all_ok &= check(f"Tokenizer: {model_id}", False, str(e))

    # 5) Seed listesi
    all_ok &= check("Seed'ler", SEEDS == [42, 123, 2026], str(SEEDS))

    # 6) Deney sayısı
    n_experiments = len(MODELS) * len(SEEDS)
    print(f"\nFine-tune deneyi: {len(MODELS)} model × {len(SEEDS)} seed = {n_experiments}")
    print(f"Baseline: {len(MODELS)} model")
    print(f"Toplam eğitim süresi: uzun (273K örnek × 3 epoch) — Kaggle oturumunu açık tutun")

    print("\n" + "=" * 60)
    if all_ok:
        print("TÜM KONTROLLER GEÇTİ → python src/train.py çalıştırabilirsiniz")
    else:
        print("BAZI KONTROLLER BAŞARISIZ — train.py'ye geçmeden düzeltin")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
