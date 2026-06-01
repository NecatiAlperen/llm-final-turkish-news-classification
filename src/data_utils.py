"""
Veri yükleme, kolon tespiti, etiket kodlama ve stratified bölme.
"""

from __future__ import annotations

import csv
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from config import (
    CATEGORY_NAMES,
    DATASET_NAME,
    INTERPRESS_CACHE_DIR,
    INTERPRESS_TEST_TSV,
    INTERPRESS_TRAIN_TSV,
    INTERPRESS_ZIP_URL,
    LABEL_COLUMN_CANDIDATES,
    TEST_RATIO,
    TEXT_COLUMN_CANDIDATES,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


def _configure_csv_field_size_limit() -> None:
    """Uzun haber satırları için csv alan boyutu sınırını yükseltir."""
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_configure_csv_field_size_limit()


def detect_columns(dataset: Dataset) -> tuple[str, str]:
    """Metin ve etiket kolonlarını otomatik belirler."""
    columns = set(dataset.column_names)

    text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in columns), None)
    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in columns), None)

    if text_col is None:
        raise ValueError(
            f"Metin kolonu bulunamadı. Mevcut kolonlar: {dataset.column_names}. "
            f"Beklenen adlardan biri: {TEXT_COLUMN_CANDIDATES}"
        )
    if label_col is None:
        raise ValueError(
            f"Etiket kolonu bulunamadı. Mevcut kolonlar: {dataset.column_names}. "
            f"Beklenen adlardan biri: {LABEL_COLUMN_CANDIDATES}"
        )

    logger.info("Kolonlar: text=%s, label=%s", text_col, label_col)
    return text_col, label_col


def _dataset_features() -> Features:
    return Features(
        {
            "content": Value("string"),
            "category": ClassLabel(names=CATEGORY_NAMES),
        }
    )


def _read_interpress_tsv(tsv_path: Path) -> Dataset:
    """Interpress TSV: news (metin) + label (0-9)."""
    _configure_csv_field_size_limit()
    contents: list[str] = []
    categories: list[int] = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            contents.append(row["news"])
            categories.append(int(row["label"]))
    return Dataset.from_dict(
        {"content": contents, "category": categories},
        features=_dataset_features(),
    )


def load_and_merge_from_interpress_zip() -> Dataset:
    """
    Hugging Face script olmadan resmi ZIP/TSV kaynağından yükler.
    datasets>=4 ortamında (Kaggle) gerekli yedek yol.
    """
    cache_dir = INTERPRESS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "interpress_news_category_tr_270k_lite.zip"
    extract_dir = cache_dir / "extracted"

    if not zip_path.exists():
        logger.info("Interpress ZIP indiriliyor: %s", INTERPRESS_ZIP_URL)
        urlretrieve(INTERPRESS_ZIP_URL, zip_path)

    train_tsv = extract_dir / INTERPRESS_TRAIN_TSV
    test_tsv = extract_dir / INTERPRESS_TEST_TSV
    if not train_tsv.exists() or not test_tsv.exists():
        logger.info("ZIP açılıyor: %s", zip_path)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    if not train_tsv.exists():
        # Bazı arşivlerde alt klasör olabilir
        found = list(extract_dir.rglob(INTERPRESS_TRAIN_TSV))
        if not found:
            raise FileNotFoundError(
                f"Train TSV bulunamadı: {INTERPRESS_TRAIN_TSV} — {extract_dir}"
            )
        train_tsv = found[0]
        test_tsv = train_tsv.parent / INTERPRESS_TEST_TSV

    train_ds = _read_interpress_tsv(train_tsv)
    test_ds = _read_interpress_tsv(test_tsv)
    logger.info("  TSV train: %d örnek", len(train_ds))
    logger.info("  TSV test: %d örnek", len(test_ds))

    merged = concatenate_datasets([train_ds, test_ds])
    logger.info("Toplam örnek: %d (Interpress ZIP/TSV)", len(merged))
    return merged


def load_and_merge_dataset(dataset_name: str = DATASET_NAME) -> Dataset:
    """HF dataset veya (script yoksa) Interpress ZIP/TSV; train+test birleştirir."""
    logger.info("Dataset yükleniyor: %s", dataset_name)
    try:
        raw = load_dataset(dataset_name)
        parts = []
        for split_name in raw.keys():
            parts.append(raw[split_name])
            logger.info("  split '%s': %d örnek", split_name, len(raw[split_name]))
        merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        logger.info("Toplam örnek: %d (Hugging Face)", len(merged))
        return merged
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        logger.warning(
            "HF script desteklenmiyor (datasets>=4). Interpress ZIP kullanılıyor."
        )
        return load_and_merge_from_interpress_zip()


def encode_labels(
    dataset: Dataset, label_col: str
) -> tuple[Dataset, dict[int, str], dict[str, int]]:
    """
    Etiketleri tamsayıya çevirir.
    ClassLabel feature varsa names kullanılır.
    """
    feature = dataset.features.get(label_col)
    if feature is not None and hasattr(feature, "names") and feature.names:
        id2label = {i: name for i, name in enumerate(feature.names)}
        label2id = {name: i for i, name in id2label.items()}

        def _map_labels(example: dict[str, Any]) -> dict[str, Any]:
            val = example[label_col]
            if isinstance(val, int):
                example["labels"] = val
            else:
                example["labels"] = label2id[str(val)]
            return example

        dataset = dataset.map(_map_labels)
    else:
        labels = dataset[label_col]
        unique = sorted(set(str(l) for l in labels))
        label2id = {name: i for i, name in enumerate(unique)}
        id2label = {i: name for name, i in label2id.items()}

        def _map_labels(example: dict[str, Any]) -> dict[str, Any]:
            example["labels"] = label2id[str(example[label_col])]
            return example

        dataset = dataset.map(_map_labels)

    num_labels = len(id2label)
    logger.info("%d sınıf: %s", num_labels, list(id2label.values()))
    return dataset, id2label, label2id


def stratified_split(
    dataset: Dataset,
    label_col: str = "labels",
    seed: int = 42,
) -> DatasetDict:
    """%80 train / %10 val / %10 test stratified bölme."""
    labels = np.array(dataset[label_col])
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,
        random_state=seed,
    )
    val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio_of_temp),
        stratify=temp_labels,
        random_state=seed,
    )

    splits = DatasetDict(
        {
            "train": dataset.select(train_idx.tolist()),
            "validation": dataset.select(val_idx.tolist()),
            "test": dataset.select(test_idx.tolist()),
        }
    )
    logger.info(
        "Bölme (seed=%d): train=%d, val=%d, test=%d",
        seed,
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    return splits


def prepare_datasets(seed: int = 42) -> tuple[DatasetDict, str, str, dict, dict, int]:
    """Tam veri hazırlık pipeline'ı."""
    merged = load_and_merge_dataset()
    text_col, label_col = detect_columns(merged)
    encoded, id2label, label2id = encode_labels(merged, label_col)
    splits = stratified_split(encoded, label_col="labels", seed=seed)
    num_labels = len(id2label)
    return splits, text_col, label_col, id2label, label2id, num_labels
