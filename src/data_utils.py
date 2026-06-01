"""
Veri yükleme, kolon tespiti, etiket kodlama ve stratified bölme.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from config import (
    DATASET_NAME,
    LABEL_COLUMN_CANDIDATES,
    TEST_RATIO,
    TEXT_COLUMN_CANDIDATES,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


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


def load_and_merge_dataset(dataset_name: str = DATASET_NAME) -> Dataset:
    """Hugging Face dataset'ini yükler; train+test birleştirir."""
    logger.info("Dataset yükleniyor: %s", dataset_name)
    raw = load_dataset(dataset_name)

    parts = []
    for split_name in raw.keys():
        parts.append(raw[split_name])
        logger.info("  split '%s': %d örnek", split_name, len(raw[split_name]))

    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    logger.info("Toplam örnek: %d", len(merged))
    return merged


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
