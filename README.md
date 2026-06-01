# Türkçe Haber Kategorisi Sınıflandırma

Türkçe haber metinlerini 10 kategoriye ayıran metin sınıflandırma projesi.

## Dataset

| | |
|--|--|
| Kaynak | [interpress_news_category_tr_lite](https://huggingface.co/datasets/yavuzkomecoglu/interpress_news_category_tr_lite) |
| Metin | `content` |
| Etiket | `category` |
| Bölme | %80 / %10 / %10 (stratified) |

## Modeller

- `dbmdz/bert-base-turkish-cased`
- `distilbert-base-multilingual-cased`
- `xlm-roberta-base`

Her model için baseline ve fine-tuning (çoklu seed) uygulanır.

## Kurulum

```bash
git clone <repo-url>
cd bdm-final
pip install -r requirements.txt
```

## Çalıştırma

```bash
python src/verify_setup.py
python src/train.py
python src/evaluate.py
```

### Kaggle

```bash
!git clone <repo-url>
%cd bdm-final
!pip install -q transformers datasets evaluate accelerate scikit-learn matplotlib
!python src/train.py
```

GPU ve Internet açık olmalı.

### Hızlı mod

`src/config.py` içinde `FAST_BENCHMARK_MODE = True` → alt küme, 1 epoch, `max_length=128`.

Tam veri için `FAST_BENCHMARK_MODE = False`.

## Çıktılar

| Dosya | |
|-------|--|
| `outputs/results.csv` | Deney sonuçları |
| `outputs/failed_examples.csv` | Hatalı tahminler |
| `outputs/plots/` | Loss grafikleri |
| `outputs/models/` | Kayıtlı modeller |

## Proje yapısı

```
bdm-final/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── metrics_utils.py
│   ├── io_utils.py
│   ├── hf_compat.py
│   ├── train.py
│   ├── evaluate.py
│   └── verify_setup.py
├── outputs/
├── requirements.txt
└── README.md
```
