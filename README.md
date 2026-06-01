# Türkçe Haber Kategorisi Sınıflandırma

Üniversite dönem projesi final teslimi: Türkçe haber metinlerini 10 kategoriye ayıran metin sınıflandırma sistemi.

## Proje amacı

[Interpress Turkish News Category Dataset (Lite)](https://huggingface.co/datasets/yavuzkomecoglu/interpress_news_category_tr_lite) üzerinde üç farklı önceden eğitilmiş dil modelinin fine-tuning ile karşılaştırılması; her model için eğitim öncesi baseline, çoklu seed tekrarları ve standart sınıflandırma metrikleriyle raporlama.

## Dataset

| Özellik | Değer |
|--------|--------|
| Kaynak | `yavuzkomecoglu/interpress_news_category_tr_lite` |
| Metin kolonu | `content` |
| Etiket kolonu | `category` (10 sınıf) |
| Bölme | %80 train / %10 validation / %10 test (stratified) |

Sınıflar: kültürsanat, ekonomi, siyaset, eğitim, dünya, spor, teknoloji, magazin, sağlık, gündem.

## Kullanılan modeller

1. `dbmdz/bert-base-turkish-cased` — Türkçe BERT
2. `xlm-roberta-base` — çok dilli RoBERTa
3. `distilbert-base-multilingual-cased` — hafif çok dilli DistilBERT

Her model için:

- **Baseline**: Eğitimsiz (rastgele başlatılmış) sınıflandırma başlığı ile test seti sonucu
- **Fine-tuning**: 3 seed (`42`, `123`, `2026`) ile eğitim; sonuçlar **ortalama ± standart sapma**

## Kurulum

```bash
git clone https://github.com/KULLANICI/bdm-final.git
cd bdm-final
pip install -r requirements.txt
```

## Kaggle'da çalıştırma

1. Yeni bir **Notebook** oluşturun; **Accelerator → GPU T4** seçin.
2. **Internet** açık olsun (Hugging Face dataset ve model indirme için).
3. Hücrelerde:

```bash
!git clone https://github.com/KULLANICI/bdm-final.git
%cd bdm-final
!pip install -q -r requirements.txt
```

4. Eğitimi başlatın:

```bash
!python src/train.py
```

5. (İsteğe bağlı) Kayıtlı modelleri yeniden değerlendirin:

```bash
!python src/evaluate.py
```

**Notlar**

- `outputs/` dizini otomatik oluşturulur.
- Varsayılan `batch_size=16`, `max_length=256`, `num_epochs=3`, `learning_rate=2e-5` — T4 için uygundur. OOM alırsanız `src/config.py` içinde `BATCH_SIZE = 8` yapın.
- Model ağırlıkları GitHub'a eklenmez; Kaggle çıktısından veya `outputs/models/` zip'inden indirilebilir.

## Eğitim komutu

Proje kök dizininden:

```bash
python src/train.py
```

Yeniden değerlendirme:

```bash
python src/evaluate.py
# Tek model / seed:
python src/evaluate.py --model dbmdz/bert-base-turkish-cased --seed 42
```

## Çıktı dosyaları

| Dosya | Açıklama |
|-------|----------|
| `outputs/results.csv` | Tüm deney sonuçları (baseline, seed bazlı, özet) |
| `outputs/failed_examples.csv` | En az 5 hatalı tahmin örneği |
| `outputs/plots/loss_*.png` | Epoch/step bazlı loss eğrileri |
| `outputs/models/<model>/seed_<N>/` | Fine-tune edilmiş model + tokenizer |
| `outputs/eval_results.csv` | `evaluate.py` çıktısı |

## Sonuç tablosu formatı (`outputs/results.csv`)

| Kolon | Açıklama |
|-------|----------|
| `model` | Hugging Face model ID |
| `seed` | Seed değeri, `N/A` (baseline) veya `mean±std` (özet) |
| `phase` | `baseline`, `fine_tuned`, `fine_tuned_aggregated`, `eval_reload` |
| `accuracy` | Doğruluk |
| `macro_f1` | Macro-F1 |
| `weighted_f1` | Weighted-F1 |
| `train_time_s` | Eğitim süresi (saniye) |
| `inference_ms_per_sample` | Çıkarım süresi (ms/örnek) |
| `model_size_mb` | Model boyutu (MB) |
| `gpu_peak_memory_mb` | GPU tepe bellek (MB); CPU'da boş olabilir |

Özet satırlarda metrikler `0.8521 ± 0.0043` formatındadır.

## Proje yapısı

```
bdm-final/
├── src/
│   ├── config.py          # Dataset, modeller, hiperparametreler
│   ├── data_utils.py      # Yükleme, bölme, etiket kodlama
│   ├── metrics_utils.py   # Metrikler ve özet istatistik
│   ├── io_utils.py        # CSV, grafik, dizinler
│   ├── train.py           # Ana eğitim scripti
│   └── evaluate.py        # Kayıtlı model değerlendirme
├── outputs/               # Eğitim çıktıları (gitignore)
├── reports/
├── notebooks/
├── requirements.txt
└── README.md
```

## Ödev gereksinimleri karşılığı

- [x] En az 3 model fine-tune
- [x] Model başına baseline (eğitim öncesi)
- [x] %80 / %10 / %10 stratified bölme
- [x] 3 seed tekrarı
- [x] Ortalama ± standart sapma özet satırları
- [x] Accuracy, Macro-F1, Weighted-F1
- [x] Eğitim / inference süresi, model boyutu, GPU bellek, loss eğrileri
- [x] `failed_examples.csv` ve `results.csv`

## Lisans ve atıf

Dataset: [yavuzkomecoglu/interpress_news_category_tr_lite](https://huggingface.co/datasets/yavuzkomecoglu/interpress_news_category_tr_lite) — Interpress Media Monitoring Company.
