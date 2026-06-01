"""
Proje yapılandırması: dataset, modeller, seed'ler ve eğitim hiperparametreleri.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Proje kökü (src/ bir üst dizin)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "yavuzkomecoglu/interpress_news_category_tr_lite"

# Olası metin / etiket kolon adları (otomatik tespit için)
TEXT_COLUMN_CANDIDATES = ("content", "text", "news", "article", "body", "sentence")
LABEL_COLUMN_CANDIDATES = ("category", "label", "labels", "class", "target")

# Train / validation / test oranları
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------
MODELS = [
    "dbmdz/bert-base-turkish-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
]

# ---------------------------------------------------------------------------
# Seed'ler
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 2026]

# ---------------------------------------------------------------------------
# Eğitim hiperparametreleri (Kaggle T4 için güvenli varsayılanlar)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16 = True  # T4'te bellek ve hız için

# ---------------------------------------------------------------------------
# Çıktı yolları
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_CSV = OUTPUTS_DIR / "results.csv"
FAILED_EXAMPLES_CSV = OUTPUTS_DIR / "failed_examples.csv"
EVAL_RESULTS_CSV = OUTPUTS_DIR / "eval_results.csv"

# Minimum hatalı örnek sayısı
MIN_FAILED_EXAMPLES = 5

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
SAVE_STRATEGY = "no"  # Ara checkpoint yok; sadece son model kaydedilir
LOGGING_STEPS = 100
EVALUATION_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1

# Baseline değerlendirmede kullanılacak örnek sayısı üst sınırı (hız için, None = tüm test)
BASELINE_MAX_SAMPLES = None
