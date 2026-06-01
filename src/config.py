from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_NAME = "yavuzkomecoglu/interpress_news_category_tr_lite"

INTERPRESS_ZIP_URL = (
    "https://www.interpress.com/downloads/interpress_news_category_tr_270k_lite.zip"
)
INTERPRESS_CACHE_DIR = PROJECT_ROOT / "data" / "interpress_cache"
INTERPRESS_TRAIN_TSV = "interpress_news_category_tr_270k_lite_train.tsv"
INTERPRESS_TEST_TSV = "interpress_news_category_tr_270k_lite_test.tsv"

CATEGORY_NAMES = [
    "kültürsanat",
    "ekonomi",
    "siyaset",
    "eğitim",
    "dünya",
    "spor",
    "teknoloji",
    "magazin",
    "sağlık",
    "gündem",
]

TEXT_COLUMN_CANDIDATES = ("content", "text", "news", "article", "body", "sentence")
LABEL_COLUMN_CANDIDATES = ("category", "label", "labels", "class", "target")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

FAST_BENCHMARK_MODE = True

MAX_TRAIN_SAMPLES = 12000
MAX_VAL_SAMPLES = 2000
MAX_TEST_SAMPLES = 2000

MODELS = [
    "dbmdz/bert-base-turkish-cased",
    "distilbert-base-multilingual-cased",
    "xlm-roberta-base",
]

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16 = True

if FAST_BENCHMARK_MODE:
    EPOCHS = 1
    NUM_EPOCHS = EPOCHS
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    SEEDS = [42]
else:
    EPOCHS = 3
    NUM_EPOCHS = EPOCHS
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    SEEDS = [42, 123, 2026]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_CSV = OUTPUTS_DIR / "results.csv"
FAILED_EXAMPLES_CSV = OUTPUTS_DIR / "failed_examples.csv"
EVAL_RESULTS_CSV = OUTPUTS_DIR / "eval_results.csv"

MIN_FAILED_EXAMPLES = 5

SAVE_STRATEGY = "no"
LOGGING_STEPS = 100
EVALUATION_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1

BASELINE_MAX_SAMPLES = None
