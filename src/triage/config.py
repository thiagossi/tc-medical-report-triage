"""Configurações do pipeline de treino."""

SEED = 42
TEST_SIZE = 0.2
DATA_PATH = "data/raw/overview-of-recordings.csv"

# TF-IDF
NGRAM_RANGE = (1, 2)
MIN_DF = 2

# Random Forest
N_ESTIMATORS = 100
MAX_DEPTH = None
