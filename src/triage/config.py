"""Configurações do pipeline de treino."""

SEED = 42
TEST_SIZE = 0.2
DATA_PATH = "data/raw/overview-of-recordings.csv"

# TF-IDF
# TF-IDF
NGRAM_RANGE = (1, 2)  # palavras: 1 ou 2 palavras juntas
CHAR_NGRAM_RANGE = (3, 5)  # letras: pedaços de 3 a 5 letras
MIN_DF = 2

# Random Forest
N_ESTIMATORS = 100
MAX_DEPTH = None

# Decisão: na dúvida, prioriza urgente
URGENT_THRESHOLD = 0.25
