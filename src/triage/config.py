"""Configurações do projeto: caminhos, sementes e hiperparâmetros."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
TEST_SIZE = 0.2
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data/raw/overview-of-recordings.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_VERSION = "1.0.0"

# TF-IDF
NGRAM_RANGE = (1, 2)  # palavras: 1 ou 2 palavras juntas
CHAR_NGRAM_RANGE = (3, 5)  # letras: pedaços de 3 a 5 letras
MIN_DF = 2

# Random Forest
N_ESTIMATORS = 100
MAX_DEPTH = None

# Decisão: na dúvida, prioriza urgente
URGENT_THRESHOLD = 0.25
