"""Montagem do pipeline de classificação."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from triage.config import (
    CHAR_NGRAM_RANGE,
    MAX_DEPTH,
    MIN_DF,
    N_ESTIMATORS,
    NGRAM_RANGE,
    SEED,
)


def build_pipeline() -> Pipeline:
    """Cria o pipeline TF-IDF (palavras + letras) + Random Forest (ainda não treinado)."""
    features = FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)),
            (
                "chars",
                TfidfVectorizer(analyzer="char", ngram_range=CHAR_NGRAM_RANGE, min_df=MIN_DF),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
