"""Montagem do pipeline de classificação."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from triage.config import MAX_DEPTH, MIN_DF, N_ESTIMATORS, NGRAM_RANGE, SEED


def build_pipeline() -> Pipeline:
    """Cria o pipeline TF-IDF + Random Forest (ainda não treinado)."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)),
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
