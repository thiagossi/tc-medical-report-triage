"""Dados e modelo compartilhados pelos testes (sem depender do dataset real)."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from triage.data import clean_dataset, load_dataset, split_dataset
from triage.train import train_model

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample.csv"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """CSV pequeno, versionado no repositório, com as 25 categorias."""
    return clean_dataset(load_dataset(FIXTURE_CSV))


@pytest.fixture(scope="session")
def trained_model(sample_df: pd.DataFrame) -> Pipeline:
    """Modelo treinado uma vez e reaproveitado por vários testes."""
    train_df, _ = split_dataset(sample_df)
    return train_model(train_df)
