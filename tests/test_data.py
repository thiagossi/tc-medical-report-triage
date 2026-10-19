"""Testes de carregamento, limpeza e separação dos dados."""

import pandas as pd

from triage.data import clean_dataset, phrase_key, split_dataset


def test_every_row_gets_an_urgency_level(sample_df):
    assert sample_df["urgency"].notna().all()


def test_clean_removes_label_that_disagrees_with_majority():
    df = pd.DataFrame(
        {
            "phrase": ["My knee hurts"] * 3 + ["I feel dizzy"],
            "prompt": ["Knee pain", "Knee pain", "Acne", "Feeling dizzy"],
        }
    )
    cleaned = clean_dataset(df)
    assert len(cleaned) == 3
    assert "Acne" not in cleaned["prompt"].values


def test_same_phrase_never_in_train_and_test(sample_df):
    train_df, test_df = split_dataset(sample_df)
    assert set(phrase_key(train_df)).isdisjoint(set(phrase_key(test_df)))


def test_split_keeps_all_levels_in_test(sample_df):
    _, test_df = split_dataset(sample_df)
    assert set(test_df["urgency"]) == {"normal", "atencao", "urgente"}
