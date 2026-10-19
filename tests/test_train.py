"""Testes de treino, avaliação e salvamento do modelo."""

from triage.data import split_dataset
from triage.predict import load_model
from triage.train import evaluate_model, save_model


def test_evaluate_returns_all_metrics(sample_df, trained_model):
    _, test_df = split_dataset(sample_df)
    metrics = evaluate_model(trained_model, test_df)
    assert set(metrics) == {"accuracy", "f1_macro", "recall_urgente", "confusion_matrix"}
    assert 0 <= metrics["accuracy"] <= 1


def test_saved_model_can_be_loaded_back(tmp_path, trained_model):
    save_model(trained_model, {"accuracy": 1.0}, n_train=10, model_dir=tmp_path)

    model, metadata = load_model(tmp_path)

    assert metadata["n_train"] == 10
    assert list(model.classes_) == list(trained_model.classes_)
