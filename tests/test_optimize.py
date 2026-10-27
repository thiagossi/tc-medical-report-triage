"""Testes da versão otimizada (ONNX) do modelo."""

import pytest

from triage.optimize import export_onnx, unique_texts
from triage.predict import load_model, predict_text
from triage.train import save_model


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, trained_model, sample_df):
    """Pasta com model.joblib, model.onnx e metadata.json, como a do treino."""
    folder = tmp_path_factory.mktemp("models")
    save_model(trained_model, {}, len(sample_df), folder)
    export_onnx(trained_model, unique_texts(sample_df), folder)
    return folder


def test_export_creates_onnx_file_and_metadata(model_dir):
    assert (model_dir / "model.onnx").exists()
    _, metadata = load_model(model_dir, backend="onnx")
    assert metadata["onnx"]["check"]["label_agreement"] == 1.0
    assert metadata["backend"] == "onnx"


@pytest.mark.parametrize(
    "text",
    ["My heart hurts a lot", "I have acne  on my face", "  I feel dizzy  ", "Ïñtérnál pain"],
)
def test_onnx_answers_exactly_like_sklearn(model_dir, text):
    sklearn_model, _ = load_model(model_dir, backend="sklearn")
    onnx_model, _ = load_model(model_dir, backend="onnx")
    assert predict_text(onnx_model, text) == predict_text(sklearn_model, text)


def test_onnx_backend_without_export_fails_clearly(tmp_path, trained_model):
    save_model(trained_model, {}, 1, tmp_path)  # salvou só o joblib
    with pytest.raises(FileNotFoundError, match="triage.optimize"):
        load_model(tmp_path, backend="onnx")


def test_unknown_backend_is_rejected(model_dir):
    with pytest.raises(ValueError, match="MODEL_BACKEND"):
        load_model(model_dir, backend="tensorflow")
