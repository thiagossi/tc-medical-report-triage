"""Otimização: converte o modelo sklearn para ONNX e confere se as respostas são iguais.

O ONNX é um formato aberto de modelo. O ONNX Runtime executa esse arquivo em C++, sem
passar pelo Python do sklearn a cada predição, o que deixa a inferência mais rápida.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.pipeline import Pipeline

from triage.config import DATA_PATH, MODEL_DIR
from triage.data import clean_dataset, load_dataset
from triage.predict import OnnxModel, apply_threshold, load_model

logger = logging.getLogger(__name__)

# O tokenizador do ONNX Runtime usa a biblioteca RE2, que não aceita o "(?u)" do padrão
# do sklearn. Esta expressão separa as mesmas palavras: 2 ou mais letras, números ou "_".
WORD_TOKEN_EXP = r"[\pL\pN_]{2,}"
# "C" existe em qualquer sistema; "en_US" pode faltar na imagem Docker e quebrar o ONNX.
LOCALE = "C"
ONNX_FILE = "model.onnx"


def convert_to_onnx(model: Pipeline) -> bytes:
    """Converte o pipeline TF-IDF + Random Forest para ONNX."""
    words, chars = (vec for _, vec in model.named_steps["features"].transformer_list)
    options = {
        id(words): {"locale": LOCALE, "tokenexp": WORD_TOKEN_EXP},
        id(chars): {"locale": LOCALE},
        # zipmap=False: as probabilidades saem como matriz, igual ao predict_proba.
        id(model.named_steps["clf"]): {"zipmap": False},
    }
    # O skl2onnx avisa que o TF-IDF de letras "pode não funcionar". Não ignoramos o risco:
    # a função compare() confere frase a frase e o export falha se houver diferença.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The conversion of CountVectorizer")
        onx = to_onnx(
            model,
            initial_types=[("text", StringTensorType([None, 1]))],
            options=options,
            target_opset=18,
        )
    return onx.SerializeToString()


def compare(model: Pipeline, onnx_model: OnnxModel, texts: list[str]) -> dict:
    """Compara sklearn e ONNX nas mesmas frases."""
    p_sklearn = model.predict_proba(texts)
    p_onnx = onnx_model.predict_proba(texts)
    same_label = apply_threshold(p_sklearn, model.classes_) == apply_threshold(
        p_onnx, onnx_model.classes_
    )
    return {
        "n_texts": len(texts),
        "label_agreement": round(float(same_label.mean()), 4),
        "max_proba_diff": float(np.abs(p_sklearn - p_onnx).max()),
    }


def export_onnx(model: Pipeline, texts: list[str], model_dir: Path = MODEL_DIR) -> Path:
    """Salva model.onnx ao lado do model.joblib, só se as respostas forem iguais."""
    onnx_bytes = convert_to_onnx(model)
    onnx_model = OnnxModel(onnx_bytes, list(model.classes_))
    check = compare(model, onnx_model, texts)
    logger.info("sklearn x ONNX: %s", check)
    if check["label_agreement"] < 1.0:
        raise ValueError(f"O ONNX não concorda com o sklearn em todas as frases: {check}")

    onnx_path = model_dir / ONNX_FILE
    onnx_path.write_bytes(onnx_bytes)

    # Guarda no metadata.json as classes (ordem das colunas) e o resultado da conferência.
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["onnx"] = {
        "file": ONNX_FILE,
        "classes": [str(c) for c in model.classes_],
        "size_mb": round(len(onnx_bytes) / 1e6, 2),
        "check": check,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return onnx_path


def unique_texts(df: pd.DataFrame) -> list[str]:
    """Frases sem repetição, para a conferência não contar a mesma frase várias vezes."""
    return sorted(df["phrase"].unique().tolist())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, _ = load_model(backend="sklearn")
    texts = unique_texts(clean_dataset(load_dataset(DATA_PATH)))
    logger.info("Modelo ONNX salvo em %s", export_onnx(model, texts))
