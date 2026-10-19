"""DAG de retreino do modelo de triagem.

Fluxo: carregar dados → validar → treinar → avaliar (barreira de qualidade) → salvar.

As tasks só chamam funções de `src/triage/` e passam entre si **caminhos de arquivo**
(não DataFrames), que ficam em `data/processed/airflow/<run_id>/`. O modelo aprovado
é salvo em `models/`, a mesma pasta que a API lê.
"""

import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from airflow.exceptions import AirflowFailException
from airflow.sdk import dag, get_current_context, task

from triage.config import DATA_PATH, MIN_RECALL_URGENTE, PROJECT_ROOT
from triage.data import clean_dataset, load_dataset, split_dataset, validate_dataset
from triage.train import evaluate_model, save_model, train_model

WORK_DIR = PROJECT_ROOT / "data" / "processed" / "airflow"


def run_dir() -> Path:
    """Pasta exclusiva desta execução da DAG."""
    run_id = get_current_context()["run_id"]
    path = WORK_DIR / re.sub(r"[^\w.-]", "_", run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dag(
    dag_id="train_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["triage", "treino"],
    doc_md=__doc__,
)
def train_pipeline():
    @task
    def load_data() -> str:
        """Lê o CSV bruto, remove rótulos inconsistentes e salva o resultado."""
        df = clean_dataset(load_dataset(DATA_PATH))
        out = run_dir() / "dataset.csv"
        df.to_csv(out, index=False)
        return str(out)

    @task
    def validate_data(dataset_path: str) -> dict:
        """Confere colunas, níveis e tamanho; depois separa treino e teste por frase."""
        df = pd.read_csv(dataset_path)
        validate_dataset(df)
        train_df, test_df = split_dataset(df)
        folder = Path(dataset_path).parent
        train_df.to_csv(folder / "train.csv", index=False)
        test_df.to_csv(folder / "test.csv", index=False)
        return {
            "train": str(folder / "train.csv"),
            "test": str(folder / "test.csv"),
            "n_train": len(train_df),
        }

    @task
    def train(splits: dict) -> str:
        """Treina o pipeline TF-IDF + Random Forest e guarda o candidato."""
        model = train_model(pd.read_csv(splits["train"]))
        out = Path(splits["train"]).parent / "candidate.joblib"
        joblib.dump(model, out)
        return str(out)

    @task
    def evaluate(candidate_path: str, splits: dict) -> dict:
        """Mede o candidato no teste. Se o recall de urgente cair, a DAG para aqui."""
        model = joblib.load(candidate_path)
        metrics = evaluate_model(model, pd.read_csv(splits["test"]))
        print(f"Métricas do candidato: {metrics}")
        if metrics["recall_urgente"] < MIN_RECALL_URGENTE:
            raise AirflowFailException(
                f"Recall de urgente {metrics['recall_urgente']} abaixo do mínimo "
                f"{MIN_RECALL_URGENTE}: modelo NÃO será salvo."
            )
        return metrics

    @task
    def save(candidate_path: str, metrics: dict, splits: dict) -> str:
        """Promove o candidato aprovado para models/, onde a API o encontra."""
        model = joblib.load(candidate_path)
        return str(save_model(model, metrics, splits["n_train"]))

    dataset = load_data()
    splits = validate_data(dataset)
    candidate = train(splits)
    metrics = evaluate(candidate, splits)
    save(candidate, metrics, splits)


train_pipeline()
