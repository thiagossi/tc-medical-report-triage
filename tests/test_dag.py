"""Teste da DAG do Airflow (só roda onde o Airflow está instalado)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_FOLDER = Path(__file__).resolve().parents[1] / "dags"


def test_dag_loads_with_tasks_in_order():
    bag = DagBag(dag_folder=DAGS_FOLDER)
    assert bag.import_errors == {}

    dag = bag.get_dag("train_pipeline")
    order = [task.task_id for task in dag.topological_sort()]
    assert order == ["load_data", "validate_data", "train", "evaluate", "save"]
