import pandas as pd

from triage.labels import URGENCY_MAP


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["phrase", "prompt"]]
    df["urgency"] = df["prompt"].map(URGENCY_MAP)
    return df