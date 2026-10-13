import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from triage.config import SEED, TEST_SIZE
from triage.labels import URGENCY_MAP


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["phrase", "prompt"]]
    df["urgency"] = df["prompt"].map(URGENCY_MAP)
    return df


def split_dataset(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    groups = df["phrase"].str.lower().str.strip()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    return df.iloc[train_idx], df.iloc[test_idx]
