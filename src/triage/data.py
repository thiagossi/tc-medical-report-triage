import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from triage.config import SEED, TEST_SIZE
from triage.labels import URGENCY_MAP


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["phrase", "prompt"]]
    df["urgency"] = df["prompt"].map(URGENCY_MAP)
    return df


def phrase_key(df: pd.DataFrame) -> pd.Series:
    """Frase normalizada, usada para identificar cópias da mesma frase."""
    return df["phrase"].str.lower().str.strip()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas cujo assunto discorda da maioria das cópias da mesma frase."""
    key = phrase_key(df)
    majority = df.groupby(key)["prompt"].agg(lambda s: s.mode().iloc[0])
    is_consistent = df["prompt"] == key.map(majority)
    return df[is_consistent].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = phrase_key(df)

    n_splits = round(1 / test_size)  # 0.2 → 5 partes
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, y=df["urgency"], groups=groups))

    return df.iloc[train_idx], df.iloc[test_idx]


if __name__ == "__main__":
    from triage.config import DATA_PATH

    df = clean_dataset(load_dataset(DATA_PATH))
    train, test = split_dataset(df)

    print("linhas:", len(df), "| treino:", len(train), "| teste:", len(test))
    print("treino:", train["urgency"].value_counts(normalize=True).round(2).to_dict())
    print("teste: ", test["urgency"].value_counts(normalize=True).round(2).to_dict())
