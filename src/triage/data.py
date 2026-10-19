import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from triage.config import MIN_ROWS, SEED, TEST_SIZE
from triage.labels import LEVELS, URGENCY_MAP

REQUIRED_COLUMNS = ["phrase", "prompt", "urgency"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[["phrase", "prompt"]]
    df["urgency"] = df["prompt"].map(URGENCY_MAP)
    return df


def validate_dataset(df: pd.DataFrame, min_rows: int = MIN_ROWS) -> None:
    """Para o processo com um erro claro se os dados não servirem para treinar."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no dataset: {missing}")

    unmapped = sorted(df.loc[df["urgency"].isna(), "prompt"].unique())
    if unmapped:
        raise ValueError(f"Categorias sem nível de urgência no labels.py: {unmapped}")

    if len(df) < min_rows:
        raise ValueError(f"Dataset com {len(df)} linhas; o mínimo é {min_rows}")

    missing_levels = sorted(set(LEVELS) - set(df["urgency"]))
    if missing_levels:
        raise ValueError(f"Níveis sem nenhum exemplo: {missing_levels}")


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
