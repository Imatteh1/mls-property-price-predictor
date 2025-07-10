from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path("property_v2.csv")
TARGET_COLUMN = "ClosePrice"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = Path("data")


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def build_modeling_dataset(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column '{target_column}' is missing.")

    dataset = dataframe[dataframe[target_column].notna()].copy()
    if dataset.empty:
        raise ValueError("No rows available for training after filtering null targets.")

    return dataset


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = dataset.sample(frac=1.0, random_state=RANDOM_STATE)
    split_index = int(len(shuffled) * (1 - TEST_SIZE))
    train_df = shuffled.iloc[:split_index].copy()
    test_df = shuffled.iloc[split_index:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed: one split is empty.")

    return train_df, test_df


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def main() -> None:
    raw_df = load_data(CSV_PATH)
    modeling_df = build_modeling_dataset(raw_df, TARGET_COLUMN)
    train_df, test_df = split_dataset(modeling_df)
    save_split(train_df, test_df, OUTPUT_DIR)

    print(f"Loaded rows: {len(raw_df):,}")
    print(f"Rows with {TARGET_COLUMN}: {len(modeling_df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Saved: {OUTPUT_DIR / 'train.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
