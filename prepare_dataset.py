from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("property_v2.csv")
DEFAULT_TARGET_COLUMN = "ClosePrice"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load MLS CSV data and create train/test CSV splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train.csv and test.csv",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column that must be non-null for modeling rows",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of rows to place in test split (0 < test_size < 1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for deterministic shuffling",
    )
    return parser.parse_args()


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


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    shuffled = dataset.sample(frac=1.0, random_state=random_state)
    split_index = int(len(shuffled) * (1 - test_size))
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
    args = parse_args()
    raw_df = load_data(args.input)
    modeling_df = build_modeling_dataset(raw_df, args.target_column)
    train_df, test_df = split_dataset(modeling_df, args.test_size, args.random_state)
    save_split(train_df, test_df, args.output_dir)

    print(f"Loaded rows: {len(raw_df):,}")
    print(f"Rows with {args.target_column}: {len(modeling_df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Saved: {args.output_dir / 'train.csv'}")
    print(f"Saved: {args.output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
