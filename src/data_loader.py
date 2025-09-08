from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STATUS_MAP = {
    "U": "SOLD",
    "SC": "SOLD_CONDITIONAL",
    "LC": "LEASED_CONDITIONAL",
    "A": "ACTIVE",
    "R": "LEASED",
}


@dataclass(frozen=True)
class MLSDataSplit:
    sold_df: pd.DataFrame
    lease_df: pd.DataFrame
    active_df: pd.DataFrame


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def normalize_status(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    mls_status = df.get("MlsStatus", pd.Series(index=df.index, dtype="object"))
    mls_status = mls_status.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "_", regex=False)

    status_aur = df.get("Status_aur", pd.Series(index=df.index, dtype="object"))
    status_aur = status_aur.fillna("").astype(str).str.strip().str.upper().map(STATUS_MAP).fillna("")

    df["normalized_status"] = mls_status.where(mls_status != "", status_aur)
    return df


def split_by_status(dataframe: pd.DataFrame) -> MLSDataSplit:
    df = normalize_status(dataframe)

    sold_mask = df["normalized_status"].isin(["SOLD", "SOLD_CONDITIONAL"]) & df["ClosePrice"].notna()
    lease_mask = df["normalized_status"].isin(["LEASED", "LEASED_CONDITIONAL"]) & df["LeaseAmount"].notna()
    active_mask = df["normalized_status"].isin(["ACTIVE", "NEW", "PRICE_CHANGE", "EXTENSION"])

    return MLSDataSplit(
        sold_df=df.loc[sold_mask].copy(),
        lease_df=df.loc[lease_mask].copy(),
        active_df=df.loc[active_mask].copy(),
    )
