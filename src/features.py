from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["ListPrice", "OriginalListPrice", "ClosePrice", "LeaseAmount", "TaxAnnualAmount", "LotSizeArea"]
DATE_COLUMNS = ["ListingContractDate", "CloseDate", "PriceChangeTimestamp"]
TEXT_COLUMNS = ["CommunityName", "PostalCode", "PropertyType", "PropertySubType", "ArchitecturalStyle", "GarageType", "Basement", "MlsStatus"]

NUMERIC_FEATURES = [
    "BedroomsTotal",
    "BedroomsAboveGrade",
    "BedroomsBelowGrade",
    "BathroomsTotalInteger",
    "KitchensTotal",
    "RoomsAboveGrade",
    "RoomsTotal",
    "GarageParkingSpaces",
    "ParkingTotal",
    "LotWidth",
    "LotDepth",
    "LotSizeArea",
    "ApproximateAge",
    "TaxAnnualAmount",
    "DaysOnMarket",
    "price_per_sqft",
    "list_to_sold_ratio",
    "sold_month",
    "sold_year",
    "days_on_market_engineered",
    "bedroom_bath_ratio",
    "Lat",
    "Lng",
    "community_target_mean",
    "community_target_median",
    "postal_target_mean",
]

CATEGORICAL_FEATURES = [
    "PropertyType",
    "PropertySubType",
    "ArchitecturalStyle",
    "GarageType",
    "Basement",
]


@dataclass
class TargetEncodingMaps:
    community_mean: dict[str, float]
    community_median: dict[str, float]
    postal_mean: dict[str, float]
    global_mean: float
    global_median: float


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def _parse_age_to_midpoint(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip()
    midpoint = raw.str.extract(r"^(\d+)\s*-\s*(\d+)$")
    midpoint_value = (pd.to_numeric(midpoint[0], errors="coerce") + pd.to_numeric(midpoint[1], errors="coerce")) / 2
    direct = pd.to_numeric(raw, errors="coerce")
    return direct.fillna(midpoint_value)


def drop_high_missing_columns(dataframe: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    keep_columns = dataframe.columns[dataframe.isna().mean() <= threshold]
    return dataframe[keep_columns].copy()


def clean_base_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = drop_high_missing_columns(dataframe)

    for column in PRICE_COLUMNS:
        if column in df.columns:
            df[column] = _clean_numeric(df[column])

    if "ApproximateAge" in df.columns:
        df["ApproximateAge"] = _parse_age_to_midpoint(df["ApproximateAge"])

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    return df


def add_engineered_features(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = dataframe.copy()

    lot_size = df.get("LotSizeArea", pd.Series(index=df.index, dtype=float)).replace(0, np.nan)
    list_price = df.get("ListPrice", pd.Series(index=df.index, dtype=float)).replace(0, np.nan)

    target = df.get(target_column, pd.Series(index=df.index, dtype=float))

    df["price_per_sqft"] = target / lot_size
    df["list_to_sold_ratio"] = target / list_price
    df["price_drop_flag"] = df.get("PriceChangeTimestamp", pd.Series(index=df.index)).notna().astype(int)

    close_date = df.get("CloseDate", pd.Series(index=df.index, dtype="datetime64[ns]"))
    list_date = df.get("ListingContractDate", pd.Series(index=df.index, dtype="datetime64[ns]"))

    df["sold_month"] = close_date.dt.month
    df["sold_year"] = close_date.dt.year
    df["days_on_market_engineered"] = (close_date - list_date).dt.days

    bathrooms = df.get("BathroomsTotalInteger", pd.Series(index=df.index, dtype=float)).replace(0, np.nan)
    bedrooms = df.get("BedroomsTotal", pd.Series(index=df.index, dtype=float))
    df["bedroom_bath_ratio"] = bedrooms / bathrooms

    df["has_garage"] = df.get("GarageYN", pd.Series(index=df.index)).astype(str).str.upper().eq("TRUE").astype(int)
    df["has_basement"] = df.get("BasementYN", pd.Series(index=df.index)).astype(str).str.upper().eq("TRUE").astype(int)
    df["has_fireplace"] = df.get("FireplaceYN", pd.Series(index=df.index)).astype(str).str.upper().eq("TRUE").astype(int)

    return df


def remove_target_outliers(dataframe: pd.DataFrame, target_column: str, lower_q: float = 0.05, upper_q: float = 0.99) -> pd.DataFrame:
    df = dataframe.copy()
    target = pd.to_numeric(df[target_column], errors="coerce")
    lower = target.quantile(lower_q)
    upper = target.quantile(upper_q)
    return df[target.between(lower, upper)].copy()


def fit_target_encoders(train_df: pd.DataFrame, target_column: str) -> TargetEncodingMaps:
    community = train_df.get("CommunityName", pd.Series(index=train_df.index, dtype="object")).fillna("").astype(str)
    postal = train_df.get("PostalCode", pd.Series(index=train_df.index, dtype="object")).fillna("").astype(str)
    target = pd.to_numeric(train_df[target_column], errors="coerce")

    valid = target.notna()
    community = community[valid]
    postal = postal[valid]
    target = target[valid]

    return TargetEncodingMaps(
        community_mean=target.groupby(community).mean().to_dict(),
        community_median=target.groupby(community).median().to_dict(),
        postal_mean=target.groupby(postal).mean().to_dict(),
        global_mean=float(target.mean()),
        global_median=float(target.median()),
    )


def apply_target_encoders(dataframe: pd.DataFrame, encoders: TargetEncodingMaps) -> pd.DataFrame:
    df = dataframe.copy()

    community = df.get("CommunityName", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)
    postal = df.get("PostalCode", pd.Series(index=df.index, dtype="object")).fillna("").astype(str)

    df["community_target_mean"] = community.map(encoders.community_mean).fillna(encoders.global_mean)
    df["community_target_median"] = community.map(encoders.community_median).fillna(encoders.global_median)
    df["postal_target_mean"] = postal.map(encoders.postal_mean).fillna(encoders.global_mean)
    return df


def coerce_feature_types(dataframe: pd.DataFrame, numeric_features: Iterable[str], categorical_features: Iterable[str]) -> pd.DataFrame:
    df = dataframe.copy()
    for column in numeric_features:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in categorical_features:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    return df
