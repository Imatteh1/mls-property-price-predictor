from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import load_csv, split_by_status
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
    apply_target_encoders,
    clean_base_dataframe,
    coerce_feature_types,
    fit_target_encoders,
    remove_target_outliers,
)

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

from sklearn.ensemble import RandomForestRegressor


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    r2: float


@dataclass
class TrainingResult:
    model_name: str
    metrics: Metrics


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _candidate_models() -> dict[str, object]:
    candidates: dict[str, object] = {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    if XGBRegressor is not None:
        candidates["xgboost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    if LGBMRegressor is not None:
        candidates["lightgbm"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    return candidates


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return Metrics(mae=mae, rmse=rmse, mape=mape, r2=r2)


def _train_best_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[Pipeline, TrainingResult]:
    best_pipeline: Pipeline | None = None
    best_result: TrainingResult | None = None

    for model_name, model in _candidate_models().items():
        pipeline = Pipeline(steps=[("preprocessor", _build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = _compute_metrics(y_test, predictions)
        result = TrainingResult(model_name=model_name, metrics=metrics)

        if best_result is None or result.metrics.mae < best_result.metrics.mae:
            best_pipeline = pipeline
            best_result = result

    if best_pipeline is None or best_result is None:
        raise RuntimeError("No model candidates available for training.")

    return best_pipeline, best_result


def _prepare_training_frame(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    cleaned = clean_base_dataframe(dataframe)
    enriched = add_engineered_features(cleaned, target_column=target_column)
    filtered = remove_target_outliers(enriched, target_column=target_column)
    return filtered


def _split_features_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    features = coerce_feature_types(dataframe, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = pd.to_numeric(dataframe[target_column], errors="coerce")
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def _train_single_target(dataframe: pd.DataFrame, target_column: str) -> tuple[Pipeline, TrainingResult, dict[str, object]]:
    prepared = _prepare_training_frame(dataframe, target_column=target_column)
    train_df, test_df = train_test_split(prepared, test_size=0.2, random_state=42, shuffle=True)

    encoders = fit_target_encoders(train_df, target_column=target_column)
    train_encoded = apply_target_encoders(train_df, encoders)
    test_encoded = apply_target_encoders(test_df, encoders)

    X_train, y_train = _split_features_target(train_encoded, target_column=target_column)
    X_test, y_test = _split_features_target(test_encoded, target_column=target_column)

    best_pipeline, result = _train_best_model(X_train, y_train, X_test, y_test)

    metadata = {
        "target_column": target_column,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoders": asdict(encoders),
        "metrics": asdict(result.metrics),
        "model_name": result.model_name,
    }
    return best_pipeline, result, metadata


def train_project(csv_path: Path, model_dir: Path, report_path: Path) -> dict[str, object]:
    raw_df = load_csv(csv_path)
    split = split_by_status(raw_df)

    sold_pipeline, sold_result, sold_meta = _train_single_target(split.sold_df, target_column="ClosePrice")
    lease_pipeline, lease_result, lease_meta = _train_single_target(split.lease_df, target_column="LeaseAmount")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(sold_pipeline, model_dir / "sold_model.pkl")
    joblib.dump(lease_pipeline, model_dir / "lease_model.pkl")

    full_report = {
        "sold": {
            "model": sold_result.model_name,
            "metrics": asdict(sold_result.metrics),
            "rows": int(len(split.sold_df)),
            "metadata": sold_meta,
        },
        "lease": {
            "model": lease_result.model_name,
            "metrics": asdict(lease_result.metrics),
            "rows": int(len(split.lease_df)),
            "metadata": lease_meta,
        },
        "active_rows": int(len(split.active_df)),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    return full_report


if __name__ == "__main__":
    report = train_project(
        csv_path=Path("property_v2.csv"),
        model_dir=Path("models"),
        report_path=Path("reports/model_evaluation.json"),
    )
    print(json.dumps(report, indent=2))
