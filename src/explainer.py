from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def _get_preprocessed_data(pipeline, features: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.transform(features)


def build_shap_explainer(pipeline, background: pd.DataFrame):
    if shap is None:
        raise ImportError("shap is not installed. Install shap>=0.44 to enable explainability.")

    model = pipeline.named_steps["model"]
    transformed = _get_preprocessed_data(pipeline, background)
    return shap.Explainer(model, transformed)


def save_waterfall_plot(explainer, pipeline, row: pd.DataFrame, output_path: Path) -> None:
    transformed = _get_preprocessed_data(pipeline, row)
    shap_values = explainer(transformed)

    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_beeswarm_plot(explainer, pipeline, features: pd.DataFrame, output_path: Path) -> None:
    transformed = _get_preprocessed_data(pipeline, features)
    shap_values = explainer(transformed)

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
