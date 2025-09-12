# MLS Property Price Predictor & Fair Value Analyzer

End-to-end MLS project based on `MLS_DS_Project_Spec.docx` using `property_v2.csv` as the data source.

## What is implemented
- CSV ingestion and status normalization (`sold`, `leased`, `active` splits)
- Data cleaning, missingness filtering, numeric/date parsing, and outlier removal
- Feature engineering from the specification (price, timing, and amenity features)
- Target encoding for `CommunityName` and `PostalCode`
- Two training pipelines:
  - Sold price model (`ClosePrice`)
  - Lease price model (`LeaseAmount`)
- Model evaluation report with MAE, RMSE, MAPE, and R2
- Saved model artifacts under `models/`
- 4-tab Streamlit app:
  - Price Predictor
  - Fair Value Checker
  - Rent Estimator
  - Market Dashboard
- SHAP utility module for local/global explainability plots

## Project structure
- `src/data_loader.py`: data loading and status split
- `src/features.py`: cleaning, feature engineering, target encoding
- `src/model.py`: model training, comparison, and evaluation
- `src/explainer.py`: SHAP plot generation
- `src/pipeline.py`: one-command training entrypoint
- `app/streamlit_app.py`: interactive dashboard
- `reports/model_evaluation.json`: generated model metrics
- `models/*.pkl`: generated trained model artifacts

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run training
```bash
python -m src.pipeline
```

## Run dashboard
```bash
streamlit run app/streamlit_app.py
```

## Notes
- If `xgboost` or `lightgbm` are unavailable, fallback model candidates are still trained.
- SHAP plots require `shap` installed.
