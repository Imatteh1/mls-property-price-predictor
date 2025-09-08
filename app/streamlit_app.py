from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_csv, split_by_status
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, clean_base_dataframe, coerce_feature_types

MODEL_DIR = Path("models")
REPORT_PATH = Path("reports/model_evaluation.json")
DATA_PATH = Path("property_v2.csv")


def _load_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _load_models():
    sold_path = MODEL_DIR / "sold_model.pkl"
    lease_path = MODEL_DIR / "lease_model.pkl"
    if not sold_path.exists() or not lease_path.exists():
        return None, None
    return joblib.load(sold_path), joblib.load(lease_path)


def _apply_target_encoding(input_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = input_df.copy()
    enc = metadata.get("encoders", {})

    community_mean = enc.get("community_mean", {})
    community_median = enc.get("community_median", {})
    postal_mean = enc.get("postal_mean", {})
    global_mean = float(enc.get("global_mean", 0.0))
    global_median = float(enc.get("global_median", global_mean))

    community = df.get("CommunityName", "").fillna("").astype(str)
    postal = df.get("PostalCode", "").fillna("").astype(str)

    df["community_target_mean"] = community.map(community_mean).fillna(global_mean)
    df["community_target_median"] = community.map(community_median).fillna(global_median)
    df["postal_target_mean"] = postal.map(postal_mean).fillna(global_mean)
    return df


def _prepare_single_input(inputs: dict, metadata: dict) -> pd.DataFrame:
    base = pd.DataFrame([inputs])
    base = _apply_target_encoding(base, metadata)
    features = coerce_feature_types(base, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _fair_value_verdict(list_price: float, predicted_price: float) -> tuple[str, float]:
    if list_price > predicted_price * 1.05:
        return "OVERPRICED", list_price - predicted_price
    if list_price < predicted_price * 0.95:
        return "UNDERPRICED", predicted_price - list_price
    return "FAIR VALUE", abs(list_price - predicted_price)


def _confidence_score(comparables_count: int) -> str:
    if comparables_count >= 20:
        return "high"
    if comparables_count >= 8:
        return "medium"
    return "low"


def _build_comparables(sold_df: pd.DataFrame, bedroom: int, bath: int, community: str) -> pd.DataFrame:
    cols = ["ListingKey", "CommunityName", "BedroomsTotal", "BathroomsTotalInteger", "ClosePrice", "ListPrice", "DaysOnMarket"]
    available_cols = [c for c in cols if c in sold_df.columns]
    comp = sold_df[available_cols].copy()

    if "BedroomsTotal" in comp.columns:
        comp["BedroomsTotal"] = pd.to_numeric(comp["BedroomsTotal"], errors="coerce")
    if "BathroomsTotalInteger" in comp.columns:
        comp["BathroomsTotalInteger"] = pd.to_numeric(comp["BathroomsTotalInteger"], errors="coerce")

    comp["score"] = 0
    if "CommunityName" in comp.columns:
        comp["score"] += (comp["CommunityName"].fillna("").astype(str) == community).astype(int) * 3
    if "BedroomsTotal" in comp.columns:
        comp["score"] += (3 - (comp["BedroomsTotal"] - bedroom).abs().clip(upper=3)).fillna(0)
    if "BathroomsTotalInteger" in comp.columns:
        comp["score"] += (2 - (comp["BathroomsTotalInteger"] - bath).abs().clip(upper=2)).fillna(0)

    return comp.sort_values("score", ascending=False).head(5)


def main() -> None:
    st.set_page_config(page_title="MLS Price Predictor", layout="wide")
    st.title("MLS Property Price Predictor & Fair Value Analyzer")

    sold_model, lease_model = _load_models()
    report = _load_report()

    raw_df = load_csv(DATA_PATH)
    split = split_by_status(clean_base_dataframe(raw_df))

    tab1, tab2, tab3, tab4 = st.tabs(["Price Predictor", "Fair Value Checker", "Rent Estimator", "Market Dashboard"])

    communities = sorted(split.sold_df.get("CommunityName", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

    with tab1:
        st.subheader("Price Predictor")
        col1, col2, col3 = st.columns(3)
        bedrooms = col1.slider("Bedrooms", 1, 8, 3)
        bathrooms = col2.slider("Bathrooms", 1, 6, 3)
        property_type = col3.selectbox("Property Type", ["Detached", "Semi-Detached", "Condo", "Att/Row/Townhouse"], index=0)

        col4, col5, col6 = st.columns(3)
        community = col4.selectbox("Community", communities if communities else ["Unknown"])
        lot_size = col5.number_input("Lot Size Area", min_value=0.0, value=2000.0)
        age = col6.slider("Approximate Age", 0, 100, 15)

        col7, col8, col9 = st.columns(3)
        has_garage = col7.toggle("Garage", value=True)
        has_basement = col8.toggle("Basement", value=True)
        has_fireplace = col9.toggle("Fireplace", value=False)

        taxes = st.number_input("Annual Taxes", min_value=0.0, value=5000.0)

        if st.button("Predict Sold Price"):
            if sold_model is None or "sold" not in report:
                st.error("Models are missing. Run: python -m src.pipeline")
            else:
                model_inputs = {
                    "BedroomsTotal": bedrooms,
                    "BedroomsAboveGrade": bedrooms,
                    "BedroomsBelowGrade": 0,
                    "BathroomsTotalInteger": bathrooms,
                    "KitchensTotal": 1,
                    "RoomsAboveGrade": bedrooms + 3,
                    "RoomsTotal": bedrooms + 4,
                    "GarageParkingSpaces": 2 if has_garage else 0,
                    "ParkingTotal": 2 if has_garage else 1,
                    "LotWidth": 40,
                    "LotDepth": 90,
                    "LotSizeArea": lot_size,
                    "ApproximateAge": age,
                    "TaxAnnualAmount": taxes,
                    "DaysOnMarket": 20,
                    "price_per_sqft": np.nan,
                    "list_to_sold_ratio": np.nan,
                    "sold_month": 6,
                    "sold_year": 2025,
                    "days_on_market_engineered": 20,
                    "bedroom_bath_ratio": bedrooms / max(bathrooms, 1),
                    "Lat": 43.46,
                    "Lng": -79.73,
                    "PropertyType": property_type,
                    "PropertySubType": property_type,
                    "ArchitecturalStyle": "2-Storey",
                    "GarageType": "Attached",
                    "Basement": "Finished",
                    "CommunityName": community,
                    "PostalCode": "",
                }

                feature_df = _prepare_single_input(model_inputs, report["sold"]["metadata"])
                pred = float(sold_model.predict(feature_df)[0])
                low, high = pred * 0.95, pred * 1.05

                st.metric("Predicted Sold Price", f"${pred:,.0f}")
                st.write(f"Confidence interval: ${low:,.0f} - ${high:,.0f}")

                comparables = _build_comparables(split.sold_df, bedrooms, bathrooms, community)
                st.write("Comparable sold properties")
                st.dataframe(comparables, use_container_width=True)

    with tab2:
        st.subheader("Fair Value Checker")
        listing_price = st.number_input("Current List Price", min_value=0.0, value=1200000.0)
        reference_bedrooms = st.slider("Bedrooms (reference)", 1, 8, 3, key="fv_bed")
        reference_bathrooms = st.slider("Bathrooms (reference)", 1, 6, 3, key="fv_bath")
        reference_community = st.selectbox("Community (reference)", communities if communities else ["Unknown"], key="fv_comm")

        if st.button("Check Fair Value"):
            if sold_model is None or "sold" not in report:
                st.error("Models are missing. Run: python -m src.pipeline")
            else:
                model_inputs = {
                    "BedroomsTotal": reference_bedrooms,
                    "BedroomsAboveGrade": reference_bedrooms,
                    "BedroomsBelowGrade": 0,
                    "BathroomsTotalInteger": reference_bathrooms,
                    "KitchensTotal": 1,
                    "RoomsAboveGrade": reference_bedrooms + 3,
                    "RoomsTotal": reference_bedrooms + 4,
                    "GarageParkingSpaces": 2,
                    "ParkingTotal": 2,
                    "LotWidth": 40,
                    "LotDepth": 90,
                    "LotSizeArea": 2000,
                    "ApproximateAge": 15,
                    "TaxAnnualAmount": 5000,
                    "DaysOnMarket": 20,
                    "price_per_sqft": np.nan,
                    "list_to_sold_ratio": np.nan,
                    "sold_month": 6,
                    "sold_year": 2025,
                    "days_on_market_engineered": 20,
                    "bedroom_bath_ratio": reference_bedrooms / max(reference_bathrooms, 1),
                    "Lat": 43.46,
                    "Lng": -79.73,
                    "PropertyType": "Detached",
                    "PropertySubType": "Detached",
                    "ArchitecturalStyle": "2-Storey",
                    "GarageType": "Attached",
                    "Basement": "Finished",
                    "CommunityName": reference_community,
                    "PostalCode": "",
                }
                features = _prepare_single_input(model_inputs, report["sold"]["metadata"])
                predicted = float(sold_model.predict(features)[0])
                verdict, diff = _fair_value_verdict(listing_price, predicted)

                comparables = _build_comparables(split.sold_df, reference_bedrooms, reference_bathrooms, reference_community)
                confidence = _confidence_score(len(comparables))

                st.metric("Predicted Fair Value", f"${predicted:,.0f}")
                st.metric("Verdict", verdict)
                st.write(f"Difference: ${diff:,.0f}")
                st.write(f"Confidence score: {confidence}")

    with tab3:
        st.subheader("Rent Estimator")
        rent_bedrooms = st.slider("Bedrooms", 1, 6, 2, key="rent_bed")
        rent_bathrooms = st.slider("Bathrooms", 1, 4, 2, key="rent_bath")
        rent_community = st.selectbox("Community", communities if communities else ["Unknown"], key="rent_comm")
        furnished = st.toggle("Furnished", value=False)

        if st.button("Estimate Rent"):
            if lease_model is None or "lease" not in report:
                st.error("Models are missing. Run: python -m src.pipeline")
            else:
                lease_inputs = {
                    "BedroomsTotal": rent_bedrooms,
                    "BedroomsAboveGrade": rent_bedrooms,
                    "BedroomsBelowGrade": 0,
                    "BathroomsTotalInteger": rent_bathrooms,
                    "KitchensTotal": 1,
                    "RoomsAboveGrade": rent_bedrooms + 2,
                    "RoomsTotal": rent_bedrooms + 3,
                    "GarageParkingSpaces": 1,
                    "ParkingTotal": 1,
                    "LotWidth": 25,
                    "LotDepth": 80,
                    "LotSizeArea": 1200,
                    "ApproximateAge": 10,
                    "TaxAnnualAmount": 0,
                    "DaysOnMarket": 15,
                    "price_per_sqft": np.nan,
                    "list_to_sold_ratio": np.nan,
                    "sold_month": 6,
                    "sold_year": 2025,
                    "days_on_market_engineered": 15,
                    "bedroom_bath_ratio": rent_bedrooms / max(rent_bathrooms, 1),
                    "Lat": 43.46,
                    "Lng": -79.73,
                    "PropertyType": "Condo",
                    "PropertySubType": "Condo",
                    "ArchitecturalStyle": "Apartment",
                    "GarageType": "Underground",
                    "Basement": "None",
                    "CommunityName": rent_community,
                    "PostalCode": "",
                    "Furnished": str(furnished),
                }
                features = _prepare_single_input(lease_inputs, report["lease"]["metadata"])
                predicted_rent = float(lease_model.predict(features)[0])
                st.metric("Predicted Monthly Rent", f"${predicted_rent:,.0f}")

    with tab4:
        st.subheader("Market Dashboard")
        sold = split.sold_df.copy()
        sold["CloseDate"] = pd.to_datetime(sold.get("CloseDate"), errors="coerce")
        sold["month"] = sold["CloseDate"].dt.to_period("M").astype(str)
        sold["ClosePrice"] = pd.to_numeric(sold.get("ClosePrice"), errors="coerce")

        monthly = sold.dropna(subset=["month", "ClosePrice"]).groupby("month", as_index=False)["ClosePrice"].median()
        if not monthly.empty:
            fig = px.line(monthly, x="month", y="ClosePrice", title="Median Sold Price Over Time")
            st.plotly_chart(fig, use_container_width=True)

        by_community = sold.dropna(subset=["CommunityName", "ClosePrice"]).groupby("CommunityName", as_index=False)["ClosePrice"].median()
        top_communities = by_community.sort_values("ClosePrice", ascending=False).head(10)
        if not top_communities.empty:
            fig2 = px.bar(top_communities, x="CommunityName", y="ClosePrice", title="Top 10 Communities by Median Sold Price")
            st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
