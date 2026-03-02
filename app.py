"""
app.py
======
Streamlit application for real-time Melbourne property price prediction.
Loads the trained model pipeline and lets users input house details to get
an instant price estimate.

Author : Project 9 — Intelligent Property Price Prediction
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Melbourne Property Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Main container ---- */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ---- Header ---- */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* ---- Prediction card ---- */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.35);
    }
    .prediction-label {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .prediction-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-title {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #667eea;
    }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        min-width: 340px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Divider ---- */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained model pipeline."""
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_resource
def load_metadata():
    """Load model metadata (features, metrics, etc.)."""
    with open(META_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Region name choices (from dataset)
# ---------------------------------------------------------------------------
REGION_CHOICES = [
    "Northern Metropolitan",
    "Western Metropolitan",
    "Southern Metropolitan",
    "Eastern Metropolitan",
    "South-Eastern Metropolitan",
    "Northern Victoria",
    "Western Victoria",
    "Eastern Victoria",
]

TYPE_CHOICES = {"House": "h", "Townhouse": "t", "Unit/Apartment": "u"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    # Header ----------------------------------------------------------------
    st.markdown('<p class="hero-title">🏠 Melbourne Property Price Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Enter property details to get an AI‑powered price estimate using a trained Random Forest model.</p>',
        unsafe_allow_html=True,
    )

    # Load model ------------------------------------------------------------
    try:
        model = load_model()
        metadata = load_metadata()
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found. Please run `python train_model.py` first "
            "to train and save the model."
        )
        return

    # Sidebar — input form -------------------------------------------------
    with st.sidebar:
        st.markdown("### 🏡 Property Details")
        st.markdown("---")

        rooms = st.slider("Rooms", min_value=1, max_value=10, value=3, step=1)
        bedroom2 = st.slider("Bedrooms", min_value=0, max_value=10, value=3, step=1)
        bathroom = st.slider("Bathrooms", min_value=0, max_value=8, value=1, step=1)
        car = st.slider("Car Spaces", min_value=0, max_value=10, value=1, step=1)

        st.markdown("---")

        property_type_label = st.selectbox("Property Type", list(TYPE_CHOICES.keys()))
        property_type = TYPE_CHOICES[property_type_label]

        regionname = st.selectbox("Region", REGION_CHOICES)

        st.markdown("---")

        distance = st.number_input(
            "Distance from CBD (km)", min_value=0.0, max_value=60.0, value=10.0, step=0.5
        )
        landsize = st.number_input(
            "Land Size (m²)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0
        )
        building_area = st.number_input(
            "Building Area (m²)", min_value=0.0, max_value=5000.0, value=150.0, step=10.0
        )
        house_age = st.number_input(
            "House Age (years)", min_value=0, max_value=200, value=30, step=1
        )

    # Build feature DataFrame -----------------------------------------------
    input_data = pd.DataFrame([{
        "Rooms": rooms,
        "Distance": distance,
        "Bedroom2": bedroom2,
        "Bathroom": bathroom,
        "Car": car,
        "Landsize": landsize,
        "BuildingArea": building_area,
        "HouseAge": house_age,
        "Type": property_type,
        "Regionname": regionname,
    }])

    # Predict ---------------------------------------------------------------
    prediction = model.predict(input_data)[0]
    prediction = max(prediction, 0)  # clamp negative predictions

    # Display prediction card -----------------------------------------------
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Property Price</div>
                <div class="prediction-value">${prediction:,.0f} AUD</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_side:
        st.markdown("#### 📊 Model Performance")
        best_name = metadata.get("best_model", "Random Forest Regressor")
        best_metrics = metadata.get("metrics", {}).get(best_name, {})

        r2_val = best_metrics.get("r2", 0)
        mae_val = best_metrics.get("mae", 0)
        rmse_val = best_metrics.get("rmse", 0)

        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:0.8rem;">
                <div class="metric-title">R² Score</div>
                <div class="metric-value">{r2_val:.4f}</div>
            </div>
            <div class="metric-card" style="margin-bottom:0.8rem;">
                <div class="metric-title">MAE</div>
                <div class="metric-value">${mae_val:,.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">RMSE</div>
                <div class="metric-value">${rmse_val:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Input summary ---------------------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### 📋 Input Summary")

    summary_cols = st.columns(5)
    labels = [
        ("🏠 Type", property_type_label),
        ("🛏️ Rooms", f"{rooms} rooms, {bedroom2} beds"),
        ("🚿 Baths", str(bathroom)),
        ("📍 Distance", f"{distance} km"),
        ("📐 Area", f"{building_area} m²"),
    ]
    for col, (label, value) in zip(summary_cols, labels):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer ----------------------------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption(
        "Built with Scikit-Learn & Streamlit · Project 9 — Intelligent Property Price Prediction"
    )


if __name__ == "__main__":
    main()
