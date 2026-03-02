# 🏠 Intelligent Property Price Prediction

**Project 9 — Capstone Project**

A modular machine learning system that predicts Melbourne property prices using classical ML models (Scikit-Learn). Features a complete data pipeline with imputation, encoding, and scaling, plus a Streamlit web interface for real-time predictions.

---

## 📁 Project Structure

```
GenAI/
├── data/
│   └── melbourne_housing.csv    # Dataset (13,581 rows)
├── models/
│   ├── best_model.pkl           # Trained model pipeline (auto-generated)
│   └── model_metadata.json      # Feature list & metrics (auto-generated)
├── data_preprocessing.py        # Data loading, feature engineering, sklearn pipeline
├── train_model.py               # Model training, evaluation & persistence
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the Melbourne Housing dataset
- Train **Linear Regression** (baseline) and **Random Forest Regressor**
- Print evaluation metrics (R², MAE, RMSE) for both models
- Save the best model to `models/best_model.pkl`

### 3. Launch the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the interactive predictor.

---

## 🔬 Methodology

### Data Pipeline (Scikit-Learn `ColumnTransformer`)

| Step | Numerical Features | Categorical Features |
|------|-------------------|---------------------|
| **Imputation** | `SimpleImputer(strategy='median')` | `SimpleImputer(strategy='most_frequent')` |
| **Transformation** | `StandardScaler` | `OneHotEncoder(handle_unknown='ignore')` |

### Feature Engineering

- **HouseAge** = Sale Year − Year Built (computed from `Date` and `YearBuilt` columns)

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Rooms | Numerical | Number of rooms |
| Distance | Numerical | Distance from CBD (km) |
| Bedroom2 | Numerical | Number of bedrooms |
| Bathroom | Numerical | Number of bathrooms |
| Car | Numerical | Number of car spaces |
| Landsize | Numerical | Land size (m²) |
| BuildingArea | Numerical | Building area (m²) |
| HouseAge | Numerical | Age of the house (years) |
| Type | Categorical | h=house, t=townhouse, u=unit |
| Regionname | Categorical | Melbourne region name |

### Models

- **Linear Regression** — Baseline model
- **Random Forest Regressor** — Primary model (200 estimators)

### Evaluation Metrics

- **R²** (Coefficient of Determination)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-Learn** — ML pipelines, models, preprocessing
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Interactive web UI
- **Joblib** — Model serialization

---

## 📄 License

This project is for educational purposes as part of a Capstone project.
