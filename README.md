# churn-prediction-pipeline

End-to-end customer churn prediction system — from raw data to a live API with explainability and drift monitoring.

Built on the IBM Telco Churn dataset. Covers the full ML lifecycle: feature engineering, XGBoost training, SHAP explainability, FastAPI deployment, and a Streamlit monitoring dashboard.

> Project in progress — README will be updated as each stage is completed.

---

## Project structure

```
churn-prediction-pipeline/
├── data/
│   ├── raw/               # Downloaded dataset (gitignored)
│   └── processed/         # Engineered features (gitignored)
├── notebooks/             # EDA and experimentation
├── src/
│   ├── data/
│   │   └── load.py        # Data loading + validation
│   ├── features/
│   │   └── engineer.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py       # XGBoost + Optuna + MLflow
│   │   └── explain.py     # SHAP summary + waterfall plots
│   ├── api/
│   │   └── main.py        # FastAPI prediction endpoint
│   └── monitoring/
│       └── drift.py       # PSI drift detection
├── tests/
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/churn-prediction-pipeline
cd churn-prediction-pipeline

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place telco_churn.csv in data/raw/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

---

## Stack

| Layer | Tool |
|---|---|
| Data | Pandas, Kaggle |
| Features | Scikit-learn, imbalanced-learn |
| Model | XGBoost, Optuna, MLflow |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Container | Docker |
| Deploy | Render / AWS |
| Monitoring | Streamlit, SciPy |
