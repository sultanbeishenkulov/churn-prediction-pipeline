# churn-prediction-pipeline

End-to-end customer churn prediction system — from raw data to a live API with explainability and drift monitoring.

Built on the IBM Telco Churn dataset. Covers the full ML lifecycle: feature engineering, XGBoost training, SHAP explainability, FastAPI deployment, and a Streamlit monitoring dashboard.

---

## Pipeline overview

```
Raw data → Feature engineering → XGBoost + Optuna → SHAP explainability
                                                           ↓
                                              FastAPI /predict endpoint
                                                           ↓
                                         Streamlit monitoring dashboard
                                                           ↓
                                           PSI drift detection → retrain
```

---

## Results

| Metric | Score |
|---|---|
| AUC-ROC | ~0.85 |
| Precision (churn) | ~0.67 |
| Recall (churn) | ~0.78 |

Top churn drivers (SHAP): contract type, tenure, monthly charges, internet service.

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

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/churn-prediction-pipeline
cd churn-prediction-pipeline
pip install -r requirements.txt

# 2. Download dataset
# Place telco_churn.csv in data/raw/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 3. Train model
python -m src.models.train

# 4. Run API locally
uvicorn src.api.main:app --reload

# 5. Run monitoring dashboard
streamlit run src/monitoring/dashboard.py
```

---

## API usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 89.5, "TotalCharges": 179.0, "contract_risk": 2, "service_count": 3}'
```

Response:
```json
{
  "churn_probability": 0.847,
  "risk_level": "high",
  "top_reasons": [
    {"feature": "contract_risk", "impact": 0.412},
    {"feature": "tenure", "impact": -0.287},
    {"feature": "MonthlyCharges", "impact": 0.198}
  ]
}
```

---

## Docker

```bash
docker build -f docker/Dockerfile -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## Key design decisions

**SHAP over feature importance** — XGBoost's built-in feature importance is unstable across runs and doesn't show directionality. SHAP values are consistent, explain individual predictions, and map directly to business narratives ("this customer is high risk because they're on a month-to-month contract with high monthly charges and only 2 months of tenure").

**PSI for drift** — Population Stability Index is the industry standard for detecting input distribution shift. Threshold: PSI > 0.2 triggers a retrain alert.

**scale_pos_weight over SMOTE** — Telco churn is ~26% positive. Adjusting the class weight inside XGBoost is faster, less likely to overfit, and avoids synthetic data artifacts.

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
