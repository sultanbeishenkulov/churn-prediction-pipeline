# Customer Churn Prediction Pipeline

Predicts customer churn on the IBM Telco dataset. Full pipeline from raw data to a deployed API with SHAP explainability and drift monitoring.

**Live API:** [churn-prediction-pipeline-1.onrender.com](https://churn-prediction-pipeline-1.onrender.com)

---

## Pipeline
```
Raw CSV → Feature Engineering → XGBoost + Optuna → SHAP Explainability
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

Top churn drivers (SHAP): `contract_risk` > `tenure` > `MonthlyCharges` > internet service type.

---

## Project Structure
```
churn-prediction-pipeline/
├── load.py              # Data loading + validation
├── engineer.py          # Feature engineering
├── train.py             # XGBoost + Optuna + MLflow
├── explain.py           # SHAP summary + waterfall plots
├── main.py              # FastAPI /predict endpoint
├── dashboard.py         # Streamlit monitoring dashboard
├── drift.py             # PSI drift detection
├── Dockerfile
├── requirements.txt
├── data/
│   └── raw/             # IBM Telco dataset (gitignored)
├── models/              # Trained artifacts (gitignored)
└── notebooks/           # EDA
```

Everything lives in the root — no `src/` nesting. Each stage runs standalone (`python train.py`, `python explain.py`, etc).

---

## Quickstart
```bash
git clone https://github.com/sultanbeishenkulov/churn-prediction-pipeline
cd churn-prediction-pipeline
pip install -r requirements.txt

# grab the dataset from Kaggle and drop it in data/raw/

python train.py
python explain.py
uvicorn main:app --reload
streamlit run dashboard.py
```

---

## API Usage
```bash
curl -X POST https://churn-prediction-pipeline-1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
    "contract_risk": 2,
    "service_count": 3
  }'
```
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

Swagger UI at [`/docs`](https://churn-prediction-pipeline-1.onrender.com/docs).

---

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The model trains during `docker build` (the Dockerfile runs `python train.py`). I commit the raw CSV but not the model binaries — so every build trains from scratch and there's no `.pkl` files cluttering up git history.

---

## Why I made certain choices

**Train at build time** — I tried committing the model binary early on and it was a pain. The file kept getting stale relative to the code, and diffs were useless. Training during the Docker build keeps everything in sync and the repo stays clean.

**SHAP over built-in feature importance** — XGBoost's `feature_importances_` gave me different rankings across runs depending on random seed. SHAP is deterministic for a given model and shows direction, not just magnitude. That matters when you're trying to tell a stakeholder *why* someone is high risk.

**PSI for drift detection** — Went with Population Stability Index since it's what I've seen used in production at scale. PSI > 0.2 flags a retrain. Simple and well-understood.

I used `scale_pos_weight` instead of SMOTE for class imbalance (~26% churn). SMOTE created noisy synthetic samples that inflated recall without actually improving the model.

**Flat layout** — I started with a nested `src/` structure and kept running into import issues. Flat modules in the root are easier to work with and honestly easier for someone reviewing the repo to navigate.

---

## What I'd do differently

- Add CI/CD with GitHub Actions to auto-deploy on push
- Swap MLflow tracking for something lighter — I'm only using it for local experiment logging and it's overkill
- Add integration tests for the `/predict` endpoint
<!-- TODO: add a screenshot of the Streamlit dashboard -->

---

## Stack

| Layer | Tool |
|---|---|
| Data | Pandas |
| Features | Scikit-learn |
| Model | XGBoost, Optuna, MLflow |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Monitoring | Streamlit, SciPy (PSI) |
| Container | Docker |
| Deploy | Render |