from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path

from engineer import engineer_features

app = FastAPI(title="Churn Prediction API", version="1.0.0")

MODEL_DIR = Path("models")
model = joblib.load(MODEL_DIR / "model.joblib")
feature_names = (MODEL_DIR / "feature_names.txt").read_text().strip().split("\n")
explainer = shap.TreeExplainer(model)


class CustomerFeatures(BaseModel):
    gender: str                          # "Male" or "Female"
    SeniorCitizen: int                   # 0 or 1
    Partner: str                         # "Yes" or "No"
    Dependents: str                      # "Yes" or "No"
    tenure: int
    PhoneService: str                    # "Yes" or "No"
    MultipleLines: str                   # "Yes", "No", "No phone service"
    InternetService: str                 # "DSL", "Fiber optic", "No"
    OnlineSecurity: str                  # "Yes", "No", "No internet service"
    OnlineBackup: str                    # "Yes", "No", "No internet service"
    DeviceProtection: str                # "Yes", "No", "No internet service"
    TechSupport: str                     # "Yes", "No", "No internet service"
    StreamingTV: str                     # "Yes", "No", "No internet service"
    StreamingMovies: str                 # "Yes", "No", "No internet service"
    Contract: str                        # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str                # "Yes" or "No"
    PaymentMethod: str                   # "Electronic check", "Mailed check", etc.
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    # Build a DataFrame that looks like raw data (minus customerID and Churn)
    raw = customer.dict()
    raw["customerID"] = "API_CALL"
    raw["Churn"] = "No"  # Placeholder — gets encoded then dropped

    df = pd.DataFrame([raw])
    df = engineer_features(df)

    # Align columns to what the model expects
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    prob = float(model.predict_proba(df)[0][1])

    sv = explainer.shap_values(df)[0]
    top_reasons = sorted(
        zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True
    )[:3]

    return {
        "churn_probability": round(prob, 4),
        "risk_level": "high" if prob > 0.7 else "medium" if prob > 0.4 else "low",
        "top_reasons": [
            {"feature": f, "impact": round(float(v), 4)} for f, v in top_reasons
        ],
    }