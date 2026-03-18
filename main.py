from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import shap
import pandas as pd

app = FastAPI(title="Churn Prediction API", version="1.0.0")
model = joblib.load("model.joblib")
explainer = shap.TreeExplainer(model)


class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    contract_risk: int       # 0=two year, 1=one year, 2=month-to-month
    service_count: int       # 0-8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.dict()])
    prob = float(model.predict_proba(df)[0][1])
    sv = explainer.shap_values(df)[0]
    top_reasons = sorted(
        zip(df.columns, sv), key=lambda x: abs(x[1]), reverse=True
    )[:3]
    return {
        "churn_probability": round(prob, 4),
        "risk_level": "high" if prob > 0.7 else "medium" if prob > 0.4 else "low",
        "top_reasons": [
            {"feature": f, "impact": round(float(v), 4)} for f, v in top_reasons
        ],
    }
