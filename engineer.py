import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()

    # Fix TotalCharges: ships as object with whitespace for brand-new customers
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
   

    # Binary encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Tenure buckets (-1 lower bound catches tenure=0 new customers)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )
    df["tenure_group"] = df["tenure_group"].astype(str)

    # Charge ratio: monthly vs total (signals recent price changes)
    df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Average monthly spend over lifetime
    df["avg_monthly_charges"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Log-transform right-skewed monetary columns
    df["log_total_charges"] = np.log1p(df["TotalCharges"])
    df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

    # Contract risk: month-to-month = highest churn risk
    df["contract_risk"] = df["Contract"].map({
        "Month-to-month": 2, "One year": 1, "Two year": 0
    })

    # Service count: more services = lower churn
    # Handles "No internet service" / "No phone service" strings correctly
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["service_count"] = df[service_cols].apply(
        lambda row: sum(
            v not in ["No", "No internet service", "No phone service"]
            for v in row
        ), axis=1
    )

    # Drop high-cardinality ID
    df = df.drop(columns=["customerID"], errors="ignore")

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df