import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()

    # Binary encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Tenure buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )

    # Charge ratio: monthly vs total (signals recent price changes)
    df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Contract risk: month-to-month = highest churn risk
    df["contract_risk"] = df["Contract"].map({
        "Month-to-month": 2, "One year": 1, "Two year": 0
    })

    # Service count: more services = lower churn
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
    df = df.drop(columns=["customerID"])

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
