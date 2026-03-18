import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/telco_churn.csv")

def load_raw() -> pd.DataFrame:
    """Load raw Telco churn dataset."""
    df = pd.read_csv(DATA_PATH)
    # TotalCharges is stored as string with spaces — fix silently
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def validate(df: pd.DataFrame) -> None:
    """Assert basic schema expectations."""
    assert "Churn" in df.columns, "Missing target column: Churn"
    assert df["customerID"].nunique() == len(df), "Duplicate customerIDs found"
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if not high_null.empty:
        print(f"Warning: High null rate in {high_null.to_dict()}")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Churn rate: {df['Churn'].map({'Yes': 1, 'No': 0}).mean():.1%}")
