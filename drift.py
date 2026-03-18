import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    < 0.1  = no significant change
    0.1–0.2 = moderate change, monitor
    > 0.2  = significant drift, retrain
    """
    def bucket_dist(arr, bins):
        counts, _ = np.histogram(arr, bins=bins)
        return counts / len(arr) + 1e-10  # avoid log(0)

    bins = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    bins[0] -= 1e-10
    bins[-1] += 1e-10

    exp_dist = bucket_dist(expected, bins)
    act_dist = bucket_dist(actual, bins)
    return round(float(np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))), 4)


def check_drift(train_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    """Check PSI for all numeric features. Returns per-feature status."""
    results = {}
    for col in train_df.select_dtypes(include="number").columns:
        score = psi(train_df[col].dropna().values, live_df[col].dropna().values)
        results[col] = {
            "psi": score,
            "status": "drift" if score > 0.2 else "monitor" if score > 0.1 else "stable"
        }
    return results
