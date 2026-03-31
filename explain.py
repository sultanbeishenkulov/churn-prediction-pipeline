import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from load import load_raw
from engineer import engineer_features

MODEL_DIR = Path("models")
PLOTS_DIR = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load trained model and feature names."""
    model = joblib.load(MODEL_DIR / "model.joblib")
    features = (MODEL_DIR / "feature_names.txt").read_text().strip().split("\n")
    return model, features


def get_explainer(model):
    """Create a SHAP TreeExplainer for XGBoost."""
    return shap.TreeExplainer(model)


def plot_summary(explainer, X_test, save_path=None):
    """Global feature importance — top features ranked by mean |SHAP|."""
    if save_path is None:
        save_path = PLOTS_DIR / "shap_summary.png"

    shap_values = explainer.shap_values(X_test)

    # Save raw SHAP values for API / dashboard reuse
    np.save(MODEL_DIR / "shap_values.npy", shap_values)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")

    return shap_values


def explain_customer(explainer, X_test, idx=0, save_path=None):
    """Waterfall plot explaining a single customer prediction."""
    if save_path is None:
        save_path = PLOTS_DIR / f"shap_waterfall_idx{idx}.png"

    sv = explainer(X_test.iloc[[idx]])
    shap.waterfall_plot(sv[0], show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall plot saved to {save_path}")

    return sv


def main():
    """Run full SHAP explainability pipeline."""
    from sklearn.model_selection import train_test_split

    # Load data + features (same pipeline as training)
    df = load_raw()
    df = engineer_features(df)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Load trained model and align columns
    model, feature_names = load_model()
    X = X[feature_names]

    # Same split as training for consistent test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    explainer = get_explainer(model)

    # Global summary
    shap_values = plot_summary(explainer, X_test)

    # Waterfall for highest-risk customer
    y_prob = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(y_prob))
    explain_customer(explainer, X_test, idx=high_risk_idx)

    # Print top global drivers
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:5]
    print("\nTop 5 churn drivers (mean |SHAP|):")
    for i in top_idx:
        print(f"  {feature_names[i]:.<30} {mean_abs[i]:.4f}")


if __name__ == "__main__":
    main()

