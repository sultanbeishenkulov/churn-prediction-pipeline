import shap
import matplotlib.pyplot as plt

def get_explainer(model):
    """Create a SHAP TreeExplainer for XGBoost."""
    return shap.TreeExplainer(model)

def plot_summary(explainer, X_test, save_path="shap_summary.png"):
    """Global feature importance — top features ranked by mean |SHAP|."""
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")

def explain_customer(explainer, X_test, idx=0):
    """Waterfall plot explaining a single customer prediction."""
    sv = explainer(X_test.iloc[[idx]])
    shap.waterfall_plot(sv[0])
