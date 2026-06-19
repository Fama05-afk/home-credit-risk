import joblib
import shap
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.append(str(Path(__file__).parent))
from model_utils import get_proba

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUT_PATH   = Path("data/references")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TARGET        = "target"
SENSITIVE_COL = "gender_clean"

model_name = sys.argv[1] if len(sys.argv) > 1 else "lgbm_baseline"

test     = pd.read_parquet(PROCESSED_DIR / "test.parquet")
X_test   = test.drop(columns=[TARGET, SENSITIVE_COL])
y_test   = test[TARGET]

model        = joblib.load(MODELS_DIR / f"{model_name}.joblib")
y_pred_proba = get_proba(model, X_test)
print(f"Model: {model_name} — ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Extract LightGBM and preprocessor depending on model type
if isinstance(model, dict):
    # Stacking bundle — SHAP on the LightGBM base model
    lgbm_model   = model["lgbm"].named_steps["model"]
    preprocessor = model["lgbm"].named_steps["preprocessor"]
else:
    # sklearn Pipeline
    lgbm_model   = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

X_test_enc  = preprocessor.transform(X_test)
X_sample    = X_test_enc.sample(n=2000, random_state=42)
explainer   = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_sample)
sv          = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(10, 10))
shap.summary_plot(sv, X_sample, show=False)
plt.title(f"SHAP Feature Importance — {model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
plt.close()

importance_df = (
    pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))
importance_df.to_csv(OUTPUT_PATH / f"shap_importance_{model_name}.csv", index=False)
print(f"Saved: shap_summary_{model_name}.png · shap_importance_{model_name}.csv")