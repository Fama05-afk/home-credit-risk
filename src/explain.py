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


is_stacking = hasattr(model, "lgbm_pipeline")
if is_stacking:
    # SHAP on the LightGBM base model only
    tree_model         = model.lgbm_pipeline.steps[-1][1]
    preprocessing_steps = model.lgbm_pipeline.steps[:-1]
else:
    # last step is always the model, regardless of step names
    tree_model         = model.steps[-1][1]
    preprocessing_steps = model.steps[:-1]

X_test_enc = X_test
for _, step in preprocessing_steps:
    X_test_enc = step.transform(X_test_enc)

X_sample    = X_test_enc.sample(n=2000, random_state=42)
explainer   = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_sample)
sv          = shap_values[1] if isinstance(shap_values, list) else shap_values

plot_title = f"SHAP Feature Importance — {model_name} (LightGBM component)" if is_stacking \
    else f"SHAP Feature Importance — {model_name}"

plt.figure(figsize=(10, 10))
shap.summary_plot(sv, X_sample, show=False)
plt.title(plot_title)
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