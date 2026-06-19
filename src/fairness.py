import joblib
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
import json
from model_utils import get_proba

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUT_PATH   = Path("data/references")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TARGET        = "target"
SENSITIVE_COL = "gender_clean"

model_name = sys.argv[1] if len(sys.argv) > 1 else "lgbm_baseline"

test     = pd.read_parquet(PROCESSED_DIR / "test.parquet")
mask_known = test[SENSITIVE_COL].isin(["F", "M"])
test     = test[mask_known]
X_test   = test.drop(columns=[TARGET, SENSITIVE_COL])
y_test   = test[TARGET]
gender   = test[SENSITIVE_COL]

sys.path.append(str(Path(__file__).parent))

model        = joblib.load(MODELS_DIR / f"{model_name}.joblib")
y_pred_proba = get_proba(model, X_test)

with open(OUTPUT_PATH / f"threshold_{model_name}.json") as f:
    THRESHOLD = json.load(f)["threshold"]

y_pred = (y_pred_proba >= THRESHOLD).astype(int)
print(f"Model: {model_name} — ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

metric_frame = MetricFrame(
    metrics={
        "selection_rate":      selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=gender
)

print(f"\nMetrics by gender:\n{metric_frame.by_group.round(4)}")
print(f"\nDifference (max - min):\n{metric_frame.difference().round(4)}")

print("\nROC-AUC by gender:")
for g in sorted(gender.unique()):
    mask = gender == g
    if mask.sum() < 10:
        continue
    print(f"  {g}: AUC={roc_auc_score(y_test[mask], y_pred_proba[mask]):.4f}  (n={mask.sum():,})")


auc_by_gender = {}
for g in sorted(gender.unique()):
    mask = gender == g
    if mask.sum() < 10:
        continue
    auc_by_gender[g] = round(float(roc_auc_score(y_test[mask], y_pred_proba[mask])), 4)

fairness_summary = {
    "model_name": model_name,
    "by_group": metric_frame.by_group.round(4).to_dict(),
    "difference": metric_frame.difference().round(4).to_dict(),
    "auc_by_gender": auc_by_gender,
}

with open(OUTPUT_PATH / f"fairness_{model_name}.json", "w") as f:
    json.dump(fairness_summary, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"Fairness Analysis — {model_name}", fontsize=13)
for ax, metric, color in zip(axes,
    ["selection_rate", "false_positive_rate", "false_negative_rate"],
    ["#c0392b", "#e67e22", "#2980b9"]):
    metric_frame.by_group[metric].plot(kind="bar", ax=ax, color=color, alpha=0.8, edgecolor="black")
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_xticklabels(metric_frame.by_group.index, rotation=0)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / f"fairness_{model_name}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: fairness_{model_name}.png")
print(f"Saved: fairness_{model_name}.json")