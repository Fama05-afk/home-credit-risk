# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import sys

# from pathlib import Path
# from sklearn.metrics import (
#     roc_auc_score, f1_score, precision_score,
#     recall_score, classification_report
# )

# PROCESSED_DIR = Path("data/processed")
# MODELS_DIR    = Path("models")
# OUTPUT_PATH   = Path("data/references")


# def get_proba(model_obj, X_enc, X_raw):
#     if isinstance(model_obj, dict):
#         cat_features = model_obj["cat_features"]
#         for col in cat_features:
#             X_raw[col] = X_raw[col].fillna("Unknown")
#         p_lgbm = model_obj["lgbm"].predict_proba(X_enc)[:, 1]
#         p_cb   = model_obj["catboost"].predict_proba(X_raw)[:, 1]
#         meta   = np.column_stack([p_lgbm, p_cb])
#         return model_obj["meta"].predict_proba(meta)[:, 1]
#     return model_obj.predict_proba(X_enc)[:, 1]

# model_name = sys.argv[1] if len(sys.argv) > 1 else "lgbm_baseline"
# X_test_raw = pd.read_parquet(PROCESSED_DIR / "X_test_raw.parquet")
# X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
# y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
# model  = joblib.load(MODELS_DIR / f"{model_name}.joblib")

# y_pred_proba = get_proba(model, X_test, X_test_raw)

# # Test all thresholds between 0.01 and 0.99
# thresholds = np.arange(0.01, 0.99, 0.01)
# results = []

# for t in thresholds:
#     y_pred = (y_pred_proba >= t).astype(int)
#     results.append({
#         "threshold":  t,
#         "f1":         f1_score(y_test, y_pred, zero_division=0),
#         "precision":  precision_score(y_test, y_pred, zero_division=0),
#         "recall":     recall_score(y_test, y_pred, zero_division=0),
#         "youden_j":   recall_score(y_test, y_pred) +
#                       recall_score(y_test, y_pred, pos_label=0) - 1, 
#     })

# results_df = pd.DataFrame(results)

# best_f1      = results_df.loc[results_df["f1"].idxmax()]
# best_youden  = results_df.loc[results_df["youden_j"].idxmax()]

# print(f"Default threshold (0.5):")
# default = results_df[results_df["threshold"].round(2) == 0.50].iloc[0]
# print(f"  F1={default['f1']:.4f}  Precision={default['precision']:.4f}  Recall={default['recall']:.4f}")

# print(f"\nBest F1 threshold: {best_f1['threshold']:.2f}")
# print(f"  F1={best_f1['f1']:.4f}  Precision={best_f1['precision']:.4f}  Recall={best_f1['recall']:.4f}")

# print(f"\nBest Youden's J threshold: {best_youden['threshold']:.2f}")
# print(f"  F1={best_youden['f1']:.4f}  Precision={best_youden['precision']:.4f}  Recall={best_youden['recall']:.4f}")

# print(f"\nClassification report at best F1 threshold ({best_f1['threshold']:.2f}):")
# print(classification_report(y_test, (y_pred_proba >= best_f1['threshold']).astype(int)))

# # Plot
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# axes[0].plot(results_df["threshold"], results_df["f1"], label="F1", color="#2ecc71")
# axes[0].plot(results_df["threshold"], results_df["precision"], label="Precision", color="#3498db")
# axes[0].plot(results_df["threshold"], results_df["recall"], label="Recall", color="#e74c3c")
# axes[0].axvline(best_f1["threshold"], color="black", linestyle="--", label=f"Best F1 ({best_f1['threshold']:.2f})")
# axes[0].set_title("Precision / Recall / F1 vs Threshold")
# axes[0].set_xlabel("Threshold")
# axes[0].legend()
# axes[0].grid(alpha=0.3)

# axes[1].plot(results_df["threshold"], results_df["youden_j"], color="#9b59b6")
# axes[1].axvline(best_youden["threshold"], color="black", linestyle="--", label=f"Best Youden's J ({best_youden['threshold']:.2f})")
# axes[1].set_title("Youden's J vs Threshold")
# axes[1].set_xlabel("Threshold")
# axes[1].legend()
# axes[1].grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig(OUTPUT_PATH / "threshold_optimization.png", dpi=150, bbox_inches="tight")
# plt.close()
# print(f"\nSaved: {OUTPUT_PATH}/threshold_optimization.png")

# with open(OUTPUT_PATH / f"threshold_{model_name}.json", "w") as f:
#     json.dump({"threshold": float(best_f1['threshold']), "model": model_name}, f)
# print(f"Threshold saved: {best_f1['threshold']:.2f}")





import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
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

sys.path.append(str(Path(__file__).parent))

model        = joblib.load(MODELS_DIR / f"{model_name}.joblib")
y_pred_proba = get_proba(model, X_test)

thresholds = np.arange(0.01, 0.99, 0.01)
results = []

for t in thresholds:
    y_pred = (y_pred_proba >= t).astype(int)
    results.append({
        "threshold": t,
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "youden_j":  recall_score(y_test, y_pred) +
                     recall_score(y_test, y_pred, pos_label=0) - 1,
    })

results_df  = pd.DataFrame(results)
best_f1     = results_df.loc[results_df["f1"].idxmax()]
best_youden = results_df.loc[results_df["youden_j"].idxmax()]

default = results_df[results_df["threshold"].round(2) == 0.50].iloc[0]
print(f"Default threshold (0.5):")
print(f"  F1={default['f1']:.4f}  Precision={default['precision']:.4f}  Recall={default['recall']:.4f}")

print(f"\nBest F1 threshold: {best_f1['threshold']:.2f}")
print(f"  F1={best_f1['f1']:.4f}  Precision={best_f1['precision']:.4f}  Recall={best_f1['recall']:.4f}")

print(f"\nBest Youden's J threshold: {best_youden['threshold']:.2f}")
print(f"  F1={best_youden['f1']:.4f}  Precision={best_youden['precision']:.4f}  Recall={best_youden['recall']:.4f}")

print(f"\nClassification report at best F1 threshold ({best_f1['threshold']:.2f}):")
print(classification_report(y_test, (y_pred_proba >= best_f1["threshold"]).astype(int)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(results_df["threshold"], results_df["f1"],        label="F1",        color="#2ecc71")
axes[0].plot(results_df["threshold"], results_df["precision"],  label="Precision",  color="#3498db")
axes[0].plot(results_df["threshold"], results_df["recall"],     label="Recall",     color="#e74c3c")
axes[0].axvline(best_f1["threshold"], color="black", linestyle="--",
                label=f"Best F1 ({best_f1['threshold']:.2f})")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].set_xlabel("Threshold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(results_df["threshold"], results_df["youden_j"], color="#9b59b6")
axes[1].axvline(best_youden["threshold"], color="black", linestyle="--",
                label=f"Best Youden's J ({best_youden['threshold']:.2f})")
axes[1].set_title("Youden's J vs Threshold")
axes[1].set_xlabel("Threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "threshold_optimization.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: threshold_optimization.png")

with open(OUTPUT_PATH / f"threshold_{model_name}.json", "w") as f:
    json.dump({"threshold": float(best_f1["threshold"]), "model": model_name}, f)
print(f"Threshold saved: {best_f1['threshold']:.2f}")