import joblib
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from model_utils import get_proba
sys.path.append(str(Path(__file__).parent))

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
with open(OUTPUT_PATH / f"threshold_{model_name}.json") as f:
    THRESHOLD = json.load(f)["threshold"]

y_pred   = (y_pred_proba >= THRESHOLD).astype(int)
accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_pred_proba)

cm  = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

tvp = tp / (tp + fn)
tfp = fp / (fp + tn)
fnr = fn / (fn + tp)

metrics_summary = {
    "model_name": model_name,
    "auc": round(float(auc), 4),
    "accuracy": round(float(accuracy), 4),
    "threshold": round(float(THRESHOLD), 4),
    "recall": round(float(tvp), 4),
    "fpr": round(float(tfp), 4),
    "fnr": round(float(fnr), 4),
    "confusion_matrix": {
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    },
}

with open(OUTPUT_PATH / f"metrics_{model_name}.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n{'='*60}")
print(f"MÉTRIQUES GLOBALES — {model_name}")
print(f"{'='*60}")
print(f"AUC              : {auc:.4f}")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Taux d'erreur    : {1 - accuracy:.4f}")
print(f"\nSeuil utilisé    : {THRESHOLD:.2f}")

print(f"\n{'='*60}")
print(f"MATRICE DE CONFUSION")
print(f"{'='*60}")
print(f"                 Prédit 0    Prédit 1")
print(f"Réel 0 (non-déf) {tn:>10,}  {fp:>10,}  ← FP")
print(f"Réel 1 (défaut)  {fn:>10,}  {tp:>10,}  ← VP")

print(f"\n{'='*60}")
print(f"MÉTRIQUES CLASSE D'INTÉRÊT (défaut = classe 1)")
print(f"{'='*60}")
print(f"TVP (Recall)     : {tvp:.4f}  → {tvp*100:.1f}% des défauts détectés")
print(f"TFP              : {tfp:.4f}  → {tfp*100:.1f}% des bons clients refusés")
print(f"FNR              : {fnr:.4f}  → {fnr*100:.1f}% des défauts ratés")
print(f"VP               : {tp:,}    défauts bien détectés")
print(f"FN               : {fn:,}    défauts ratés")
print(f"FP               : {fp:,}    bons clients refusés à tort")
print(f"VN               : {tn:,}    bons clients bien acceptés")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Métriques — {model_name}", fontsize=13)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Non-défaut", "Défaut"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Matrice de confusion")

fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr_arr, tpr_arr, color="#2980b9", lw=2, label=f"AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Modèle aléatoire")
axes[1].scatter([tfp], [tvp], color="red", zorder=5, label=f"Seuil {THRESHOLD:.2f}")
axes[1].set_xlabel("TFP (False Positive Rate)")
axes[1].set_ylabel("TVP (True Positive Rate)")
axes[1].set_title("Courbe ROC")
axes[1].legend()
axes[1].grid(alpha=0.3)

precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_pred_proba)
prevalence = y_test.mean()
axes[2].plot(recall_arr, precision_arr, color="#2ecc71", lw=2)
axes[2].axhline(prevalence, color="gray", linestyle="--",
                label=f"Prévalence ({prevalence:.2f})")
axes[2].set_xlabel("Recall (TVP)")
axes[2].set_ylabel("Precision")
axes[2].set_title("Courbe Precision / Recall")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / f"metrics_{model_name}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: metrics_{model_name}.png")
print(f"Saved: metrics_{model_name}.json")