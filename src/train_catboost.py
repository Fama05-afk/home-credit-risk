import joblib
import mlflow
import mlflow.catboost
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET        = "target"
SENSITIVE_COL = "gender_clean"

train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")

X_train = train.drop(columns=[TARGET, SENSITIVE_COL])
y_train = train[TARGET]
X_val   = val.drop(columns=[TARGET, SENSITIVE_COL])
y_val   = val[TARGET]

print(f"Train: {X_train.shape[0]:,} · Val: {X_val.shape[0]:,}")

cat_features = X_train.select_dtypes(include="object").columns.tolist()
print(f"Categorical features ({len(cat_features)}): {cat_features}")

for col in cat_features:
    X_train[col] = X_train[col].fillna("Unknown")
    X_val[col]   = X_val[col].fillna("Unknown")

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("home_credit_risk")

params = {
    "iterations":            1000,
    "learning_rate":         0.05,
    "depth":                 6,
    "scale_pos_weight":      3,
    "eval_metric":           "AUC",
    "random_seed":           42,
    "verbose":               100,
    "early_stopping_rounds": 50,
}

with mlflow.start_run(run_name="catboost_baseline"):
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
    )
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    mlflow.log_params(params)
    mlflow.log_metric("roc_auc_val", auc)
    mlflow.log_metric("best_iteration", model.best_iteration_)
    mlflow.catboost.log_model(model, "model")
    joblib.dump(model, MODELS_DIR / "catboost_baseline.joblib")
    print(f"CatBoost ROC-AUC (val): {auc:.4f}  (best iteration: {model.best_iteration_})")