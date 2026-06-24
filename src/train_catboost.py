import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from model_utils import FillNACategorical

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

model = CatBoostClassifier(**params)

pipeline = Pipeline([
    ("fillna", FillNACategorical(cat_features, feature_order=X_train.columns.tolist())),
    ("model",  model),
])

X_val_filled = pipeline.named_steps["fillna"].transform(X_val)

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("home_credit_risk")

with mlflow.start_run(run_name="catboost_baseline"):
    pipeline.fit(
        X_train, y_train,
        model__cat_features=cat_features,
        model__eval_set=(X_val_filled, y_val),
    )
    auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
    mlflow.log_params(params)
    mlflow.log_metric("roc_auc_val", auc)
    mlflow.log_metric("best_iteration", model.best_iteration_)
    mlflow.sklearn.log_model(pipeline, name="pipeline")
    joblib.dump(pipeline, MODELS_DIR / "catboost_baseline.joblib")
    print(f"CatBoost ROC-AUC (val): {auc:.4f}  (best iteration: {model.best_iteration_})")