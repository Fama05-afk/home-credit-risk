import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET        = "target"
SENSITIVE_COL = "gender_clean"

ORDINAL_COLS = ["age_group"]
TARGET_COLS  = ["occupation_type", "education_type", "family_status",
                "income_type", "organization_type"]

train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")

X_train = train.drop(columns=[TARGET, SENSITIVE_COL])
y_train = train[TARGET]
X_val   = val.drop(columns=[TARGET, SENSITIVE_COL])
y_val   = val[TARGET]

print(f"Train: {X_train.shape[0]:,} · Val: {X_val.shape[0]:,} · Features: {X_train.shape[1]}")

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(categories=[["young", "adult", "senior"]]), ORDINAL_COLS),
        ("target",  TargetEncoder(cols=TARGET_COLS, smoothing=10), TARGET_COLS),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("imputer",      SimpleImputer(strategy="median")),
    ("scaler",       StandardScaler()),
    ("model",        LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
])

pipeline.fit(X_train, y_train)

auc_train = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
auc_val   = roc_auc_score(y_val,   pipeline.predict_proba(X_val)[:, 1])

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("home_credit_risk")

with mlflow.start_run(run_name="baseline_logistic_regression"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("roc_auc_train", auc_train)
    mlflow.log_metric("roc_auc_val",   auc_val)
    mlflow.sklearn.log_model(pipeline, name="pipeline")
    joblib.dump(pipeline, MODELS_DIR / "baseline_lr.joblib")
    print(f"Train AUC : {auc_train:.4f}")
    print(f"Val   AUC : {auc_val:.4f}")