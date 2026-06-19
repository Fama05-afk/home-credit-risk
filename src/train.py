import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
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

X_train_enc = preprocessor.fit_transform(X_train, y_train)
X_val_enc   = preprocessor.transform(X_val)

params = {
    "boosting_type":   "gbdt",
    "objective":       "binary",
    "metric":          "auc",
    "max_depth":       3,
    "num_leaves":      32,
    "learning_rate":   0.05,
    "max_bin":         512,
    "subsample":       0.7,
    "colsample_bytree": 0.8,
    "reg_alpha":       20,
    "reg_lambda":      20,
    "min_split_gain":  0.5,
    "n_estimators":    2500,
    #"n_jobs":          -1,
    "n_jobs": 1,  # au lieu de -1
    "random_state":    42,
    "verbose":         -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train_enc, y_train,
    eval_set=[(X_val_enc, y_val)],

    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model",        model),
])

auc_train = roc_auc_score(y_train, model.predict_proba(X_train_enc)[:, 1])
auc_val   = roc_auc_score(y_val,   model.predict_proba(X_val_enc)[:, 1])

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("home_credit_risk")

with mlflow.start_run(run_name="lightgbm_kaggle_params"):
    mlflow.log_params(params)
    mlflow.log_metric("roc_auc_train",  auc_train)
    mlflow.log_metric("roc_auc_val",    auc_val)
    mlflow.log_metric("best_iteration", model.best_iteration_)
    mlflow.sklearn.log_model(pipeline, name="pipeline")
    joblib.dump(pipeline, MODELS_DIR / "lgbm_baseline.joblib")
    print(f"Train AUC: {auc_train:.4f}")
    print(f"Val   AUC: {auc_val:.4f}")
    print(f"Best iteration: {model.best_iteration_}")