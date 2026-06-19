import joblib
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

TARGET        = "target"
SENSITIVE_COL = "gender_clean"

ORDINAL_COLS = ["age_group"]
TARGET_COLS  = ["occupation_type", "education_type", "family_status",
                "income_type", "organization_type"]

train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")

X_train = train.drop(columns=[TARGET, SENSITIVE_COL]).reset_index(drop=True)
y_train = train[TARGET].reset_index(drop=True)
X_val   = val.drop(columns=[TARGET, SENSITIVE_COL])
y_val   = val[TARGET]

print(f"Train: {X_train.shape[0]:,} · Val: {X_val.shape[0]:,}")

cat_features = X_train.select_dtypes(include="object").columns.tolist()
print(f"CatBoost categorical features ({len(cat_features)}): {cat_features}")

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("ordinal", OrdinalEncoder(categories=[["young", "adult", "senior"]]), ORDINAL_COLS),
            ("target",  TargetEncoder(cols=TARGET_COLS, smoothing=10), TARGET_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

lgbm_params = {
    "objective":         "binary",
    "metric":            "auc",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 20,
    "scale_pos_weight":  3,
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
}

catboost_params = {
    "iterations":            500,
    "learning_rate":         0.05,
    "depth":                 6,
    "scale_pos_weight":      3,
    "eval_metric":           "AUC",
    "random_seed":           42,
    "verbose":               0,
    "early_stopping_rounds": 50,
}

N_SPLITS     = 5
skf          = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_lgbm     = np.zeros(len(X_train))
oof_catboost = np.zeros(len(X_train))

print(f"Starting {N_SPLITS}-fold OOF stacking...")

for fold, (train_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

    X_fold = X_train.iloc[train_idx]
    y_fold = y_train.iloc[train_idx]
    X_oof  = X_train.iloc[oof_idx]
    y_oof  = y_train.iloc[oof_idx]

    es_idx_local = np.arange(len(X_fold))
    fit_idx, es_idx = train_test_split(
        es_idx_local, test_size=0.1, random_state=42, stratify=y_fold
    )

    X_fit = X_fold.iloc[fit_idx]
    X_es  = X_fold.iloc[es_idx]
    y_fit = y_fold.iloc[fit_idx]
    y_es  = y_fold.iloc[es_idx]

    # LightGBM — preprocessing inside fold (no leakage)
    preprocessor = build_preprocessor()
    X_fit_enc = preprocessor.fit_transform(X_fit, y_fit)
    X_es_enc  = preprocessor.transform(X_es)
    X_oof_enc = preprocessor.transform(X_oof)

    lgbm_fold = lgb.LGBMClassifier(**lgbm_params)
    lgbm_fold.fit(
        X_fit_enc, y_fit,
        eval_set=[(X_es_enc, y_es)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgbm[oof_idx] = lgbm_fold.predict_proba(X_oof_enc)[:, 1]

    # CatBoost — takes raw data, handles categoricals natively
    X_fit_raw = X_fit.copy()
    X_es_raw  = X_es.copy()
    X_oof_raw = X_oof.copy()
    for col in cat_features:
        X_fit_raw[col] = X_fit_raw[col].fillna("Unknown")
        X_es_raw[col]  = X_es_raw[col].fillna("Unknown")
        X_oof_raw[col] = X_oof_raw[col].fillna("Unknown")

    cb_fold = CatBoostClassifier(**catboost_params)
    cb_fold.fit(
        X_fit_raw, y_fit,
        cat_features=cat_features,
        eval_set=(X_es_raw, y_es),
    )
    oof_catboost[oof_idx] = cb_fold.predict_proba(X_oof_raw)[:, 1]

    print(f"  LightGBM OOF AUC : {roc_auc_score(y_oof, oof_lgbm[oof_idx]):.4f}")
    print(f"  CatBoost OOF AUC : {roc_auc_score(y_oof, oof_catboost[oof_idx]):.4f}")

oof_auc_lgbm     = roc_auc_score(y_train, oof_lgbm)
oof_auc_catboost = roc_auc_score(y_train, oof_catboost)
print(f"\nOOF global LightGBM  : {oof_auc_lgbm:.4f}")
print(f"OOF global CatBoost  : {oof_auc_catboost:.4f}")

meta_train = np.column_stack([oof_lgbm, oof_catboost])
meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_train, y_train)

# Val evaluation — use trained base models
lgbm_pipeline  = joblib.load(MODELS_DIR / "lgbm_baseline.joblib")
catboost_model = joblib.load(MODELS_DIR / "catboost_baseline.joblib")

X_val_raw = X_val.copy()
for col in cat_features:
    X_val_raw[col] = X_val_raw[col].fillna("Unknown")

val_lgbm     = lgbm_pipeline.predict_proba(X_val)[:, 1]
val_catboost = catboost_model.predict_proba(X_val_raw)[:, 1]
meta_val     = np.column_stack([val_lgbm, val_catboost])
val_stacking = meta_model.predict_proba(meta_val)[:, 1]

auc_val = roc_auc_score(y_val, val_stacking)
print(f"Stacking AUC (val)   : {auc_val:.4f}")

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("home_credit_risk")

with mlflow.start_run(run_name="stacking_lgbm_catboost"):
    mlflow.log_param("base_models", "lgbm_baseline + catboost_baseline")
    mlflow.log_param("meta_model",  "LogisticRegression")
    mlflow.log_param("n_splits",    N_SPLITS)
    mlflow.log_metric("oof_auc_lgbm",     oof_auc_lgbm)
    mlflow.log_metric("oof_auc_catboost", oof_auc_catboost)
    mlflow.log_metric("roc_auc_val",      auc_val)

bundle = {
    "lgbm":         lgbm_pipeline,
    "catboost":     catboost_model,
    "meta":         meta_model,
    "cat_features": cat_features,
}
joblib.dump(bundle, MODELS_DIR / "stacking_model.joblib")
print(f"Stacking model saved → models/stacking_model.joblib")