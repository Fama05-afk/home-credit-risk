import joblib
import optuna
import mlflow
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_AUC  = 0.7818
N_TRIALS      = 25
N_SPLITS      = 5
SEED          = 42

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

print(f"Train: {X_train.shape[0]:,} · Val: {X_val.shape[0]:,}")

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("home_credit_risk")


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("ordinal", OrdinalEncoder(categories=[["young", "adult", "senior"]]), ORDINAL_COLS),
            ("target",  TargetEncoder(cols=TARGET_COLS, smoothing=10), TARGET_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def objective(trial):
    params = {
        "objective":          "binary",
        "metric":             "auc",
        "n_estimators":       1000,
        "learning_rate":      0.05,
        "random_state":       SEED,
        "verbose":            -1,
        "n_jobs":             -1,
        "scale_pos_weight":   3,  
        "num_leaves":         trial.suggest_int("num_leaves", 50, 200),
        "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction":   trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":   trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":       trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1":          trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
        "lambda_l2":          trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
    }

    print(f"\n>>> Trial {trial.number}: num_leaves={params['num_leaves']}, "
          f"lr={params['learning_rate']:.4f}, "
          f"feat_frac={params['feature_fraction']:.3f}, "
          f"scale_pos={params['scale_pos_weight']}")


    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_aucs, fold_iters = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        preprocessor = build_preprocessor()
        X_tr_enc = preprocessor.fit_transform(X_tr, y_tr)
        X_va_enc = preprocessor.transform(X_va)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_va_enc, y_va)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = model.predict_proba(X_va_enc)[:, 1]
        fold_auc = roc_auc_score(y_va, preds)
        fold_aucs.append(fold_auc)
        fold_iters.append(model.best_iteration_)

        intermediate_value = float(np.mean(fold_aucs))
        trial.report(intermediate_value, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("std_auc",        float(np.std(fold_aucs)))
    trial.set_user_attr("mean_best_iter", int(np.mean(fold_iters)))

    return float(np.mean(fold_aucs))


print(f"\nStarting Optuna ({N_TRIALS} trials, {N_SPLITS}-fold CV, with pruning)...\n")

with mlflow.start_run(run_name="optuna_lgbm_pruned"):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nCompleted: {len(completed)} · Pruned: {len(pruned)}")

    best_params = study.best_params
    best_auc    = study.best_value
    best_std    = study.best_trial.user_attrs["std_auc"]
    best_iter   = study.best_trial.user_attrs["mean_best_iter"]

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_auc",         best_auc)
    mlflow.log_metric("best_cv_std",         best_std)
    mlflow.log_metric("mean_best_iteration", best_iter)
    mlflow.log_metric("n_completed_trials",  len(completed))
    mlflow.log_metric("n_pruned_trials",     len(pruned))

print(f"\nBest CV AUC : {best_auc:.4f} ± {best_std:.4f}")
print(f"Baseline    : {BASELINE_AUC:.4f}")
print(f"Gain        : {best_auc - BASELINE_AUC:+.4f}")
print(f"Mean best_iter : {best_iter}")
print(f"\nBest params : {best_params}")

final_params = {
    "objective":    "binary",
    "metric":        "auc",
    "n_estimators":  int(best_iter * 1.1),
    "random_state":  SEED,
    "verbose":       -1,
    "n_jobs":        -1,
    **best_params,
}

preprocessor_final = build_preprocessor()
X_train_enc = preprocessor_final.fit_transform(X_train, y_train)
X_val_enc   = preprocessor_final.transform(X_val)

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X_train_enc, y_train)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor_final),
    ("model",        final_model),
])

auc_val = roc_auc_score(y_val, final_model.predict_proba(X_val_enc)[:, 1])

with mlflow.start_run(run_name="lgbm_tuned_pipeline"):
    mlflow.log_params(final_params)
    mlflow.log_metric("roc_auc_val", auc_val)
    mlflow.log_metric("cv_auc_mean", best_auc)
    mlflow.log_metric("cv_auc_std",  best_std)
    mlflow.sklearn.log_model(final_pipeline, name="pipeline")

joblib.dump(final_pipeline, MODELS_DIR / "lgbm_tuned_pipeline.joblib")

print(f"\nFinal model AUC val : {auc_val:.4f}")
print(f"Saved → models/lgbm_tuned_pipeline.joblib")