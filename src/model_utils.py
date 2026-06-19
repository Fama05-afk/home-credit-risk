import numpy as np
import pandas as pd


def get_proba(model, X_test):
    if isinstance(model, dict):
        X_raw = X_test.copy()
        for col in model["cat_features"]:
            X_raw[col] = X_raw[col].fillna("Unknown")
        p_lgbm = model["lgbm"].predict_proba(X_test)[:, 1]
        p_cb   = model["catboost"].predict_proba(X_raw)[:, 1]
        meta   = np.column_stack([p_lgbm, p_cb])
        return model["meta"].predict_proba(meta)[:, 1]
    return model.predict_proba(X_test)[:, 1]


def load_model_and_predict(model, X_test):
    return get_proba(model, X_test)