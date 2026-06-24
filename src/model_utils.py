import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class FillNACategorical(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features):
        self.cat_features = cat_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].fillna("Unknown")
        return X


class StackingClassifier(BaseEstimator, ClassifierMixin):
    """Combines two pre-fitted pipelines via a meta-model on their OOF probabilities"""

    def __init__(self, lgbm_pipeline, catboost_pipeline, meta_model):
        self.lgbm_pipeline      = lgbm_pipeline
        self.catboost_pipeline  = catboost_pipeline
        self.meta_model         = meta_model

    def fit(self, X, y):
        return self  # base models and meta_model are already fitted upstream

    def predict_proba(self, X):
        p_lgbm     = self.lgbm_pipeline.predict_proba(X)[:, 1]
        p_catboost = self.catboost_pipeline.predict_proba(X)[:, 1]
        meta_X     = np.column_stack([p_lgbm, p_catboost])
        return self.meta_model.predict_proba(meta_X)

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5


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