import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

MODELS_DIR    = Path("models")
REFERENCES    = Path("data/references")
MODEL_NAME    = "stacking_model"
SENSITIVE_COL = "gender_clean"

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Predicts loan default probability — Stacking LightGBM + CatBoost",
    version="2.0.0",
)

bundle       = joblib.load(MODELS_DIR / f"{MODEL_NAME}.joblib")
threshold    = json.load(open(REFERENCES / f"threshold_{MODEL_NAME}.json"))["threshold"]
cat_features = bundle["cat_features"]

print(f"Model loaded — threshold: {threshold}")


class LoanApplication(BaseModel):
    income_amount:            float
    credit_amount:            float
    annuity_amount:           float
    age_years:                float
    days_employed:            Optional[float] = None
    years_employed:           Optional[float] = None
    is_employed:              int = 0
    is_working:               int = 0
    is_pensioner:             int = 0
    is_unemployed:            int = 0
    owns_car:                 int = 0
    owns_realty:              int = 0
    own_car_age:              Optional[float] = None
    cnt_children:             float = 0
    cnt_fam_members:          float = 1
    age_group:                str = "adult"
    gender_clean:             Optional[str] = None
    income_type:              str = "Working"
    occupation_type:          Optional[str] = None
    education_type:           str = "Secondary / secondary special"
    family_status:            str = "Married"
    organization_type:        str = "Business Entity Type 3"
    region_rating:            float = 2
    reg_city_not_work_city:   int = 0
    flag_document_3:          int = 0
    days_id_publish:          float = 0
    days_registration:        float = 0
    days_last_phone_change:   Optional[float] = None
    phone_to_birth_ratio:     Optional[float] = None
    ext_source_1:             Optional[float] = None
    ext_source_2:             Optional[float] = None
    ext_source_3:             Optional[float] = None
    ext_source_mean:          Optional[float] = None
    ext_source_1_available:   int = 0
    ratio_income_credit:      Optional[float] = None
    ratio_annuity_income:     Optional[float] = None
    credit_to_annuity_ratio:  Optional[float] = None
    credit_to_goods_ratio:    Optional[float] = None
    income_per_family_member: Optional[float] = None
    income_per_child:         Optional[float] = None
    employed_to_birth_ratio:  Optional[float] = None
    obs_30_social_circle:     float = 0
    def_30_social_circle:     float = 0
    obs_60_social_circle:     float = 0
    def_60_social_circle:     float = 0
    def_30_social_ratio:      float = 0
    def_60_social_ratio:      float = 0
    req_bureau_hour:          float = 0
    req_bureau_day:           float = 0
    req_bureau_week:          float = 0
    req_bureau_mon:           float = 0
    req_bureau_qrt:           float = 0
    req_bureau_year:          float = 0
    req_bureau_recent:        float = 0
    bureau_credit_count:      float = 0
    bureau_active_count:      float = 0
    bureau_debt_total:        float = 0
    bureau_overdue_max:       float = 0
    debt_credit_ratio:        float = 0
    overdue_debt_ratio:       float = 0
    loan_types_bureau:        float = 0
    bb_bad_months:            float = 0
    bb_bad_months_ratio:      float = 0
    bb_had_severe_overdue:    float = 0
    inst_late_avg:            float = 0
    inst_late_ratio:          float = 0
    inst_payment_ratio:       float = 1.0
    inst_late_max:            float = 0
    prev_application_count:   float = 0
    prev_refused_count:       float = 0
    prev_refused_ratio:       float = 0
    prev_credit_ratio:        float = 1.0
    cc_utilization_avg:       float = 0
    cc_atm_ratio:             float = 0
    cc_late_ratio:            float = 0
    pos_dpd_max:              float = 0
    pos_late_ratio:           float = 0
    ext_source_1_x_2:         Optional[float] = None
    ext_source_2_x_3:         Optional[float] = None
    ext_source_1_x_2_x_3:    Optional[float] = None


def prepare_input(application: LoanApplication) -> pd.DataFrame:
    data = application.model_dump()
    data.pop(SENSITIVE_COL, None)

    income  = data.get("income_amount", 1) or 1
    credit  = data.get("credit_amount", 1) or 1
    annuity = data.get("annuity_amount", 0) or 0

    if data.get("ratio_income_credit") is None:
        data["ratio_income_credit"] = income / credit
    if data.get("ratio_annuity_income") is None:
        data["ratio_annuity_income"] = annuity / income
    if data.get("credit_to_annuity_ratio") is None:
        data["credit_to_annuity_ratio"] = credit / annuity if annuity else 0
    if data.get("income_per_family_member") is None:
        fam = data.get("cnt_fam_members") or 1
        data["income_per_family_member"] = income / fam
    if data.get("income_per_child") is None:
        children = data.get("cnt_children") or 0
        data["income_per_child"] = income / children if children else 0
    if data.get("ext_source_mean") is None:
        sources = [data.get(f"ext_source_{i}") for i in [1, 2, 3]
                   if data.get(f"ext_source_{i}") is not None]
        data["ext_source_mean"] = np.mean(sources) if sources else None
    if data.get("ext_source_1_x_2") is None and data.get("ext_source_1") and data.get("ext_source_2"):
        data["ext_source_1_x_2"] = data["ext_source_1"] * data["ext_source_2"]
    if data.get("ext_source_2_x_3") is None and data.get("ext_source_2") and data.get("ext_source_3"):
        data["ext_source_2_x_3"] = data["ext_source_2"] * data["ext_source_3"]
    if data.get("ext_source_1_x_2_x_3") is None:
        e1 = data.get("ext_source_1")
        e2 = data.get("ext_source_2")
        e3 = data.get("ext_source_3")
        if e1 and e2 and e3:
            data["ext_source_1_x_2_x_3"] = e1 * e2 * e3

    df = pd.DataFrame([data])

    known_cat_cols = set(cat_features) | {"age_group"}
    for col in df.columns:
        if df[col].dtype == object and col not in known_cat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     MODEL_NAME,
        "threshold": threshold,
    }


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        X = prepare_input(application)

        catboost_feature_names = bundle["catboost"].feature_names_
        X_raw = X.copy()

        for col in catboost_feature_names:
            if col not in X_raw.columns:
                X_raw[col] = np.nan

        for col in catboost_feature_names:
            if col in cat_features:
                X_raw[col] = X_raw[col].fillna("Unknown").astype(str)
            else:
                X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

        X_raw = X_raw[catboost_feature_names]

        p_lgbm = bundle["lgbm"].predict_proba(X)[:, 1]
        p_cb   = bundle["catboost"].predict_proba(X_raw)[:, 1]
        meta   = np.column_stack([p_lgbm, p_cb])
        proba  = float(bundle["meta"].predict_proba(meta)[0][1])

        decision = int(proba >= threshold)

        return {
            "default_probability": round(proba, 4),
            "threshold":           threshold,
            "decision":            decision,
            "risk_level":          "high" if proba >= 0.5
                                   else "medium" if proba >= threshold
                                   else "low",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))