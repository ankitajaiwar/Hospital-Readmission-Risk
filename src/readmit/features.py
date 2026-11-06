"""Feature engineering. Kept as plain functions + one sklearn ColumnTransformer so
the exact same code runs in training, batch scoring and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import MED_COLS, NUMERIC_COLS


def icd9_group(code) -> str:
    """Map an ICD-9 code to a clinical group (after Strack et al., 2014)."""
    if code is None or pd.isna(code) or code == "Unknown":
        return "Missing"
    s = str(code)
    if s[0] in "VE":
        return "Other"
    try:
        x = float(s)
    except ValueError:
        return "Other"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if int(x) == 250:
        return "Diabetes"
    if 800 <= x <= 999:
        return "Injury"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


AGE_MID = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}

ENGINEERED_NUMERIC = NUMERIC_COLS + [
    "age_years", "prior_utilization", "n_meds_changed", "n_meds_active",
]
ENGINEERED_CATEGORICAL = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "max_glu_serum", "A1Cresult", "change", "diabetesMed",
    "insulin", "diag_1_group", "diag_2_group", "diag_3_group", "payer_known",
]
# race / gender are deliberately excluded from features; they are used only to audit.


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(df[c]).astype(float)
    out["age_years"] = df["age"].map(AGE_MID).astype(float).fillna(65.0)
    out["prior_utilization"] = (df["number_outpatient"] + df["number_emergency"]
                                + df["number_inpatient"]).astype(float)
    meds = df[MED_COLS].astype(str)
    out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)
    out["n_meds_active"] = (~meds.isin(["No", "Unknown"])).sum(axis=1).astype(float)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id",
              "medical_specialty", "max_glu_serum", "A1Cresult", "change",
              "diabetesMed", "insulin"]:
        out[c] = df[c].astype(str)
    for k in (1, 2, 3):
        out[f"diag_{k}_group"] = df[f"diag_{k}"].map(icd9_group)
    out["payer_known"] = np.where(df["payer_code"].astype(str) == "Unknown", "no", "yes")
    return out


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num = Pipeline([("scale", StandardScaler())]) if scale else "passthrough"
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                        sparse_output=False)
    return ColumnTransformer([("num", num, ENGINEERED_NUMERIC),
                              ("cat", cat, ENGINEERED_CATEGORICAL)])
