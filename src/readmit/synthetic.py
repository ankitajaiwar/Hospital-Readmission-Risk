"""Synthetic data with the same schema as the UCI file.

Used ONLY for tests and CI so the pipeline can run without the real data.
Numbers produced from it mean nothing clinically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import AGE_BANDS, MED_COLS


def make_synthetic(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_pat = int(n * 0.75)
    df = pd.DataFrame({
        "encounter_id": np.arange(n) * 7 + 1000,
        "patient_nbr": rng.integers(0, n_pat, n),
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"],
                           n, p=[.72, .18, .03, .01, .02, .04]),
        "gender": rng.choice(["Female", "Male"], n),
        "age": rng.choice(AGE_BANDS, n, p=[.002, .008, .02, .04, .1, .17, .22, .25, .16, .03]),
        "weight": "?",
        "admission_type_id": rng.choice([1, 2, 3, 5, 6], n),
        "discharge_disposition_id": rng.choice([1, 2, 3, 6, 11, 18, 22], n,
                                               p=[.6, .03, .14, .13, .02, .04, .04]),
        "admission_source_id": rng.choice([1, 2, 7, 17], n, p=[.3, .05, .55, .1]),
        "time_in_hospital": rng.integers(1, 15, n),
        "payer_code": rng.choice(["MC", "HM", "SP", "BC", "?"], n, p=[.3, .06, .05, .05, .54]),
        "medical_specialty": rng.choice(["InternalMedicine", "Cardiology", "Emergency/Trauma",
                                         "Family/GeneralPractice", "?"], n,
                                        p=[.15, .06, .08, .07, .64]),
        "num_lab_procedures": rng.integers(1, 100, n),
        "num_procedures": rng.integers(0, 7, n),
        "num_medications": rng.integers(1, 60, n),
        "number_outpatient": rng.poisson(.4, n),
        "number_emergency": rng.poisson(.2, n),
        "number_inpatient": rng.poisson(.6, n),
        "diag_1": rng.choice(["428", "414", "786", "250.83", "V57", "715", "486", "?"], n),
        "diag_2": rng.choice(["276", "428", "250", "427", "?"], n),
        "diag_3": rng.choice(["250", "401", "276", "428", "?"], n),
        "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"], n, p=[.95, .025, .015, .01]),
        "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n, p=[.83, .05, .04, .08]),
    })
    for m in MED_COLS:
        df[m] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[.85, .1, .03, .02])
    df["change"] = np.where((df[MED_COLS].isin(["Up", "Down"])).any(axis=1), "Ch", "No")
    df["diabetesMed"] = np.where((df[MED_COLS] != "No").any(axis=1), "Yes", "No")

    # Plant a signal so tests can check the model learns something.
    logit = (-2.6 + 0.45 * df["number_inpatient"] + 0.25 * df["number_emergency"]
             + 0.04 * df["time_in_hospital"] + 0.3 * (df["discharge_disposition_id"] == 3)
             + 0.25 * (df["diag_1"] == "428"))
    p = 1 / (1 + np.exp(-logit))
    u = rng.random(n)
    df["readmitted"] = np.where(u < p, "<30", np.where(u < p + 0.33, ">30", "NO"))
    return df
