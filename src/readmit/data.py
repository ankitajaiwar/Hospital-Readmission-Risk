"""Load, validate and clean raw encounter data.

Every check that fails raises DataValidationError with a readable message, so a
bad extract stops the pipeline before a model is trained or scored on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (AGE_BANDS, CATEGORICAL_COLS, EXCLUDED_DISPOSITIONS, MED_COLS,
                     NUMERIC_COLS, REQUIRED_COLS, TARGET, TARGET_RAW)


class DataValidationError(ValueError):
    pass


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict = field(default_factory=dict)
    missing_rate: dict = field(default_factory=dict)
    positive_rate: float = float("nan")

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


def load_raw(path: str | Path) -> pd.DataFrame:
    # The file uses "?" for missing. keep_default_na=False stops pandas turning the
    # literal lab value "None" (= test not done) into NaN.
    df = pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df["encounter_id"].duplicated().any():
        raise DataValidationError("encounter_id is not unique")
    for c in NUMERIC_COLS:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any():
            raise DataValidationError(f"Non-numeric or missing values in {c}")
        if (vals < 0).any():
            raise DataValidationError(f"Negative values in {c}")
    if not (df["time_in_hospital"].between(1, 14)).all():
        raise DataValidationError("time_in_hospital outside 1-14 days")
    bad_target = set(df[TARGET_RAW].unique()) - {"NO", ">30", "<30"}
    if bad_target:
        raise DataValidationError(f"Unexpected readmitted values: {bad_target}")
    bad_age = set(df["age"].dropna().unique()) - set(AGE_BANDS)
    if bad_age:
        raise DataValidationError(f"Unexpected age bands: {bad_age}")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a modelling table: one row per patient (first encounter), with target."""
    validate(df)
    rep = CleaningReport(rows_in=len(df))
    df = df.copy()

    # 1. Remove encounters where readmission is impossible.
    disp = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    mask = disp.isin(EXCLUDED_DISPOSITIONS)
    rep.dropped["expired_or_hospice"] = int(mask.sum())
    df = df[~mask]

    # 2. Unknown gender is a handful of rows and not a usable category.
    mask = ~df["gender"].isin(["Male", "Female"])
    rep.dropped["unknown_gender"] = int(mask.sum())
    df = df[~mask]

    # 3. Keep each patient's first encounter so the same person never appears in
    #    both train and test, and observations are independent.
    df = df.sort_values("encounter_id")
    first = ~df["patient_nbr"].duplicated(keep="first")
    rep.dropped["repeat_encounters"] = int((~first).sum())
    df = df[first]

    # 4. Normalise categoricals: missing -> "Unknown"; IDs as strings.
    for c in CATEGORICAL_COLS + MED_COLS:
        df[c] = df[c].astype("string").fillna("Unknown").str.strip()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c])

    df[TARGET] = (df[TARGET_RAW] == "<30").astype(int)

    rep.rows_out = len(df)
    rep.missing_rate = {c: float((df[c] == "Unknown").mean()) for c in CATEGORICAL_COLS
                        if (df[c] == "Unknown").any()}
    rep.positive_rate = float(df[TARGET].mean())
    return df.reset_index(drop=True), rep


def pseudo_temporal_split(df: pd.DataFrame, train=0.6, valid=0.2):
    """Split by encounter_id order (a proxy for admission time; the public data has
    no dates). Train on the earliest encounters, calibrate on the next block, test on
    the latest: this mimics prospective validation better than a random split."""
    df = df.sort_values("encounter_id").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train), int(n * (train + valid))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
