"""Scoring service. Takes one or more discharge records in the raw UCI column format
and returns calibrated 30-day readmission risk.

    MODEL_DIR=artifacts/model uvicorn readmit.api:app --port 8000
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import build_features
from .schema import CATEGORICAL_COLS, MED_COLS, NUMERIC_COLS

app = FastAPI(title="30-day readmission risk", version="0.1.0")


class Encounter(BaseModel):
    encounter_id: int
    age: str = Field(..., examples=["[70-80)"])
    admission_type_id: str = "1"
    discharge_disposition_id: str = "1"
    admission_source_id: str = "7"
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(0, ge=0)
    num_procedures: int = Field(0, ge=0)
    num_medications: int = Field(0, ge=0)
    number_outpatient: int = Field(0, ge=0)
    number_emergency: int = Field(0, ge=0)
    number_inpatient: int = Field(0, ge=0)
    number_diagnoses: int = Field(1, ge=0)
    diag_1: str = "Unknown"
    diag_2: str = "Unknown"
    diag_3: str = "Unknown"
    payer_code: str = "Unknown"
    medical_specialty: str = "Unknown"
    max_glu_serum: str = "None"
    A1Cresult: str = "None"
    change: str = "No"
    diabetesMed: str = "No"
    medications: dict[str, str] = Field(default_factory=dict,
                                        examples=[{"insulin": "Up", "metformin": "Steady"}])


class Batch(BaseModel):
    encounters: list[Encounter]


@lru_cache
def load():
    d = Path(os.environ.get("MODEL_DIR", "artifacts/model"))
    if not (d / "model.joblib").exists():
        raise RuntimeError(f"No model in {d}. Run the pipeline first.")
    return joblib.load(d / "model.joblib"), json.loads((d / "model_card.json").read_text())


def to_frame(encs: list[Encounter]) -> pd.DataFrame:
    rows = []
    for e in encs:
        r = e.model_dump(exclude={"medications"})
        for m in MED_COLS:
            r[m] = e.medications.get(m, "No")
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in CATEGORICAL_COLS:
        if c in df:
            df[c] = df[c].astype(str)
    return df


@app.get("/health")
def health():
    try:
        _, card = load()
        return {"status": "ok", "model_version": card["version"]}
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/model-card")
def model_card():
    return load()[1]


@app.post("/score")
def score(batch: Batch):
    model, card = load()
    unknown = {m for e in batch.encounters for m in e.medications} - set(MED_COLS)
    if unknown:
        raise HTTPException(422, f"Unknown medication fields: {sorted(unknown)}")
    X = build_features(to_frame(batch.encounters))
    p = model.predict_proba(X)[:, 1]
    return {"model_version": card["version"],
            "results": [{"encounter_id": e.encounter_id, "risk_30d": round(float(r), 4)}
                        for e, r in zip(batch.encounters, p)]}
