import numpy as np
import pandas as pd
import pytest

from readmit import data as D
from readmit.evaluate import capacity_table, headline_metrics
from readmit.features import build_features, icd9_group
from readmit.monitor import psi_numeric
from readmit.pipeline import run
from readmit.schema import TARGET
from readmit.synthetic import make_synthetic


@pytest.fixture(scope="module")
def raw():
    return make_synthetic(3000, seed=1).replace("?", pd.NA)


def test_validation_catches_missing_column(raw):
    with pytest.raises(D.DataValidationError, match="Missing required columns"):
        D.validate(raw.drop(columns=["number_inpatient"]))


def test_validation_catches_bad_target(raw):
    bad = raw.copy()
    bad.loc[0, "readmitted"] = "maybe"
    with pytest.raises(D.DataValidationError, match="readmitted"):
        D.validate(bad)


def test_validation_catches_negative_counts(raw):
    bad = raw.copy()
    bad.loc[0, "number_emergency"] = -1
    with pytest.raises(D.DataValidationError, match="Negative"):
        D.validate(bad)


def test_clean_one_row_per_patient_and_no_hospice(raw):
    df, rep = D.clean(raw)
    assert df["patient_nbr"].is_unique
    assert not df["discharge_disposition_id"].isin(["11", "13", "14", "19", "20", "21"]).any()
    assert rep.rows_out == len(df)
    assert set(df[TARGET].unique()) <= {0, 1}


def test_split_is_ordered_and_disjoint(raw):
    df, _ = D.clean(raw)
    tr, va, te = D.pseudo_temporal_split(df)
    assert tr.encounter_id.max() < va.encounter_id.min() < te.encounter_id.min()
    assert not set(tr.patient_nbr) & set(te.patient_nbr)


@pytest.mark.parametrize("code,group", [("428", "Circulatory"), ("250.83", "Diabetes"),
                                        ("V57", "Other"), ("786", "Respiratory"),
                                        ("Unknown", "Missing"), ("820", "Injury")])
def test_icd9_grouping(code, group):
    assert icd9_group(code) == group


def test_features_exclude_protected_attributes(raw):
    X = build_features(D.clean(raw)[0])
    assert "race" not in X.columns and "gender" not in X.columns
    assert not X.isna().any().any()


def test_capacity_table_monotone_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    cap = capacity_table(y, p)
    assert cap.recall.is_monotonic_increasing
    assert (cap.lift_vs_random > 1).all()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi_numeric(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi_numeric(a, rng.normal(1, 1, 5000)) > 0.25


def test_end_to_end_beats_chance(tmp_path):
    s = run(None, tmp_path, synthetic=True, n_boot=50)
    assert s["test"]["AUROC"]["value"] > 0.55
    for f in ["REPORT.md", "evaluation.png", "model/model.joblib", "model/model_card.json"]:
        assert (tmp_path / f).exists()
