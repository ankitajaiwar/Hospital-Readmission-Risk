"""End-to-end run: data -> validation -> features -> model -> evaluation -> report.

    python -m readmit.pipeline --data data/raw/diabetic_data.csv --out artifacts
    python -m readmit.pipeline --synthetic --out artifacts_synthetic   # no data needed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import data as D
from .evaluate import (ProgramAssumptions, calibration_table, capacity_table,
                       headline_metrics, subgroup_audit)
from .features import build_features
from .monitor import drift_report
from .schema import AUDIT_COLS, TARGET
from .synthetic import make_synthetic
from .train import fit_and_select, save


def fmt_ci(m):
    return f"{m['value']:.3f} ({m['ci95'][0]:.3f}–{m['ci95'][1]:.3f})"


def run(data_path: str | None, out: Path, synthetic: bool, n_boot: int = 500) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = make_synthetic(20000) if synthetic else D.load_raw(data_path)
    if synthetic:
        raw = raw.replace("?", pd.NA)
    df, rep = D.clean(raw)
    rep.to_json(out / "cleaning_report.json")
    train, valid, test = D.pseudo_temporal_split(df)

    model, best, cand = fit_and_select(train, valid)
    save(model, best, cand, train, out / "model", asdict(rep))

    Xte, yte = build_features(test), test[TARGET].values
    p = model.predict_proba(Xte)[:, 1]
    metrics = headline_metrics(yte, p, n_boot=n_boot)
    cal = calibration_table(yte, p)
    sub = subgroup_audit(test.reset_index(drop=True), yte, p, AUDIT_COLS)
    a = ProgramAssumptions()
    cap = capacity_table(yte, p, a=a)

    Xtr = build_features(train)
    p_tr = model.predict_proba(Xtr)[:, 1]
    drift = drift_report(Xtr, Xte, p_tr, p)

    # Which inputs matter (model-agnostic, on a subsample for speed)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xte), min(4000, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte.iloc[idx], yte[idx], scoring="roc_auc",
                                n_repeats=3, random_state=0)
    imp = (pd.DataFrame({"feature": Xte.columns, "auroc_drop": pi.importances_mean})
           .sort_values("auroc_drop", ascending=False).head(12))

    # Figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([0, cal.observed_rate.max() * 1.1], [0, cal.observed_rate.max() * 1.1],
               ls="--", c="#999", lw=1)
    ax[0].plot(cal.mean_predicted, cal.observed_rate, marker="o", c="#1F3A5F")
    ax[0].set(xlabel="Predicted risk", ylabel="Actual 30-day readmission rate",
              title="Predicted vs. actual risk")
    ax[1].bar(cap.outreach_capacity, cap.recall, color="#1F3A5F")
    ax[1].set(xlabel="Patients the care team calls", ylabel="Share of readmissions caught",
              title="Readmissions caught vs. patients called")
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "evaluation.png", dpi=150); plt.close(fig)

    summary = {"selected_model": best, "candidates": cand, "test": metrics,
               "data_source": "synthetic (pipeline test only)" if synthetic else str(data_path)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cap.to_csv(out / "capacity.csv", index=False)
    sub.to_csv(out / "subgroups.csv", index=False)
    drift.to_csv(out / "drift.csv", index=False)

    write_report(out, rep, train, valid, test, best, cand, metrics, cap, sub, drift, imp, a,
                 synthetic)
    return summary


def write_report(out, rep, train, valid, test, best, cand, m, cap, sub, drift, imp, a, synthetic):
    """Write a plain-language report of one run."""
    NICE = {
        "discharge_disposition_id": "where the patient went after discharge",
        "number_inpatient": "hospital stays in the past year",
        "number_emergency": "emergency visits in the past year",
        "number_outpatient": "outpatient visits in the past year",
        "prior_utilization": "total visits in the past year",
        "diag_1_group": "main diagnosis",
        "diag_2_group": "second diagnosis",
        "diag_3_group": "third diagnosis",
        "number_diagnoses": "number of diagnoses",
        "n_meds_active": "number of diabetes medicines",
        "n_meds_changed": "diabetes medicines changed",
        "medical_specialty": "admitting doctor's specialty",
        "payer_known": "insurance recorded or not",
        "time_in_hospital": "days in hospital",
        "admission_type_id": "type of admission",
        "admission_source_id": "where the patient was admitted from",
        "num_lab_procedures": "number of lab tests",
        "num_medications": "number of medicines given",
        "num_procedures": "number of procedures",
        "age_years": "age",
        "A1Cresult": "A1c test result",
        "max_glu_serum": "glucose test result",
        "insulin": "insulin dose change",
        "change": "any diabetes medicine changed",
        "diabetesMed": "on diabetes medicine",
        "__risk_score__": "the model's risk scores",
    }
    name = lambda k: NICE.get(k, k.replace("_", " "))
    model_name = {"logistic_regression": "logistic regression",
                  "gradient_boosting": "gradient-boosted trees"}
    pct = lambda x: f"{x:.0%}" if x >= 0.1 else f"{x:.1%}"
    top10 = cap.iloc[1]

    L = ["# Readmission risk: results of this run\n"]
    if synthetic:
        L.append("> **This run used made-up test data.** The numbers only show that the code "
                 "works. Run it on the real file for real results.\n")

    L.append("## In short\n")
    L.append(f"The model ranks a readmitted patient above a non-readmitted one "
             f"{m['AUROC']['value']:.0%} of the time (50% would be a coin flip). If a care team "
             f"called the top 10% of patients by risk score, it would reach "
             f"{top10.recall:.0%} of the patients who came back within 30 days.\n")

    L.append("## The patients\n")
    L.append(f"- Hospital stays in the file: {rep.rows_in:,}")
    L.append(f"- Removed because the patient died or went to hospice: "
             f"{rep.dropped.get('expired_or_hospice', 0):,}")
    L.append(f"- Removed as repeat visits by the same patient (I kept each patient's first "
             f"visit): {rep.dropped.get('repeat_encounters', 0):,}")
    if rep.dropped.get("unknown_gender"):
        L.append(f"- Removed because sex was not recorded: {rep.dropped['unknown_gender']:,}")
    L.append(f"- Patients left: {rep.rows_out:,}, of whom {pct(rep.positive_rate)} came back "
             f"within 30 days")
    L.append(f"- The earliest {len(train):,} visits were used to train the model, the next "
             f"{len(valid):,} to fine-tune it, and the most recent {len(test):,} to test it.\n")

    L.append("## Choosing a model\n")
    L.append("I tried two models and kept the one that ranked patients better on the "
             "fine-tuning set.\n")
    L.append("| Model | Ranking score (AUROC) |\n|---|---|")
    for k, v in cand.items():
        L.append(f"| {model_name.get(k, k)}{' (kept)' if k == best else ''} | "
                 f"{v['valid_auroc']:.3f} |")
    L.append("")

    L.append("## How well it did on the most recent patients\n")
    lo, hi = m["AUROC"]["ci95"]
    L.append(f"- **Ranking score (AUROC): {m['AUROC']['value']:.3f}**, likely between "
             f"{lo:.3f} and {hi:.3f}. This is the chance the model scores a readmitted patient "
             f"higher than one who wasn't readmitted.")
    L.append(f"- **Readmission rate in this group: {pct(m['prevalence'])}.**")
    L.append(f"- **Average error of the risk estimates (Brier score): "
             f"{m['Brier']['value']:.3f}.** Lower is better.\n")
    L.append("![charts](evaluation.png)\n")
    L.append("The left chart compares the model's predicted risk with what actually "
             "happened. Points on the dashed line mean the predictions were right on "
             "average; points below it mean the model overestimated risk. The right chart "
             "shows what share of readmissions a care team would catch depending on how "
             "many patients it can call.\n")

    L.append("## What a care team would get\n")
    L.append(f"These figures assume each readmission costs ${a.cost_per_readmission:,.0f}, "
             f"each follow-up call costs ${a.cost_per_intervention:,.0f}, and a call prevents "
             f"{a.relative_risk_reduction:.0%} of readmissions. They are my placeholders, not "
             f"real hospital figures.\n")
    L.append("| Patients called | Calls per 1,000 patients | Of those called, readmitted | "
             "Share of all readmissions caught | Times the normal rate | Calls needed to prevent one readmission | Net savings per 1,000 patients |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in cap.iterrows():
        L.append(f"| {r.outreach_capacity} | {r.patients_per_1000} | {r.precision:.0%} | "
                 f"{r.recall:.0%} | {r.lift_vs_random:.1f}x | "
                 f"{r.number_needed_to_contact:.0f} | ${r.net_value_per_1000_usd:,.0f} |")
    L.append("")

    L.append("## Does it work equally well for everyone?\n")
    L.append("Race and sex were not given to the model. I used them only for this check. "
             "Everyone is flagged using the same cutoff (the top 10% of risk scores).\n")
    L.append("| Group | Patients | Actually readmitted | Model's average prediction | "
             "Share flagged | Share of their readmissions caught | Ranking score |")
    L.append("|---|---|---|---|---|---|---|")
    label = {"race": "", "gender": "", "age": "Age "}
    for _, r in sub[sub.n >= 100].iterrows():
        g = str(r.group).replace("[", "").replace(")", "")
        g = (label[r.attribute] + g) if r.attribute == "age" else {"AfricanAmerican": "African American",
                                                                     "Caucasian": "White",
                                                                     "Unknown": "Race not recorded"}.get(g, g)
        auc = "–" if pd.isna(r.auroc) else f"{r.auroc:.2f}"
        rec = "–" if pd.isna(r.recall_at_threshold) else f"{r.recall_at_threshold:.0%}"
        L.append(f"| {g} | {r.n:,} | {r.observed_rate:.1%} | {r.mean_predicted:.1%} | "
                 f"{r.flag_rate:.0%} | {rec} | {auc} |")
    L.append("\nGroups with fewer than 100 patients are left out here (they're in "
             "subgroups.csv).\n")

    L.append("## What the model relied on most\n")
    L.append("How much the ranking score drops when each piece of information is scrambled. "
             "A bigger drop means the model leans on it more.\n")
    for _, r in imp.head(8).iterrows():
        L.append(f"- {name(r.feature)}: {r.auroc_drop:.3f}")
    L.append("")

    L.append("## Did the data change over time?\n")
    L.append("This compares the training patients with the most recent ones. Under 0.1 means "
             "little change, 0.1 to 0.25 is worth watching, and over 0.25 is a big shift.\n")
    for _, r in drift.head(6).iterrows():
        word = {"stable": "little change", "watch": "worth watching",
                "investigate": "big shift"}[str(r.status)]
        L.append(f"- {name(r.feature)}: {r.psi:.2f} ({word})")
    L.append("")

    L.append("## Limits\n")
    L.append("- The data is from 1999 to 2008 and covers diabetic patients only. Hospitals "
             "work differently now, so a real model would be retrained on recent records.")
    L.append("- There are no dates in the data, so I used the order of visit IDs as a stand-in "
             "for time.")
    L.append("- Readmissions to a different hospital aren't in the data.")
    L.append("- The savings figures depend entirely on my placeholder costs.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/diabetic_data.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    s = run(a.data, Path(a.out), a.synthetic, a.n_boot)
    print(json.dumps({"selected": s["selected_model"],
                      "test_auroc": round(s["test"]["AUROC"]["value"], 3)}, indent=2))


if __name__ == "__main__":
    main()
