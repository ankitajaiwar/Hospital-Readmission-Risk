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
    ax[0].set(xlabel="Mean predicted risk", ylabel="Observed 30-day readmission rate",
              title="Calibration (test block, deciles)")
    ax[1].bar(cap.outreach_capacity, cap.recall, color="#1F3A5F")
    ax[1].set(ylabel="Share of readmissions flagged", title="Recall by outreach capacity")
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "evaluation.png", dpi=150); plt.close(fig)

    summary = {"selected_model": best, "candidates": cand, "test": metrics,
               "data_source": "synthetic (pipeline test only)" if synthetic else str(data_path)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    cap.to_csv(out / "capacity.csv", index=False)
    sub.to_csv(out / "subgroups.csv", index=False)
    drift.to_csv(out / "drift.csv", index=False)

    write_report(out, rep, train, valid, test, best, cand, metrics, cap, sub, drift, imp, a,
                 synthetic)
    return summary


def write_report(out, rep, train, valid, test, best, cand, m, cap, sub, drift, imp, a, synthetic):
    L = []
    L.append("# 30-day readmission risk: evaluation report\n")
    if synthetic:
        L.append("> **Synthetic data.** These numbers only show the pipeline runs. "
                 "Run on the UCI file for real results.\n")
    L.append("## Cohort\n")
    L.append(f"- Encounters in: {rep.rows_in:,}; modelling rows (first encounter per patient, "
             f"excluding expired/hospice): {rep.rows_out:,}")
    L.append(f"- Dropped: " + ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in rep.dropped.items()))
    L.append(f"- 30-day readmission rate: {rep.positive_rate:.1%}")
    L.append(f"- Split by encounter order (proxy for time): train {len(train):,} / "
             f"calibration {len(valid):,} / test {len(test):,}\n")
    L.append("## Model selection (validation block)\n")
    L.append("| Model | AUROC | AUPRC |\n|---|---|---|")
    for k, v in cand.items():
        L.append(f"| {k}{' (selected)' if k == best else ''} | {v['valid_auroc']:.3f} | {v['valid_auprc']:.3f} |")
    L.append("\n## Held-out test block (latest encounters), 95% bootstrap CI\n")
    L.append(f"- AUROC {fmt_ci(m['AUROC'])}\n- AUPRC {fmt_ci(m['AUPRC'])} "
             f"(prevalence {m['prevalence']:.3f})\n- Brier {fmt_ci(m['Brier'])}\n")
    L.append("![evaluation](evaluation.png)\n")
    L.append("## Operational view: what a care-management team would get\n")
    L.append(f"Assumptions (placeholders, replace with real figures): readmission cost "
             f"${a.cost_per_readmission:,.0f}; outreach cost ${a.cost_per_intervention:,.0f} per "
             f"patient; outreach prevents {a.relative_risk_reduction:.0%} of readmissions among "
             f"those reached.\n")
    L.append("| Capacity | Patients / 1,000 | Precision | Recall | Lift | NNC | Prevented / 1,000 | Net value / 1,000 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in cap.iterrows():
        L.append(f"| {r.outreach_capacity} | {r.patients_per_1000} | {r.precision:.2f} | {r.recall:.2f} | "
                 f"{r.lift_vs_random:.1f}× | {r.number_needed_to_contact:.0f} | "
                 f"{r.readmissions_prevented_per_1000:.1f} | ${r.net_value_per_1000_usd:,.0f} |")
    L.append("\nNNC = patients contacted per readmission prevented.\n")
    L.append("## Subgroup audit (race and gender are not model inputs)\n")
    L.append("Flag rate and recall use the single top-10% threshold for everyone.\n")
    L.append("| Attribute | Group | n | Observed | Mean predicted | Flag rate | Recall | AUROC |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in sub[sub.n >= 100].iterrows():
        auc = "—" if pd.isna(r.auroc) else f"{r.auroc:.3f}"
        rec = "—" if pd.isna(r.recall_at_threshold) else f"{r.recall_at_threshold:.2f}"
        L.append(f"| {r.attribute} | {r.group} | {r.n:,} | {r.observed_rate:.3f} | "
                 f"{r.mean_predicted:.3f} | {r.flag_rate:.2f} | {rec} | {auc} |")
    L.append("\nGroups under 100 patients are left out of this table (see subgroups.csv).\n")
    L.append("## Most influential inputs (permutation importance, AUROC drop)\n")
    for _, r in imp.iterrows():
        L.append(f"- {r.feature}: {r.auroc_drop:.4f}")
    L.append("\n## Drift check: training vs. test block (PSI)\n")
    top = drift.head(6)
    for _, r in top.iterrows():
        L.append(f"- {r.feature}: {r.psi:.3f} ({r.status})")
    L.append("\n## Limitations\n")
    L.append("- Public data from 1999–2008; coding, care patterns and populations have changed. "
             "A local model would be retrained and validated prospectively on current EHR data.")
    L.append("- No admission dates, so 'temporal' validation uses encounter order as a proxy.")
    L.append("- Readmissions to other hospitals are not captured.")
    L.append("- ROI figures depend entirely on the placeholder assumptions above.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


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
