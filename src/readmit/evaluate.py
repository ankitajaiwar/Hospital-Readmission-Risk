"""Evaluation on the held-out (latest) block: discrimination, calibration, subgroup
audit and an operational capacity / ROI analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score)

SEED = 7


def bootstrap_ci(y, p, fn, n=500, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def headline_metrics(y, p, n_boot=500) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for name, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score),
                     ("Brier", brier_score_loss)]:
        lo, hi = bootstrap_ci(y, p, fn, n=n_boot)
        out[name] = {"value": float(fn(y, p)), "ci95": [lo, hi]}
    out["prevalence"] = float(y.mean())
    out["n"] = int(len(y))
    return out


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def subgroup_audit(df: pd.DataFrame, y, p, cols, top_frac=0.1, min_n=100) -> pd.DataFrame:
    """Per-subgroup AUROC, calibration-in-the-large and flag rate at the global
    top-`top_frac` threshold. Small groups are reported but not scored."""
    y, p = np.asarray(y), np.asarray(p)
    thr = np.quantile(p, 1 - top_frac)
    rows = []
    for col in cols:
        for g, idx in df.groupby(col).indices.items():
            yi, pi = y[idx], p[idx]
            row = {"attribute": col, "group": g, "n": len(idx),
                   "observed_rate": yi.mean(), "mean_predicted": pi.mean(),
                   "flag_rate": (pi >= thr).mean()}
            flagged_pos = yi[pi >= thr]
            row["recall_at_threshold"] = (flagged_pos.sum() / yi.sum()) if yi.sum() else np.nan
            row["auroc"] = (roc_auc_score(yi, pi) if len(idx) >= min_n and 0 < yi.sum() < len(yi)
                            else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class ProgramAssumptions:
    """Placeholders. Replace with the health system's own figures before using the
    output for any decision."""
    cost_per_readmission: float = 15000.0
    cost_per_intervention: float = 250.0
    relative_risk_reduction: float = 0.20


def capacity_table(y, p, fracs=(0.05, 0.10, 0.15, 0.20, 0.30),
                   a: ProgramAssumptions = ProgramAssumptions()) -> pd.DataFrame:
    """If the care-management team can reach the top X% of discharges, what does the
    model catch, and what is the expected net value per 1,000 discharges?"""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    rows = []
    for f in fracs:
        k = max(1, int(round(len(p) * f)))
        flagged = y[order[:k]]
        tp = flagged.sum()
        scale = 1000 / len(p)
        prevented = tp * a.relative_risk_reduction * scale
        net = prevented * a.cost_per_readmission - k * scale * a.cost_per_intervention
        rows.append({
            "outreach_capacity": f"top {int(f * 100)}%",
            "patients_per_1000": round(k * scale),
            "precision": tp / k,
            "recall": tp / y.sum(),
            "lift_vs_random": (tp / k) / y.mean(),
            "number_needed_to_contact": (1 / (tp / k * a.relative_risk_reduction)
                                         if tp else np.inf),
            "readmissions_prevented_per_1000": prevented,
            "net_value_per_1000_usd": net,
        })
    return pd.DataFrame(rows)
