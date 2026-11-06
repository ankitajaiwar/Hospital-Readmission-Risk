"""Post-deployment monitoring: population stability index (PSI) for inputs and scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC

EPS = 1e-4


def psi_numeric(ref, cur, bins=10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    c = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur) -> float:
    r = pd.Series(ref).value_counts(normalize=True)
    c = pd.Series(cur).value_counts(normalize=True)
    cats = r.index.union(c.index)
    r, c = r.reindex(cats, fill_value=0) + EPS, c.reindex(cats, fill_value=0) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref_X: pd.DataFrame, cur_X: pd.DataFrame, ref_scores=None,
                 cur_scores=None) -> pd.DataFrame:
    """PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate (common rule of thumb)."""
    rows = []
    for c in ENGINEERED_NUMERIC:
        rows.append({"feature": c, "psi": psi_numeric(ref_X[c], cur_X[c])})
    for c in ENGINEERED_CATEGORICAL:
        rows.append({"feature": c, "psi": psi_categorical(ref_X[c], cur_X[c])})
    if ref_scores is not None and cur_scores is not None:
        rows.append({"feature": "__risk_score__", "psi": psi_numeric(ref_scores, cur_scores)})
    out = pd.DataFrame(rows)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf],
                           labels=["stable", "watch", "investigate"])
    return out.sort_values("psi", ascending=False).reset_index(drop=True)
