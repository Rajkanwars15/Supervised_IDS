"""Shared evaluation utilities: fixed metric set, bootstrap CIs, paired bootstrap differences, dedup helpers.

Fixed metric set: F1, MCC, PR-AUC (average precision), benign FPR (+ recall, balanced accuracy, ROC-AUC).
Family macro-recall is computed in the family-holdout routines.
"""
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)

METRICS = ["f1", "mcc", "ap", "fpr", "recall", "ba", "auc"]


def chance(prior):
    """Analytic no-skill references for the attack class (positive) with attack prior `prior`."""
    return dict(f1_allpos=2 * prior / (1 + prior), f1_prior_guess=prior, mcc=0.0, ap=prior, auc=0.5, ba=0.5, fpr_allpos=1.0)


def metrics(y, prob=None, pred=None):
    """y: true labels; prob: P(attack) (enables AP/AUC); pred: hard labels (default prob>=0.5)."""
    y = np.asarray(y).astype(int)
    if pred is None:
        pred = (np.asarray(prob) >= 0.5).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = dict(f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
             fpr=fp / max(fp + tn, 1), recall=tp / max(tp + fn, 1), ba=balanced_accuracy_score(y, pred),
             n=int(len(y)), n_pos=int(y.sum()), prior=float(y.mean()))
    if prob is not None and 0 < y.sum() < len(y):
        m["ap"] = average_precision_score(y, prob); m["auc"] = roc_auc_score(y, prob)
    else:
        m["ap"] = m["auc"] = float("nan")
    return m


def boot_ci(y, prob=None, pred=None, keys=("f1", "mcc", "ap", "fpr"), n=1000, seed=0):
    """Test-set bootstrap 95% CI (resamples test rows). Not an independent replication."""
    y = np.asarray(y); rng = np.random.RandomState(seed); N = len(y)
    prob = None if prob is None else np.asarray(prob); pred = None if pred is None else np.asarray(pred)
    out = {k: [] for k in keys}
    for _ in range(n):
        i = rng.randint(0, N, N)
        if y[i].min() == y[i].max():
            continue
        m = metrics(y[i], None if prob is None else prob[i], None if pred is None else pred[i])
        for k in keys:
            out[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def paired_boot_diff(y, prob_a, prob_b, keys=("f1", "mcc", "ap"), n=500, seed=0):
    """CI of metric(b) - metric(a) on identical resampled test rows. Returns {key: (diff, lo, hi)}."""
    y = np.asarray(y); pa, pb = np.asarray(prob_a), np.asarray(prob_b); rng = np.random.RandomState(seed)
    d = {k: [] for k in keys}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ma, mb = metrics(y[i], pa[i]), metrics(y[i], pb[i])
        for k in keys:
            d[k].append(mb[k] - ma[k])
    pt_a, pt_b = metrics(y, pa), metrics(y, pb)
    return {k: (float(pt_b[k] - pt_a[k]), float(np.nanpercentile(d[k], 2.5)), float(np.nanpercentile(d[k], 97.5))) for k in keys}


def row_hash(X):
    return pd.util.hash_pandas_object(X.round(6).astype("float64"), index=False).to_numpy()


def seen_mask(Xtr, Xte):
    return pd.Series(row_hash(Xte)).isin(set(row_hash(Xtr))).to_numpy()


def summarize(rows, keys=METRICS):
    """mean/sd across repeated random resplits (mean ± SD only for these)."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))
    return out
