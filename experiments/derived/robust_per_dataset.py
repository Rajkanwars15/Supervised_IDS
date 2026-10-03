"""One tuned decision tree per dataset, 5 iterations, with confidence metrics.

Per dataset and protocol:
  official : the dataset's own train/test split (where one exists); iterations vary the tree seed + CV folds
  random   : 5 different stratified 80/20 re-splits of the pooled data
Model/feature-set selection uses ONLY the training part (3-fold CV on a <=100k subsample).
Compared against the untuned baseline DT (raw features, defaults) on the identical split.
Reports mean/std/95% t-interval over iterations, a within-split bootstrap CI for F1,
and prediction-confidence metrics (Brier, ECE, mean confidence, high-confidence coverage/accuracy).
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from derived_dt_experiment import load, DATA
from derive_features import REGISTRY
import shared_analysis as sa

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
N_ITER = 5
GRID = [dict(max_depth=d, min_samples_leaf=m, criterion=c, class_weight=w)
        for d, m, c, w in itertools.product([None, 12, 20], [1, 5, 20], ["gini", "entropy"], [None, "balanced"])]
T975 = stats.t.ppf(0.975, N_ITER - 1)


def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).astype("float32")


def get(name):
    """-> dict(raw, derived, y, is_train or None)"""
    if name == "farm-flow":
        files = sorted((DATA / "farm-flow/Datasets").glob("*/*_Farm-Flows.csv"))
        df = pd.concat(pd.read_csv(f, low_memory=False) for f in files).reset_index(drop=True)
        y = df.pop("is_attack").astype(int)
        raw = df.select_dtypes("number").drop(columns=["traffic"], errors="ignore")  # `traffic` = attack-type label
        p = df.orig_pkts + df.resp_pkts
        D = sa.schema(df.orig_ip_bytes, df.resp_ip_bytes, df.orig_pkts, df.resp_pkts, df.flow_duration,
                      sa.ratio(df.flow_RST_flag_count, p))
        return dict(raw=clean(raw), derived=clean(D), y=y, is_train=None)
    Xtr, Xte, ytr, yte = load(name)
    X = pd.concat([Xtr, Xte]).reset_index(drop=True)
    y = pd.concat([ytr, yte]).reset_index(drop=True).astype(int)
    is_train = np.r_[np.ones(len(Xtr), bool), np.zeros(len(Xte), bool)] if name != "sensornetguard" else None
    D = REGISTRY[name](X)
    if name in ("unsw-nb15", "nsl-kdd", "cicids2017"):
        D = pd.concat([D.add_prefix("a_"), sa.derive(name, X).add_prefix("b_")], axis=1)
    D = D.dropna(axis=1, how="all")
    D = D.loc[:, D.nunique() > 1]
    return dict(raw=clean(X), derived=clean(D), y=y, is_train=is_train)


def fsets(d):
    both = pd.concat([d["raw"].add_prefix("r_"), d["derived"].add_prefix("d_")], axis=1)
    return {"raw": d["raw"], "derived": d["derived"], "raw+derived": both}


def fit_pred(Xtr, ytr, Xte, params, seed):
    med = Xtr.median().fillna(0)
    clf = DecisionTreeClassifier(random_state=seed, **params).fit(Xtr.fillna(med), ytr)
    return clf, clf.predict_proba(Xte.fillna(med))[:, 1]


def tune(sets, idx_tr, y, seed):
    """3-fold CV F1 over (feature set x grid) on a <=100k subsample of the TRAIN rows."""
    rng = np.random.RandomState(seed)
    sub = rng.choice(idx_tr, min(100000, len(idx_tr)), replace=False)
    ys = y.iloc[sub]
    best = (-1, None, None)
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for fs, X in sets.items():
        Xs = X.iloc[sub]
        Xs = Xs.fillna(Xs.median().fillna(0))
        for p in GRID:
            pred = cross_val_predict(DecisionTreeClassifier(random_state=seed, **p), Xs, ys, cv=skf)
            f = f1_score(ys, pred)
            if f > best[0]:
                best = (f, fs, p)
    return best


def conf_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    conf = np.where(pred == 1, prob, 1 - prob)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    hi = conf >= 0.9
    return dict(brier=brier_score_loss(y, prob), ece=ece, mean_conf=conf.mean(), hiconf_cov=hi.mean(),
                hiconf_acc=(pred[hi] == y[hi]).mean() if hi.any() else np.nan)


def metrics(y, prob):
    y = np.asarray(y); pred = (prob >= 0.5).astype(int)
    m = dict(acc=accuracy_score(y, pred), prec=precision_score(y, pred, zero_division=0),
             rec=recall_score(y, pred), f1=f1_score(y, pred), mcc=matthews_corrcoef(y, pred),
             auc=roc_auc_score(y, prob), ap=average_precision_score(y, prob))
    m.update(conf_metrics(y, prob))
    # within-split bootstrap CI for F1 (multinomial resampling of the confusion cells)
    tp, fp, fn, tn = [(pred == a) & (y == b) for a, b in ((1, 1), (1, 0), (0, 1), (0, 0))]
    cells = np.array([c.sum() for c in (tp, fp, fn, tn)])
    s = np.random.RandomState(0).multinomial(len(y), cells / cells.sum(), 1000)
    f1s = 2 * s[:, 0] / np.maximum(2 * s[:, 0] + s[:, 1] + s[:, 2], 1)
    m["f1_boot_lo"], m["f1_boot_hi"] = np.percentile(f1s, [2.5, 97.5])
    return m


def run_protocol(name, d, proto):
    sets, y = fsets(d), d["y"]
    rows = []
    for it in range(N_ITER):
        seed = 42 + it
        if proto == "official":
            tr, te = np.flatnonzero(d["is_train"]), np.flatnonzero(~d["is_train"])
        else:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        t0 = time.time()
        cv_f1, fs, params = tune(sets, tr, y, seed)
        Xtr, Xte = sets[fs].iloc[tr], sets[fs].iloc[te]
        _, prob = fit_pred(Xtr, y.iloc[tr], Xte, params, seed)
        Rtr, Rte = d["raw"].iloc[tr], d["raw"].iloc[te]
        _, bprob = fit_pred(Rtr, y.iloc[tr], Rte, {}, seed)
        row = dict(iter=it, seed=seed, feature_set=fs, params=params, cv_f1=cv_f1,
                   tuned=metrics(y.iloc[te], prob), baseline=metrics(y.iloc[te], bprob))
        rows.append(row)
        print(f"  [{name}/{proto}] it{it} {fs} {params}  tuned F1 {row['tuned']['f1']:.4f}"
              f" (boot {row['tuned']['f1_boot_lo']:.4f}-{row['tuned']['f1_boot_hi']:.4f})"
              f" | baseline F1 {row['baseline']['f1']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return rows


def agg(rows, which):
    out = {}
    for k in rows[0][which]:
        v = np.array([r[which][k] for r in rows], float)
        sd = v.std(ddof=1)
        out[k] = dict(mean=v.mean(), std=sd, ci_lo=v.mean() - T975 * sd / np.sqrt(N_ITER), ci_hi=v.mean() + T975 * sd / np.sqrt(N_ITER))
    return out


def main(names):
    for name in names:
        res_file = OUT / f"robust_{name}.json"
        allres = {}
        print(f"== {name}", flush=True)
        d = get(name)
        print(f"   rows={len(d['y'])} raw={d['raw'].shape[1]} derived={d['derived'].shape[1]} attack={d['y'].mean():.3f}", flush=True)
        for proto in (["official", "random"] if d["is_train"] is not None else ["random"]):
            rows = run_protocol(name, d, proto)
            allres[f"{name}|{proto}"] = dict(iterations=rows, tuned=agg(rows, "tuned"), baseline=agg(rows, "baseline"))
            res_file.write_text(json.dumps(allres, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["sensornetguard", "nsl-kdd", "unsw-nb15", "farm-flow", "cic-iov-2024", "cicids2017"])
