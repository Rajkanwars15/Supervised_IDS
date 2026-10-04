"""Item 10: paired bootstrap comparisons on identical test rows, fixed metric set.

Per dataset and split (random 80/20 seed 42; benchmark where one exists; time where valid): fit DT (default), RF, XGB, LGBM on RAW and
DERIVED features (train capped at 300k), then
  RQ1  : derived - raw   (per model)        paired bootstrap CI for F1, MCC, PR-AUC
  model: ensemble - DT   (raw features)     paired bootstrap CI
and test-set bootstrap CIs (F1, MCC, PR-AUC, benign FPR) for each fit. 'Single-split' CIs reflect test-set sampling, not replication.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import raw_loaders, derive_raw
import baselines
import split_ladder as sl
from derived_dt_experiment import load

OUT = Path(__file__).parent / "results"
CAP = 300_000
MODELS = ["DT", "RF", "XGB", "LGBM"]
import sklearn.tree as st
M = dict(baselines.MODELS)


def probs(mname, X, y, tr, te, seed=42):
    if len(tr) > CAP:
        tr = np.random.RandomState(seed).choice(tr, CAP, replace=False)
    med = X.iloc[tr].median().fillna(0)
    clf = M[mname](seed).fit(X.iloc[tr].fillna(med), y.iloc[tr])
    return clf.predict_proba(X.iloc[te].fillna(med))[:, 1] if len(clf.classes_) == 2 else np.zeros(len(te))


def main(names):
    for name in names:
        t0 = time.time(); d = raw_loaders.LOADERS[name](); X, y = d["X"], d["y"]; D = derive_raw.derived(name, X)
        n = len(y); splits = {"random_seed42": train_test_split(np.arange(n), test_size=0.2, random_state=42, stratify=y)}
        if d["time"] is not None and d["time"].nunique() > 10:
            tr, te = sl.time_split(d)
            if y.iloc[te].nunique() == 2: splits["time"] = (tr, te)
        res = dict(dataset=name, n_raw=X.shape[1], n_derived=D.shape[1], splits={})
        if name in ("nsl-kdd", "unsw-nb15"):
            Xtr, Xte, ytr, yte = load(name)
            Xb = pd.concat([Xtr, Xte]).reset_index(drop=True); yb = pd.concat([ytr, yte]).reset_index(drop=True)
            Db = derive_raw.derived(name, Xb)  # silver UNSW columns are already lower-case, rename is a no-op
            splits["benchmark"] = ("BENCH", Xb, yb, Db, np.arange(len(Xtr)), np.arange(len(Xtr), len(Xb)))
        for sname, sp in splits.items():
            if isinstance(sp[0], str):
                _, Xs, ys, Ds, tr, te = sp
            else:
                Xs, ys, Ds = X, y, D; tr, te = sp
            yt = ys.iloc[te].to_numpy(); P = {}
            for fs, F in (("raw", Xs), ("derived", Ds)):
                for m in MODELS:
                    P[(fs, m)] = probs(m, F, ys, tr, te)
            cell = dict(n_test=int(len(te)), test_attack_rate=float(yt.mean()), fits={}, derived_minus_raw={}, ensemble_minus_dt_raw={})
            for (fs, m), p in P.items():
                cell["fits"][f"{fs}|{m}"] = dict(metrics=eu.metrics(yt, p), ci=eu.boot_ci(yt, p, n=300))
            for m in MODELS:
                cell["derived_minus_raw"][m] = eu.paired_boot_diff(yt, P[("raw", m)], P[("derived", m)], n=300)
            for m in MODELS[1:]:
                cell["ensemble_minus_dt_raw"][m] = eu.paired_boot_diff(yt, P[("raw", "DT")], P[("raw", m)], n=300)
            res["splits"][sname] = cell
            dm = cell["derived_minus_raw"]["DT"]
            print(f"  [{name}/{sname}] n_test={len(te)} DT raw F1 {cell['fits']['raw|DT']['metrics']['f1']:.4f} | derived-raw F1 {dm['f1'][0]:+.4f} [{dm['f1'][1]:+.4f},{dm['f1'][2]:+.4f}] MCC {dm['mcc'][0]:+.4f}", flush=True)
        (OUT / f"final_stats_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"   [{name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(raw_loaders.LOADERS))
