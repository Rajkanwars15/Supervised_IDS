"""Stronger baselines on the same protocol: DT / RandomForest / XGBoost / LightGBM x {raw, derived} x {random, benchmark}.

random   : 5 stratified 80/20 resplits (pool capped at 500k rows)
benchmark: the dataset's own train/test where one exists (nsl-kdd, unsw-nb15); 5 model seeds (no sampling variance)
Training rows capped at 300k (stratified) for the ensembles. Fixed, modest hyper-parameters (no per-split tuning).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import robust_per_dataset as rp

OUT = Path(__file__).parent / "results"
POOL, TRAIN_CAP, N_ITER = 500_000, 300_000, 5
MODELS = {
    "DT": lambda s: DecisionTreeClassifier(random_state=s),
    "RF": lambda s: RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=s, min_samples_leaf=1, max_features="sqrt"),
    "XGB": lambda s: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=4, random_state=s, verbosity=0),
    "LGBM": lambda s: lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.1, n_jobs=4, random_state=s, verbose=-1),
}


def fit_eval(name, X, y, tr, te, seed):
    if len(tr) > TRAIN_CAP:
        tr, _ = train_test_split(tr, train_size=TRAIN_CAP, random_state=seed, stratify=y.iloc[tr])
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    med = Xtr.median().fillna(0)
    t0 = time.time(); clf = MODELS[name](seed).fit(Xtr.fillna(med), y.iloc[tr]); ft = time.time() - t0
    m = rp.metrics(y.iloc[te], clf.predict_proba(Xte.fillna(med))[:, 1]); m["fit_s"] = ft
    return m


def main(names):
    for name in names:
        d = rp.get(name); y = d["y"]; n = len(y); res = dict(dataset=name, runs={})
        pool = np.arange(n)
        if n > POOL:
            pool, _ = train_test_split(pool, train_size=POOL, random_state=0, stratify=y)
        protos = {"random": [train_test_split(pool, test_size=0.2, random_state=42 + i, stratify=y.iloc[pool]) for i in range(N_ITER)]}
        if d["is_train"] is not None and name != "cicids2017" and name != "cic-iov-2024":
            tr, te = np.flatnonzero(d["is_train"]), np.flatnonzero(~d["is_train"])
            protos["benchmark"] = [(tr, te)] * N_ITER
        for proto, splits in protos.items():
            for fs in ("raw", "derived"):
                for mn in MODELS:
                    res["runs"][f"{proto}|{fs}|{mn}"] = [fit_eval(mn, d[fs], y, np.asarray(tr), np.asarray(te), 42 + i) for i, (tr, te) in enumerate(splits)]
                    f1 = np.mean([r["f1"] for r in res["runs"][f"{proto}|{fs}|{mn}"]])
                    print(f"  [{name}] {proto} {fs} {mn} F1 {f1:.4f}", flush=True)
            (OUT / f"baselines_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["sensornetguard", "nsl-kdd", "unsw-nb15", "cic-iov-2024", "farm-flow", "cicids2017"])
