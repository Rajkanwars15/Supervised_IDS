"""Item 7: raw-vs-derived compression under non-IID protocols (random / temporal / group / dedup-unseen), DT and LightGBM.

Reports F1, MCC and PR-AUC retention (derived/raw) with a paired test-set bootstrap CI on identical test rows, plus benign FPR.
Train rows capped at TRAIN_CAP (stratified) for speed. Protocols (where valid): random(5), dedup_group(3), time(1), group(3).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import raw_loaders, derive_raw
import split_ladder as sl
import ids2017_groups as ig

OUT = Path(__file__).parent / "results"
TRAIN_CAP = 600_000


def mk(model, seed):
    return DecisionTreeClassifier(random_state=seed) if model == "DT" else lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.1, n_jobs=4, verbose=-1, random_state=seed)


def probs(model, X, y, tr, te, seed):
    if len(tr) > TRAIN_CAP:
        tr = np.random.RandomState(seed).choice(tr, TRAIN_CAP, replace=False)
    Xtr = X.iloc[tr]; med = Xtr.median().fillna(0)
    clf = mk(model, seed).fit(Xtr.fillna(med), y.iloc[tr])
    return clf.predict_proba(X.iloc[te].fillna(med))[:, 1] if len(clf.classes_) == 2 else np.zeros(len(te))


def splits_for(name, d, H):
    X, y = d["X"], d["y"]; n = len(y)
    out = {"random": [train_test_split(np.arange(n), test_size=0.2, random_state=42 + i, stratify=y) for i in range(5)]}
    codes = pd.factorize(H)[0]; dd = []
    for i in range(3):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42 + i).split(X, y, codes))
        if y.iloc[te].nunique() == 2: dd.append((tr, te))
    out["dedup_unseen"] = dd
    if d["time"] is not None and d["time"].nunique() > 10:
        tr, te = sl.time_split(d)
        if y.iloc[te].nunique() == 2: out["time"] = [(tr, te)]
    if name == "cicids2017":
        ts = d["extra"]["ts"].astype("int64").to_numpy() // 10**9; blk = ts // 300; codes_b = pd.factorize(blk)[0]; rng = np.random.RandomState(0); g = []
        for f in range(3):
            r = ig.make_fold(codes_b, y, rng)
            if r:
                te_mask, _ = r; near = set(b + o for b in np.unique(blk[te_mask]) for o in (-1, 0, 1))
                tr_mask = ~(te_mask | (np.isin(blk, list(near)) & ~te_mask)); g.append((np.flatnonzero(tr_mask), np.flatnonzero(te_mask)))
        if g: out["group(5min time-block)"] = g
    elif d["group"] is not None and d["group"].nunique() > 10:
        g = []
        for i in range(3):
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42 + i).split(X, y, d["group"]))
            if y.iloc[te].nunique() == 2: g.append((tr, te))
        if g: out["group(entity)"] = g
    return out


def main(names):
    for name in names:
        t0 = time.time(); d = raw_loaders.LOADERS[name](); X, y = d["X"], d["y"]
        D = derive_raw.derived(name, X); assert len(D) == len(X)
        H = eu.row_hash(X)
        res = dict(dataset=name, n=len(y), n_raw=X.shape[1], n_derived=D.shape[1], protocols={})
        print(f"== {name}: raw {X.shape[1]} -> derived {D.shape[1]}", flush=True)
        for proto, sp in splits_for(name, d, H).items():
            res["protocols"][proto] = {}
            for model in ("DT", "LGBM"):
                rows = []
                for k, (tr, te) in enumerate(sp):
                    pr, pd_ = probs(model, X, y, tr, te, 42 + k), probs(model, D, y, tr, te, 42 + k)
                    yt = y.iloc[te].to_numpy()
                    mr, md = eu.metrics(yt, pr), eu.metrics(yt, pd_)
                    diff = eu.paired_boot_diff(yt, pr, pd_, keys=("f1", "mcc", "ap"), n=200, seed=k)
                    rows.append(dict(raw=mr, derived=md, diff=diff, n_test=int(len(te)), test_attack_rate=float(yt.mean())))
                ret = {kk: float(np.mean([r["derived"][kk] for r in rows]) / np.mean([r["raw"][kk] for r in rows])) for kk in ("f1", "mcc", "ap")}
                res["protocols"][proto][model] = dict(folds=rows, retention=ret,
                    raw_mean={kk: float(np.nanmean([r["raw"][kk] for r in rows])) for kk in ("f1", "mcc", "ap", "fpr")},
                    derived_mean={kk: float(np.nanmean([r["derived"][kk] for r in rows])) for kk in ("f1", "mcc", "ap", "fpr")})
                print(f"  {proto:24} {model:4} raw F1 {res['protocols'][proto][model]['raw_mean']['f1']:.4f} -> derived {res['protocols'][proto][model]['derived_mean']['f1']:.4f} | retention F1 {ret['f1']*100:.1f}% MCC {ret['mcc']*100:.1f}% AP {ret['ap']*100:.1f}%", flush=True)
        (OUT / f"compression_noniid_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"   [{name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(raw_loaders.LOADERS))
