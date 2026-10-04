"""Split-protocol ladder: same DecisionTree (default params, raw features) under random / group / time / family-holdout splits.

random : 5 stratified 80/20 resplits
group  : 5 GroupShuffleSplit(20%) by entity key (src IP etc.), where a group key exists
time   : earliest 80% -> latest 20% (deterministic; for IoV applied within each source file)
family : leave-one-attack-family-out (train: benign + other families; test: held-out family + 20% of benign)
Loader features follow raw_loaders.py; see its notes for differences vs the silver feature sets.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import raw_loaders
import robust_per_dataset as rp

OUT = Path(__file__).parent / "results"
N_ITER = 5


def run(X, y, tr, te, seed=42):
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    med = Xtr.median().fillna(0)
    clf = DecisionTreeClassifier(random_state=seed).fit(Xtr.fillna(med), y.iloc[tr])
    prob = clf.predict_proba(Xte.fillna(med))[:, 1] if len(clf.classes_) == 2 else np.zeros(len(te))
    yt = y.iloc[te]
    if yt.nunique() < 2:
        return None
    return rp.metrics(yt, prob)


def keycheck(d, tr, te, col):
    if d[col] is not None:
        inter = set(d[col].iloc[tr]) & set(d[col].iloc[te])
        assert not inter, f"{col} overlap {len(inter)}"


def time_split(d, frac=0.8):
    t, n = d["time"], len(d["y"])
    if d["family"] is not None and d["note"].find("within each source file") >= 0:
        rank = pd.Series(0.0, index=range(n))
        for f, idx in d["family"].groupby(d["family"]).groups.items():
            rank[idx] = t[idx].rank(pct=True, method="first").to_numpy()
        te = np.flatnonzero(rank.to_numpy() > frac)
    else:
        order = t.rank(method="first").to_numpy()
        te = np.flatnonzero(order > frac * n)
    tr = np.setdiff1d(np.arange(n), te)
    return tr, te


def ladder(name):
    d = raw_loaders.LOADERS[name]()
    X, y = d["X"], d["y"]
    n = len(y)
    print(f"== {name}: {X.shape} attack={y.mean():.3f} | {d['note']}", flush=True)
    res = dict(dataset=name, n=n, n_features=X.shape[1], note=d["note"], attack_rate=float(y.mean()), protocols={})
    # random
    rows = []
    for it in range(N_ITER):
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=42 + it, stratify=y)
        rows.append(run(X, y, tr, te, 42 + it))
    res["protocols"]["random"] = rows; print("  random done", flush=True)
    # group
    if d["group"] is not None and d["group"].nunique() > 10:
        rows = []
        for it in range(N_ITER):
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42 + it).split(X, y, d["group"]))
            keycheck(d, tr, te, "group")
            m = run(X, y, tr, te, 42 + it)
            if m: m["test_attack_rate"] = float(y.iloc[te].mean()); rows.append(m)
        res["protocols"]["group"] = rows; print("  group done", flush=True)
    else:
        res["protocols"]["group"] = "N/A (no group key)"
    # time
    if d["time"] is not None and d["time"].nunique() > 10:
        tr, te = time_split(d)
        m = run(X, y, tr, te)
        if m: m["test_attack_rate"] = float(y.iloc[te].mean()); m["train_attack_rate"] = float(y.iloc[tr].mean())
        res["protocols"]["time"] = [m] if m else "N/A (single-class test)"; print("  time done", flush=True)
    else:
        res["protocols"]["time"] = "N/A (no timestamp)"
    # family holdout
    if d["family"] is not None:
        fam = d["family"]
        fams = [f for f in fam.unique() if f != "benign" and (fam == f).sum() >= 50]
        ben = np.flatnonzero(y.to_numpy() == 0)
        btr, bte = train_test_split(ben, test_size=0.2, random_state=42)
        out = {}
        for f in fams:
            fi = np.flatnonzero(fam.to_numpy() == f)
            others = np.flatnonzero((y.to_numpy() == 1) & (fam.to_numpy() != f))
            tr = np.r_[btr, others]; te = np.r_[bte, fi]
            Xtr, Xte = X.iloc[tr], X.iloc[te]; med = Xtr.median().fillna(0)
            clf = DecisionTreeClassifier(random_state=42).fit(Xtr.fillna(med), y.iloc[tr])
            p = clf.predict(Xte.fillna(med)); yt = y.iloc[te].to_numpy()
            out[f] = dict(n_family=int(len(fi)), recall_on_family=float((p[yt == 1] == 1).mean()), benign_fpr=float((p[yt == 0] == 1).mean()))
        w = np.array([v["n_family"] for v in out.values()], float)
        res["protocols"]["family_holdout"] = dict(per_family=out,
            macro_recall=float(np.mean([v["recall_on_family"] for v in out.values()])),
            weighted_recall=float(np.average([v["recall_on_family"] for v in out.values()], weights=w)),
            mean_fpr=float(np.mean([v["benign_fpr"] for v in out.values()])))
        print("  family done", flush=True)
    else:
        res["protocols"]["family_holdout"] = "N/A (no family labels)"
    (OUT / f"ladder_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return res


if __name__ == "__main__":
    for nme in sys.argv[1:] or list(raw_loaders.LOADERS):
        t0 = time.time(); ladder(nme); print(f"   [{nme}] {time.time()-t0:.0f}s", flush=True)
