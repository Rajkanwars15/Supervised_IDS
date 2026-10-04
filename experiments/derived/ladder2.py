"""Item 4: duplicate-aware evaluation ladder (fixed metric set) for all datasets.

Protocols (default DT; RF/XGB/LGBM additionally for IoV & UNSW on the IID and dedup splits, train capped at 300k):
  random         : 5 stratified 80/20 resplits (duplication-contaminated), with the SAME fitted model scored on
                   seen / unseen test subsets (exact feature vector present / absent in train)
  dedup_group    : identical feature vectors form a group; whole groups go to train or test (5 seeds)  -> no exact test row in train
  benchmark      : the dataset's published train/test (UNSW partition, NSL-KDD), with seen/unseen subsets
  time           : earliest 80% -> latest 20% (single run; proxy where noted)
  group          : GroupShuffleSplit by entity key where valid (UNSW src IP, Farm-Flow orig host); IDS2017 -> ids2017_groups.json
Also: label-conflict ceiling = F1 of an oracle predicting the majority label per identical vector on the full data.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import raw_loaders
import baselines
import split_ladder as sl
from derived_dt_experiment import load

OUT = Path(__file__).parent / "results"
N_ITER, TRAIN_CAP = 5, 300_000
ENSEMBLE = {"unsw-nb15", "cic-iov-2024"}


def model(name, seed):
    return DecisionTreeClassifier(random_state=seed) if name == "DT" else baselines.MODELS[name](seed)


def fit_score(mname, X, y, tr, te, hashes, seed):
    if mname != "DT" and len(tr) > TRAIN_CAP:
        tr = np.random.RandomState(seed).choice(tr, TRAIN_CAP, replace=False)
    Xtr, Xte = X.iloc[tr], X.iloc[te]; med = Xtr.median().fillna(0)
    clf = model(mname, seed).fit(Xtr.fillna(med), y.iloc[tr])
    prob = clf.predict_proba(Xte.fillna(med))[:, 1] if len(clf.classes_) == 2 else np.zeros(len(te))
    yt = y.iloc[te]
    out = {"all": eu.metrics(yt, prob)}
    seen = np.isin(hashes[te], hashes[tr]); out["seen_frac"] = float(seen.mean())
    for tag, mk in (("seen", seen), ("unseen", ~seen)):
        sub = yt.to_numpy()[mk]
        out[tag] = eu.metrics(sub, prob[mk]) if mk.sum() >= 20 and 0 < sub.sum() < len(sub) else dict(n=int(mk.sum()), n_pos=int(sub.sum()), note="too few rows / single class for metrics")
    return out


def ceiling(y, hashes):
    df = pd.DataFrame({"h": hashes, "y": y.to_numpy()})
    maj = df.groupby("h").y.transform("mean").ge(0.5).astype(int)
    return eu.metrics(df.y.to_numpy(), pred=maj.to_numpy())


def ladder2(name):
    t0 = time.time()
    d = raw_loaders.LOADERS[name](); X, y = d["X"], d["y"]; n = len(y)
    H = eu.row_hash(X)
    res = dict(dataset=name, n=n, n_features=X.shape[1], note=d["note"], attack_rate=float(y.mean()),
               unique_vectors=int(len(np.unique(H))), duplicate_pct=float(100 * (1 - len(np.unique(H)) / n)),
               label_conflict_ceiling=ceiling(y, H), protocols={})
    models = ["DT"] + (["RF", "XGB", "LGBM"] if name in ENSEMBLE else [])
    print(f"== {name} n={n} unique={res['unique_vectors']} dup%={res['duplicate_pct']:.1f} ceiling F1={res['label_conflict_ceiling']['f1']:.4f}", flush=True)
    # random (contaminated) with seen/unseen
    rnd = {m: [] for m in models}
    for it in range(N_ITER):
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=42 + it, stratify=y)
        for m in models:
            rnd[m].append(fit_score(m, X, y, tr, te, H, 42 + it))
    res["protocols"]["random"] = rnd; print("  random done", flush=True)
    # dedup-group split
    codes = pd.factorize(H)[0]; dd = {m: [] for m in models}
    for it in range(N_ITER):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42 + it).split(X, y, codes))
        assert not (set(H[tr]) & set(H[te])), "dedup split leaks identical vectors"
        if y.iloc[te].nunique() < 2: continue
        for m in models:
            r = fit_score(m, X, y, tr, te, H, 42 + it); assert r["seen_frac"] == 0.0
            r["test_attack_rate"] = float(y.iloc[te].mean()); dd[m].append(r)
    res["protocols"]["dedup_group"] = dd; print("  dedup done", flush=True)
    # time
    if d["time"] is not None and d["time"].nunique() > 10:
        tr, te = sl.time_split(d)
        if y.iloc[te].nunique() == 2:
            r = fit_score("DT", X, y, tr, te, H, 42); r["test_attack_rate"] = float(y.iloc[te].mean())
            res["protocols"]["time"] = r
        else:
            res["protocols"]["time"] = "N/A (single-class test slice)"
    else:
        res["protocols"]["time"] = "N/A (no timestamp)"
    # group (skip IDS2017: see ids2017_groups.json)
    if name == "cicids2017":
        res["protocols"]["group"] = "see ids2017_groups.json (source-IP groups are degenerate)"
    elif d["group"] is not None and d["group"].nunique() > 10:
        rows = []
        for it in range(N_ITER):
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42 + it).split(X, y, d["group"]))
            sl.keycheck(d, tr, te, "group")
            if y.iloc[te].nunique() < 2: continue
            r = fit_score("DT", X, y, tr, te, H, 42 + it); r["test_attack_rate"] = float(y.iloc[te].mean()); rows.append(r)
        res["protocols"]["group"] = rows
    else:
        res["protocols"]["group"] = "N/A (no group key)"
    # benchmark (published split) for UNSW partition / NSL-KDD
    if name in ("nsl-kdd", "unsw-nb15"):
        Xtr, Xte, ytr, yte = load(name)
        Xb = pd.concat([Xtr, Xte]).reset_index(drop=True); yb = pd.concat([ytr, yte]).reset_index(drop=True)
        Hb = eu.row_hash(Xb); tr = np.arange(len(Xtr)); te = np.arange(len(Xtr), len(Xb))
        bm = {m: [fit_score(m, Xb, yb, tr, te, Hb, 42 + i) for i in range(N_ITER if m == "DT" else 2)] for m in models}
        res["protocols"]["benchmark"] = bm; print("  benchmark done", flush=True)
    (OUT / f"ladder2_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"   [{name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for nme in sys.argv[1:] or list(raw_loaders.LOADERS):
        ladder2(nme)
