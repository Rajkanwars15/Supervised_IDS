"""RQ1: formal compression table. Raw vs derived feature sets, F1/MCC/AP retention, and model-cost metrics.

Per dataset x depth cap {5, 10, None}: 5 stratified 80/20 resplits (pool capped at CAP rows, stratified, for speed),
default-parameter DecisionTree (class_weight=None), same seed/split for raw and derived (paired).
Cost: nodes, leaves, depth, serialized size (joblib bytes), inference latency (median ms per 1000 rows over 9
timed predicts on <=20k test rows, single process) - run on an otherwise idle machine.
"""
import io, json, sys, time
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import robust_per_dataset as rp

OUT = Path(__file__).parent / "results"
CAP = 500_000
N_ITER = 5
DEPTHS = [5, 10, None]


def size_bytes(clf):
    b = io.BytesIO(); joblib.dump(clf, b); return b.tell()


def latency_ms_per_1k(clf, X):
    X = X.iloc[:20000]
    ts = []
    for _ in range(9):
        t0 = time.perf_counter(); clf.predict(X); ts.append(time.perf_counter() - t0)
    return float(np.median(ts) / len(X) * 1000 * 1000)


def one(Xtr, ytr, Xte, yte, depth, seed):
    med = Xtr.median().fillna(0)
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]; pred = (prob >= 0.5).astype(int)
    return dict(f1=f1_score(yte, pred), mcc=matthews_corrcoef(yte, pred), ap=average_precision_score(yte, prob),
                nodes=int(clf.tree_.node_count), leaves=int(clf.tree_.n_leaves), depth=int(clf.tree_.max_depth),
                bytes=size_bytes(clf), lat=latency_ms_per_1k(clf, Xte))


def main(names):
    for name in names:
        d = rp.get(name)
        y = d["y"]; n = len(y)
        idx = np.arange(n)
        if n > CAP:
            idx, _ = train_test_split(idx, train_size=CAP, random_state=0, stratify=y)
        res = dict(dataset=name, n_pool=int(len(idx)), n_raw=d["raw"].shape[1], n_derived=d["derived"].shape[1], runs={})
        for depth in DEPTHS:
            for fs in ("raw", "derived"):
                rows = []
                for it in range(N_ITER):
                    tr, te = train_test_split(idx, test_size=0.2, random_state=42 + it, stratify=y.iloc[idx])
                    X = d[fs]
                    rows.append(one(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], depth, 42 + it))
                res["runs"][f"{fs}|{depth}"] = rows
            print(f"  [{name}] depth={depth} done", flush=True)
        (OUT / f"compression_{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["sensornetguard", "nsl-kdd", "unsw-nb15", "cic-iov-2024", "farm-flow", "cicids2017"])
