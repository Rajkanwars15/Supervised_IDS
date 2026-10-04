"""Item 8: model-complexity side of compression. Raw vs derived at depth caps {5,10,unlimited}: nodes, leaves, depth,
serialized bytes, inference latency (us/row, 100k-row batches x15, median + IQR, single process), peak tracemalloc during predict.
Run on an otherwise idle machine. One fixed stratified 80/20 split (seed 42), pool capped at 500k rows.
"""
import io, json, sys, time, tracemalloc
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import robust_per_dataset as rp

OUT = Path(__file__).parent / "results"
CAP, BATCH, REPS = 500_000, 100_000, 15


def cost(clf, Xte):
    X = Xte.iloc[:BATCH].to_numpy(dtype=np.float32)
    clf.predict(X)  # warmup
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); clf.predict(X); ts.append((time.perf_counter() - t0) / len(X) * 1e6)
    tracemalloc.start(); clf.predict(X); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    b = io.BytesIO(); joblib.dump(clf, b)
    return dict(nodes=int(clf.tree_.node_count), leaves=int(clf.tree_.n_leaves), depth=int(clf.tree_.max_depth), bytes=b.tell(),
                lat_us_per_row=dict(median=float(np.median(ts)), q25=float(np.percentile(ts, 25)), q75=float(np.percentile(ts, 75))),
                peak_predict_bytes=int(peak), n_batch=int(len(X)))


def main(names):
    for name in names:
        d = rp.get(name); y = d["y"]; idx = np.arange(len(y))
        if len(idx) > CAP: idx, _ = train_test_split(idx, train_size=CAP, random_state=0, stratify=y)
        tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y.iloc[idx])
        res = dict(dataset=name, n_raw=d["raw"].shape[1], n_derived=d["derived"].shape[1], runs={})
        for depth in (5, 10, None):
            for fs in ("raw", "derived"):
                X = d[fs]; med = X.iloc[tr].median().fillna(0)
                clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X.iloc[tr].fillna(med), y.iloc[tr])
                Xte = X.iloc[te].fillna(med); prob = clf.predict_proba(Xte)[:, 1]
                m = eu.metrics(y.iloc[te], prob); res["runs"][f"{fs}|{depth}"] = dict(metrics=m, **cost(clf, Xte))
            a, b = res["runs"][f"raw|{depth}"], res["runs"][f"derived|{depth}"]
            print(f"  [{name}] depth {depth}: nodes {a['nodes']}->{b['nodes']} bytes {a['bytes']}->{b['bytes']} lat {a['lat_us_per_row']['median']:.3f}->{b['lat_us_per_row']['median']:.3f} us/row", flush=True)
        (OUT / f"compression_latency_{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["sensornetguard", "nsl-kdd", "unsw-nb15", "cic-iov-2024", "farm-flow", "cicids2017"])
