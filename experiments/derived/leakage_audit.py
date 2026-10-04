"""Leakage & dataset-artefact audit, part 1 (feature-level, uses the splits the experiments actually use).

For each (dataset, split): exact-duplicate rates (within train/test/class), cross-split duplicates,
label-conflicting duplicates, F1 on test rows seen vs. unseen in train (dedup evaluation), and a
drop-top-k-feature ablation. Default-parameter DecisionTree, raw features (same as the baseline).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from derived_dt_experiment import load
import robust_per_dataset as rp

OUT = Path(__file__).parent / "results"
SPLITS = {  # name -> list of split definitions
    "sensornetguard": ["random"], "nsl-kdd": ["benchmark", "random"], "unsw-nb15": ["benchmark", "random"],
    "cicids2017": ["own80/20", "random"], "cic-iov-2024": ["own80/20", "random"], "farm-flow": ["random"],
}


def row_hash(X):
    return pd.util.hash_pandas_object(X.round(6).astype("float64"), index=False).to_numpy()


def get_split(name, proto):
    """-> Xtr, Xte, ytr, yte (raw features, reset index)"""
    if name == "farm-flow":
        d = rp.get(name); X, y = d["raw"], d["y"]
    elif proto == "random":
        Xtr, Xte, ytr, yte = load(name)
        X = pd.concat([Xtr, Xte]).reset_index(drop=True); y = pd.concat([ytr, yte]).reset_index(drop=True)
    else:
        Xtr, Xte, ytr, yte = load(name)
        return [a.reset_index(drop=True) for a in (Xtr, Xte, ytr, yte)]
    X = rp.clean(X)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    return X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)


def fitpred(Xtr, ytr, Xte, cols=None, seed=42):
    cols = cols or list(Xtr.columns)
    med = Xtr[cols].median().fillna(0)
    clf = DecisionTreeClassifier(random_state=seed).fit(Xtr[cols].fillna(med), ytr)
    return clf, clf.predict(Xte[cols].fillna(med))


def safe_f1(y, p):
    return float(f1_score(y, p)) if len(y) and y.nunique() > 1 else None


def audit(name, proto):
    t0 = time.time()
    Xtr, Xte, ytr, yte = get_split(name, proto)
    htr, hte = row_hash(Xtr), row_hash(Xte)
    s_tr, s_te = pd.Series(htr), pd.Series(hte)
    in_train = s_te.isin(set(htr)).to_numpy()
    dup_tr = s_tr.duplicated(keep=False).to_numpy()
    # label conflicts: same feature hash but both labels present in train
    g = pd.DataFrame({"h": htr, "y": ytr.to_numpy()}).groupby("h").y.nunique()
    conflict_tr = float(g.gt(1).sum() * 1.0 / len(g))
    conflict_rows_tr = float(pd.Series(htr).map(g).gt(1).mean())
    r = dict(dataset=name, split=proto, n_train=len(Xtr), n_test=len(Xte), n_features=Xtr.shape[1],
             attack_train=float(ytr.mean()), attack_test=float(yte.mean()),
             dup_train_pct=float(s_tr.duplicated().mean() * 100), dup_test_pct=float(s_te.duplicated().mean() * 100),
             unique_train=int(s_tr.nunique()), test_in_train_pct=float(in_train.mean() * 100),
             test_in_train_attack_pct=float(in_train[yte.to_numpy() == 1].mean() * 100) if (yte == 1).any() else None,
             test_in_train_benign_pct=float(in_train[yte.to_numpy() == 0].mean() * 100) if (yte == 0).any() else None,
             conflicting_hash_pct=conflict_tr * 100, conflicting_rows_train_pct=conflict_rows_tr * 100)
    clf, pred = fitpred(Xtr, ytr, Xte)
    r["f1_full"] = safe_f1(yte, pred)
    r["f1_seen_in_train"] = safe_f1(yte[in_train], pred[in_train])
    r["f1_unseen"] = safe_f1(yte[~in_train], pred[~in_train])
    r["n_unseen"] = int((~in_train).sum())
    # train deduplicated (keep first), test on unseen only: the clean estimate
    keep = ~s_tr.duplicated().to_numpy()
    clf2, pred2 = fitpred(Xtr[keep], ytr[keep], Xte)
    r["f1_dedup_train_full"] = safe_f1(yte, pred2)
    r["f1_dedup_train_unseen"] = safe_f1(yte[~in_train], pred2[~in_train])
    # drop-top-k features ablation (importance from the full model)
    imp = pd.Series(clf.feature_importances_, Xtr.columns).sort_values(ascending=False)
    r["top_features"] = {k: round(float(v), 4) for k, v in imp.head(5).items()}
    r["drop_top"] = {}
    for k in (1, 3):
        cols = [c for c in Xtr.columns if c not in imp.index[:k]]
        _, p = fitpred(Xtr, ytr, Xte, cols)
        r["drop_top"][k] = dict(dropped=list(imp.index[:k]), f1=safe_f1(yte, p))
    # label-shuffle sanity: should be ~prior-level F1
    rng = np.random.RandomState(0)
    _, p = fitpred(Xtr, pd.Series(rng.permutation(ytr.to_numpy())), Xte)
    r["f1_label_shuffled"] = safe_f1(yte, p)
    r["seconds"] = round(time.time() - t0)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("top_features",)}), flush=True)
    return r


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPLITS)
    for n in names:
        res = [audit(n, p) for p in SPLITS[n]]
        (OUT / f"audit_{n}.json").write_text(json.dumps(res, indent=2, default=str))
