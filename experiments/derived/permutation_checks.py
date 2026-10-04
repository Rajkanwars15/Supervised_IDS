"""Items 1-2: label-permutation sanity checks (all datasets) and Farm-Flow identifier / top-feature ablation.

Permutation: stratified shuffle of y (prior preserved), full pipeline refit (imputation medians fit on train), default DT.
  variant 'before' : permute y, THEN build features from raw columns and refit preprocessing  (Farm-Flow only has a raw rebuild)
  variant 'after'  : build features first, then permute y
Pool capped at CAP rows (stratified) so unlimited-depth trees fit noise in reasonable time.
Chance references (attack = positive): F1_allpos = 2p/(1+p), MCC = 0, AP = p, AUC = 0.5.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import robust_per_dataset as rp
import shared_analysis as sa
from derived_dt_experiment import DATA

OUT = Path(__file__).parent / "results"
CAP = 200_000


def fit_eval(Xtr, ytr, Xte, yte, seed=42, depth=None):
    med = Xtr.median().fillna(0)
    clf = DecisionTreeClassifier(random_state=seed, max_depth=depth).fit(Xtr.fillna(med), ytr)
    prob = clf.predict_proba(Xte.fillna(med))[:, 1]
    return eu.metrics(yte, prob), clf


def split(idx, y, seed=42):
    return train_test_split(idx, test_size=0.2, random_state=seed, stratify=y.iloc[idx])


def pool(y, seed=0):
    idx = np.arange(len(y))
    if len(idx) > CAP:
        idx, _ = train_test_split(idx, train_size=CAP, random_state=seed, stratify=y)
    return idx


def agg(rows, keys=("f1", "mcc", "ap", "auc", "ba", "fpr")):
    return {k: dict(mean=float(np.nanmean([r[k] for r in rows])), sd=float(np.nanstd([r[k] for r in rows], ddof=1)) if len(rows) > 1 else float("nan")) for k in keys}


def permutation(X, y, n_perm, seed=0, label=""):
    idx = pool(y); tr, te = split(idx, y)
    real, _ = fit_eval(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
    rng = np.random.RandomState(seed); rows = []
    for p in range(n_perm):
        ys = y.copy()
        ys.iloc[idx] = rng.permutation(y.iloc[idx].to_numpy())  # permutation within the pool (prior preserved)
        m, _ = fit_eval(X.iloc[tr], ys.iloc[tr], X.iloc[te], ys.iloc[te], seed=seed + p)
        rows.append(m)
    pr = float(y.iloc[idx].mean())
    res = dict(label=label, n_pool=int(len(idx)), prior=pr, chance=eu.chance(pr), real=real, shuffled=agg(rows), n_perm=n_perm,
               pass_mcc=bool(abs(np.mean([r["mcc"] for r in rows])) <= 0.01),
               pass_auc=bool(abs(np.nanmean([r["auc"] for r in rows]) - 0.5) <= 0.01))
    print(f"  [{label}] real F1 {real['f1']:.4f} MCC {real['mcc']:.4f} | shuffled F1 {res['shuffled']['f1']['mean']:.4f} MCC {res['shuffled']['mcc']['mean']:+.4f}±{res['shuffled']['mcc']['sd']:.4f} AUC {res['shuffled']['auc']['mean']:.4f} (chance F1 {res['chance']['f1_allpos']:.4f}) pass={res['pass_mcc'] and res['pass_auc']}", flush=True)
    return res


# ---- Farm-Flow raw rebuild (so label shuffling can precede feature construction) ----
def ff_raw_df():
    fs = sorted((DATA / "farm-flow/Datasets").glob("*/*_Farm-Flows.csv"))
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


IDENT = lambda c: c.startswith("id.")  # IP/port identifiers


def ff_build(df, keep_ports=True):
    d = df.drop(columns=["is_attack", "traffic"], errors="ignore")
    X = d.select_dtypes("number")
    if not keep_ports:
        X = X[[c for c in X.columns if not IDENT(c)]]
    p = d.orig_pkts + d.resp_pkts
    D = sa.schema(d.orig_ip_bytes, d.resp_ip_bytes, d.orig_pkts, d.resp_pkts, d.flow_duration, sa.ratio(d.flow_RST_flag_count, p))
    return rp.clean(X), rp.clean(D)


def farmflow_checks(n_perm):
    df = ff_raw_df(); y = df["is_attack"].astype(int).reset_index(drop=True)
    out = {}
    # variant 'after': features built from true-label frame, labels permuted afterwards
    X, D = ff_build(df, keep_ports=True)
    assert len(X) == len(y)
    out["after_raw_with_ports"] = permutation(X, y, n_perm, label="farm-flow raw(with ports) | permute after")
    # variant 'before': permute y first, then rebuild features + preprocessing from raw columns
    rng = np.random.RandomState(1); idx = pool(y); tr, te = split(idx, y); rows = []
    for p in range(n_perm):
        ys = y.copy(); ys.iloc[idx] = rng.permutation(y.iloc[idx].to_numpy())
        d2 = df.copy(); d2["is_attack"] = ys.to_numpy()
        Xb, _ = ff_build(d2, keep_ports=True)  # rebuild after permuting
        yb = d2["is_attack"].astype(int)
        m, _ = fit_eval(Xb.iloc[tr], yb.iloc[tr], Xb.iloc[te], yb.iloc[te], seed=p); rows.append(m)
    pr = float(y.iloc[idx].mean())
    out["before_raw_with_ports"] = dict(label="farm-flow raw | permute before derivation", prior=pr, chance=eu.chance(pr), shuffled=agg(rows), n_perm=n_perm)
    print(f"  [permute-before] shuffled F1 {out['before_raw_with_ports']['shuffled']['f1']['mean']:.4f} MCC {out['before_raw_with_ports']['shuffled']['mcc']['mean']:+.4f}", flush=True)
    out["after_derived"] = permutation(D, y, n_perm, label="farm-flow derived | permute after")
    return out, df, y


def farmflow_ablation(df, y, n_perm):
    X_all, _ = ff_build(df, keep_ports=True)
    X_noid, _ = ff_build(df, keep_ports=False)
    idx = pool(y); tr, te = split(idx, y)
    # importance on identifier-free real-label model
    _, clf = fit_eval(X_noid.iloc[tr], y.iloc[tr], X_noid.iloc[te], y.iloc[te])
    order = list(pd.Series(clf.feature_importances_, X_noid.columns).sort_values(ascending=False).index)
    print("  top features (identifier-free):", order[:6], flush=True)
    levels = {"L0 all numeric (incl. ports, as used in earlier experiments)": X_all,
              "L1 identifiers removed (id.* ports/IPs)": X_noid}
    for k in (1, 3, 5):
        levels[f"L1 + drop top-{k} ({','.join(order[:k])})"] = X_noid.drop(columns=order[:k])
    # balanced test (undersample the majority class in the test fold only)
    rng = np.random.RandomState(0); yte = y.iloc[te]
    pos, neg = te[yte.to_numpy() == 1], te[yte.to_numpy() == 0]
    nmin = min(len(pos), len(neg)); bal = np.r_[rng.choice(pos, nmin, replace=False), rng.choice(neg, nmin, replace=False)]
    res = {}
    for name, X in levels.items():
        real, _ = fit_eval(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
        realb, _ = fit_eval(X.iloc[tr], y.iloc[tr], X.iloc[bal], y.iloc[bal])
        rng2 = np.random.RandomState(7); rows = []
        for p in range(n_perm):
            ys = y.copy(); ys.iloc[idx] = rng2.permutation(y.iloc[idx].to_numpy())
            m, _ = fit_eval(X.iloc[tr], ys.iloc[tr], X.iloc[te], ys.iloc[te], seed=p); rows.append(m)
        res[name] = dict(n_features=X.shape[1], real=real, real_balanced_test=realb, shuffled=agg(rows), n_perm=n_perm)
        print(f"  [{name[:60]}] real F1 {real['f1']:.4f} MCC {real['mcc']:.4f} | balanced-test MCC {realb['mcc']:.4f} BA {realb['ba']:.4f} | shuffled MCC {res[name]['shuffled']['mcc']['mean']:+.4f} F1 {res[name]['shuffled']['f1']['mean']:.4f}", flush=True)
    res["_top_features"] = order[:10]
    return res


def others(n_perm):
    out = {}
    for n in ["sensornetguard", "nsl-kdd", "unsw-nb15", "cic-iov-2024", "cicids2017"]:
        d = rp.get(n); y = d["y"]
        out[n] = dict(raw=permutation(rp.clean(d["raw"]), y, n_perm, label=f"{n} raw"), derived=permutation(rp.clean(d["derived"]), y, n_perm, label=f"{n} derived"))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "farmflow"
    t0 = time.time()
    if which == "farmflow":
        pc, df, y = farmflow_checks(30)
        (OUT / "permutation_farmflow.json").write_text(json.dumps(pc, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        ab = farmflow_ablation(df, y, 20)
        (OUT / "farmflow_ablation.json").write_text(json.dumps(ab, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    else:
        (OUT / "permutation_others.json").write_text(json.dumps(others(20), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done {time.time()-t0:.0f}s")
