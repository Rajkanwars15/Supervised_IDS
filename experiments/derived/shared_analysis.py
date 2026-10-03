"""Stage 2: (1) which derived features are genuinely shared across flow datasets, (2) cross-dataset transfer.

Flow group: UNSW-NB15, NSL-KDD, CIC-IDS2017 (silver splits) and Farm-Flow (raw monthly CSVs, stratified 80/20).
All frames are row-for-row; features a dataset cannot support are NaN.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from derived_dt_experiment import load

DATA = Path(__file__).parents[2] / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
CANDS = ["log_duration", "log_src_bytes", "log_dst_bytes", "log_total_bytes", "src_byte_share",
         "log_byte_rate", "log_mean_pkt_size", "src_pkt_share", "log_pkt_rate", "error_rate"]
NAMES = ["unsw-nb15", "nsl-kdd", "cicids2017", "farm-flow"]
RNG = np.random.RandomState(42)


def share(a, b):
    t = (a + b).astype(float)
    return (a / t.where(t > 0)).replace([np.inf, -np.inf], np.nan)


def ratio(a, b):
    return (a / b.astype(float).where(b > 0)).replace([np.inf, -np.inf], np.nan)


def schema(sb, db, sp=None, dp=None, dur=None, err=None):
    """sb/db: src/dst bytes; sp/dp: packets; dur: seconds; err: per-row error rate."""
    idx = sb.index
    o = pd.DataFrame(index=idx, columns=CANDS, dtype=float)
    tb = sb + db
    o["log_src_bytes"], o["log_dst_bytes"], o["log_total_bytes"] = np.log1p(sb), np.log1p(db), np.log1p(tb)
    o["src_byte_share"] = share(sb, db)
    if dur is not None:
        o["log_duration"] = np.log1p(dur.clip(lower=0))
        o["log_byte_rate"] = np.log1p(ratio(tb, dur.clip(lower=0)).fillna(0))
    if sp is not None:
        tp = sp + dp
        o["src_pkt_share"] = share(sp, dp)
        o["log_mean_pkt_size"] = np.log1p(ratio(tb, tp).fillna(0))
        if dur is not None:
            o["log_pkt_rate"] = np.log1p(ratio(tp, dur.clip(lower=0)).fillna(0))
    if err is not None:
        o["error_rate"] = err
    return o


def derive(name, X):
    if name == "unsw-nb15":
        return schema(X.sbytes, X.dbytes, X.spkts, X.dpkts, X.dur)
    if name == "nsl-kdd":
        return schema(X["4"], X["5"], dur=X["0"], err=(X["24"] + X["26"]) / 2)
    if name == "cicids2017":
        p = X["Total Fwd Packets"] + X["Total Backward Packets"]
        return schema(X["Total Length of Fwd Packets"], X["Total Length of Bwd Packets"],
                      X["Total Fwd Packets"], X["Total Backward Packets"], X["Flow Duration"] / 1e6,
                      ratio(X["RST Flag Count"], p))
    raise ValueError(name)


def farmflow():
    cols = ["flow_duration", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "flow_RST_flag_count", "is_attack"]
    df = pd.concat(pd.read_csv(f, usecols=cols) for f in sorted((DATA / "farm-flow/Datasets").glob("*/*_Farm-Flows.csv")))
    df = df.reset_index(drop=True)
    y = df.pop("is_attack")
    p = df.orig_pkts + df.resp_pkts
    D = schema(df.orig_ip_bytes, df.resp_ip_bytes, df.orig_pkts, df.resp_pkts, df.flow_duration, ratio(df.flow_RST_flag_count, p))
    assert len(D) == len(y)
    return train_test_split(D, y, test_size=0.2, random_state=42, stratify=y)


def build():
    sets = {}
    for n in NAMES[:3]:
        Xtr, Xte, ytr, yte = load(n)
        sets[n] = (derive(n, Xtr), derive(n, Xte), ytr.reset_index(drop=True), yte.reset_index(drop=True))
        assert len(sets[n][0]) == len(ytr) and len(sets[n][1]) == len(yte)
        sets[n] = (sets[n][0].reset_index(drop=True), sets[n][1].reset_index(drop=True), sets[n][2], sets[n][3])
    Dtr, Dte, ytr, yte = farmflow()
    sets["farm-flow"] = (Dtr.reset_index(drop=True), Dte.reset_index(drop=True), ytr.reset_index(drop=True), yte.reset_index(drop=True))
    return sets


def sub(D, y, n):
    i = RNG.choice(len(D), min(n, len(D)), replace=False)
    return D.iloc[i].reset_index(drop=True), y.iloc[i].reset_index(drop=True)


def part1(sets):
    cov, auc = {}, {}
    for n, (Dtr, _, ytr, _) in sets.items():
        D, y = sub(Dtr, ytr, 100000)
        cov[n] = D.notna().mean().round(2)
        a = {}
        for c in CANDS:
            m = D[c].notna()
            a[c] = roc_auc_score(y[m], D.loc[m, c]) - 0.5 if m.mean() > 0.5 and y[m].nunique() == 2 and D.loc[m, c].nunique() > 1 else np.nan
        auc[n] = pd.Series(a)
    cov, auc = pd.DataFrame(cov), pd.DataFrame(auc)
    print("== coverage (fraction non-null)"); print(cov.to_string())
    print("\n== signed AUC-0.5 (positive: attacks have higher value)"); print(auc.round(2).to_string())
    # benign-distribution shift: mean pairwise KS of benign rows between datasets
    shift = {}
    ben = {n: sub(s[0][s[2] == 0].reset_index(drop=True), s[2][s[2] == 0].reset_index(drop=True), 30000)[0] for n, s in sets.items()}
    for c in CANDS:
        ks = [ks_2samp(ben[a][c].dropna(), ben[b][c].dropna()).statistic for i, a in enumerate(NAMES) for b in NAMES[i + 1:]
              if ben[a][c].notna().any() and ben[b][c].notna().any()]
        shift[c] = np.mean(ks) if ks else np.nan
    shift = pd.Series(shift)
    print("\n== mean pairwise KS of benign distributions (0 = aligned, 1 = disjoint)"); print(shift.round(2).to_string())
    # selection
    rows = []
    for c in CANDS:
        v = auc.loc[c].dropna()
        ok = len(v) >= 3 and (v.abs() > 0.1).all() and (np.sign(v) == np.sign(v.iloc[0])).all()
        rows.append((c, len(v), ok))
    all4 = [c for c in CANDS if cov.loc[c].min() > 0.5]
    short = [c for c, k, ok in rows if ok]
    print("\navailable in all 4:", all4); print("consistent-direction shortlist (>=3 datasets, |AUC-.5|>.1, same sign):", short)
    # domain classifier: which dataset does a benign row come from?
    for tag, fs in [("all-4 features", all4), ("shortlist", short)]:
        if not fs: continue
        X = pd.concat([ben[n][fs].assign(d=n) for n in NAMES]).fillna(0)
        Xa, Xb, ya, yb = train_test_split(X[fs], X["d"], test_size=0.3, random_state=42, stratify=X["d"])
        acc = DecisionTreeClassifier(max_depth=6, random_state=42).fit(Xa, ya).score(Xb, yb)
        print(f"domain classifier on benign rows using {tag}: acc {acc:.3f} (chance 0.25)")
    return all4, short, dict(coverage=cov.to_dict(), auc=auc.to_dict(), ks=shift.to_dict())


def prepare(sets, feats, mode):
    out = {}
    for n, (Dtr, Dte, ytr, yte) in sets.items():
        a, b = Dtr[feats], Dte[feats]
        fill = a.median().fillna(0)
        if mode == "quantile":
            q = QuantileTransformer(n_quantiles=200, subsample=200000, random_state=42).fit(a.fillna(fill))
            f = lambda Z: pd.DataFrame(q.transform(Z.fillna(fill)), columns=feats)
        else:
            f = lambda Z: Z.fillna(fill)
        out[n] = (f(a), f(b), ytr, yte)
    return out


def fit(X, y, depth):
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42).fit(X, y)


def ev(clf, X, y):
    p = clf.predict(X)
    return dict(f1=f1_score(y, p), rec=recall_score(y, p), auc=roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def part2(sets, feats, label):
    res = {}
    for mode in ["asis", "quantile"]:
        P = prepare(sets, feats, mode)
        for depth in [5, 10]:
            T = pd.DataFrame(index=NAMES, columns=NAMES, dtype=float)
            for a in NAMES:
                clf = fit(*[P[a][0], P[a][2]], depth)
                for b in NAMES:
                    T.loc[a, b] = ev(clf, P[b][1], P[b][3])["f1"]
            L = {}
            for held in NAMES:
                parts = [sub(P[n][0], P[n][2], 150000) for n in NAMES if n != held]
                clf = fit(pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts]), depth)
                L[held] = ev(clf, P[held][1], P[held][3])
            pooled = {}
            for use_id in [False, True]:
                parts = [sub(P[n][0], P[n][2], 150000) + (n,) for n in NAMES]
                add = lambda X, n: X.assign(**{f"id_{m}": float(m == n) for m in NAMES}) if use_id else X
                Xtr = pd.concat([add(p[0], p[2]) for p in parts]); ytr = pd.concat([p[1] for p in parts])
                clf = fit(Xtr, ytr, depth)
                pooled["with_id" if use_id else "no_id"] = {n: ev(clf, add(P[n][1], n), P[n][3])["f1"] for n in NAMES}
            print(f"\n=== {label} | {mode} | depth {depth}\ntransfer F1 (rows=train, cols=test)\n{T.round(2).to_string()}")
            print("LODO:", {k: {m: round(v, 3) for m, v in d.items()} for k, d in L.items()})
            print("pooled F1 per dataset:", {k: {n: round(v, 3) for n, v in d.items()} for k, d in pooled.items()})
            res[f"{label}|{mode}|{depth}"] = dict(transfer=T.to_dict(), lodo=L, pooled=pooled)
    return res


if __name__ == "__main__":
    sets = build()
    print({n: (len(s[0]), len(s[1]), round(float(s[2].mean()), 2)) for n, s in sets.items()})
    all4, short, p1 = part1(sets)
    out = {"part1": p1, "all4": all4, "short": short, "part2": {}}
    for label, fs in [("all4", all4)] + ([("shortlist", short)] if short and set(short) != set(all4) else []):
        out["part2"].update(part2(sets, fs, label))
    (OUT / "shared_analysis.json").write_text(json.dumps(out, indent=2, default=str))
