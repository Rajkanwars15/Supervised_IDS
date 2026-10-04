"""Item 9: cross-dataset transfer matrix on harmonised features (feature set per pair = those computable in BOTH datasets).

Nodes: unsw-nb15, nsl-kdd, cicids2017, farm-flow (flow semantics) and sensornetguard (partial node: pkt/byte rate + error rate only;
its 'error_rate' and rates are node-health quantities, so its cells are descriptive). IoV is excluded (bit-level domain).
Source train <= 300k (stratified, 3 seeds); target test <= 200k (stratified, fixed). Transforms: as-is and per-dataset quantile
(fit label-free on each dataset's OWN train features). Models: DT(max_depth 10) and LightGBM(200). Fixed metric set + chance references.
Shuffled-train-label sanity cells included.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import shared_analysis as sa
import permutation_checks as pc
import robust_per_dataset as rp
from derived_dt_experiment import load

OUT = Path(__file__).parent / "results"
NAMES = ["unsw-nb15", "nsl-kdd", "cicids2017", "farm-flow", "sensornetguard"]
SRC_CAP, TGT_CAP, SEEDS = 300_000, 200_000, 3


def sng_schema(X):
    o = pd.DataFrame(index=X.index, columns=sa.CANDS, dtype=float)
    o["log_byte_rate"] = np.log1p(X["Data_Throughput"].clip(lower=0)); o["log_pkt_rate"] = np.log1p(X["Packet_Rate"].clip(lower=0))
    o["error_rate"] = X["Error_Rate"]
    return o


def get_nodes():
    nodes = {}
    for n in ["unsw-nb15", "nsl-kdd", "cicids2017"]:
        Xtr, Xte, ytr, yte = load(n)
        nodes[n] = dict(Xtr=sa.derive(n, Xtr), Xte=sa.derive(n, Xte), ytr=ytr.reset_index(drop=True), yte=yte.reset_index(drop=True))
        for k in ("Xtr", "Xte"): nodes[n][k] = nodes[n][k].reset_index(drop=True)
    df = pc.ff_raw_df(); y = df["is_attack"].astype(int).reset_index(drop=True)
    _, D = pc.ff_build(df, keep_ports=False)
    # farm-flow harmonised schema via sa.schema (CANDS columns)
    p = df.orig_pkts + df.resp_pkts
    S = sa.schema(df.orig_ip_bytes, df.resp_ip_bytes, df.orig_pkts, df.resp_pkts, df.flow_duration, sa.ratio(df.flow_RST_flag_count, p))
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    nodes["farm-flow"] = dict(Xtr=S.iloc[tr].reset_index(drop=True), Xte=S.iloc[te].reset_index(drop=True), ytr=y.iloc[tr].reset_index(drop=True), yte=y.iloc[te].reset_index(drop=True))
    Xtr, Xte, ytr, yte = load("sensornetguard")
    nodes["sensornetguard"] = dict(Xtr=sng_schema(Xtr).reset_index(drop=True), Xte=sng_schema(Xte).reset_index(drop=True), ytr=ytr.reset_index(drop=True), yte=yte.reset_index(drop=True))
    for n, v in nodes.items():
        v["Xtr"] = v["Xtr"][sa.CANDS].replace([np.inf, -np.inf], np.nan).astype("float32"); v["Xte"] = v["Xte"][sa.CANDS].replace([np.inf, -np.inf], np.nan).astype("float32")
        v["cov"] = v["Xtr"].notna().mean()
        print(f"  node {n}: train {len(v['Xtr'])} test {len(v['Xte'])} attack {v['ytr'].mean():.3f} cols>50%: {list(v['cov'][v['cov']>0.5].index)}", flush=True)
    return nodes


def strat_sub(n, y, cap, seed):
    idx = np.arange(n)
    return idx if n <= cap else train_test_split(idx, train_size=cap, random_state=seed, stratify=y)[0]


def make_transform(mode, Xown):
    med = Xown.median().fillna(0)
    if mode == "asis":
        return lambda Z: Z.fillna(med)
    sub = Xown.fillna(med).iloc[np.random.RandomState(0).choice(len(Xown), min(len(Xown), 200_000), replace=False)]
    q = QuantileTransformer(n_quantiles=200, random_state=0).fit(sub)
    return lambda Z: pd.DataFrame(q.transform(Z.fillna(med)), columns=Z.columns, index=Z.index)


def mkmodel(name, seed):
    return DecisionTreeClassifier(max_depth=10, random_state=seed) if name == "DT" else lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=4, verbose=-1, random_state=seed)


def main():
    t0 = time.time(); nodes = get_nodes(); res = dict(features=sa.CANDS, cells={})
    tgt_idx = {n: strat_sub(len(v["yte"]), v["yte"], TGT_CAP, 0) for n, v in nodes.items()}
    for a in NAMES:
        for b in NAMES:
            feats = [c for c in sa.CANDS if nodes[a]["cov"][c] > 0.5 and nodes[b]["cov"][c] > 0.5]
            key = f"{a}->{b}"
            if len(feats) < 2:
                res["cells"][key] = dict(features=feats, note="N/A (fewer than 2 harmonised features)"); continue
            yte = nodes[b]["yte"].iloc[tgt_idx[b]].to_numpy(); prior = float(yte.mean())
            cell = dict(features=feats, n_test=int(len(yte)), target_prior=prior, chance=eu.chance(prior), runs={})
            for mode in ("asis", "quantile"):
                ta = make_transform(mode, nodes[a]["Xtr"][feats]); tb = make_transform(mode, nodes[b]["Xtr"][feats])
                Xb = tb(nodes[b]["Xte"][feats].iloc[tgt_idx[b]])
                for model in ("DT", "LGBM"):
                    rows = []; boot = None
                    for s in range(SEEDS):
                        si = strat_sub(len(nodes[a]["ytr"]), nodes[a]["ytr"], SRC_CAP, s)
                        Xa = ta(nodes[a]["Xtr"][feats].iloc[si]); ya = nodes[a]["ytr"].iloc[si]
                        clf = mkmodel(model, s).fit(Xa, ya)
                        prob = clf.predict_proba(Xb)[:, 1] if len(clf.classes_) == 2 else np.zeros(len(yte))
                        rows.append(eu.metrics(yte, prob))
                        if s == 0: boot = eu.boot_ci(yte, prob, keys=("f1", "mcc", "ap", "fpr"), n=200)
                    cell["runs"][f"{mode}|{model}"] = dict(mean={k: float(np.nanmean([r[k] for r in rows])) for k in ("f1", "mcc", "ap", "fpr", "recall", "ba")},
                                                          sd={k: float(np.nanstd([r[k] for r in rows], ddof=1)) for k in ("f1", "mcc", "ap", "fpr")}, boot_ci_seed0=boot)
            # shuffled-train-label sanity (as-is, DT) 
            ta = make_transform("asis", nodes[a]["Xtr"][feats]); tb = make_transform("asis", nodes[b]["Xtr"][feats])
            si = strat_sub(len(nodes[a]["ytr"]), nodes[a]["ytr"], SRC_CAP, 0); ys = np.random.RandomState(0).permutation(nodes[a]["ytr"].iloc[si].to_numpy())
            clf = mkmodel("DT", 0).fit(ta(nodes[a]["Xtr"][feats].iloc[si]), ys)
            cell["shuffled_label_mcc"] = eu.metrics(yte, clf.predict_proba(tb(nodes[b]["Xte"][feats].iloc[tgt_idx[b]]))[:, 1])["mcc"]
            res["cells"][key] = cell
            r = cell["runs"]["asis|DT"]["mean"]; q = cell["runs"]["quantile|LGBM"]["mean"]
            print(f"  {key:30} feats={len(feats)} asis-DT F1 {r['f1']:.3f} MCC {r['mcc']:+.3f} | quantile-LGBM F1 {q['f1']:.3f} MCC {q['mcc']:+.3f} | shuffled MCC {cell['shuffled_label_mcc']:+.3f}", flush=True)
            (OUT / "transfer_matrix.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
