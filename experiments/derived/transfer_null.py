"""Item 9 supplement: permutation null for every transfer cell (as-is, DT depth 10).

For each ordered pair A->B: 20 permutations of A's TRAIN labels (source <= 100k rows), fit, score B's test; record mean/SD of MCC and
F1 over permutations. Real MCC is compared against this null (z-score and empirical one-sided p = share of null MCC >= real MCC).
A single shuffled run is not a valid null under distribution shift: with few features a noise-fit tree can land on |MCC| ~ 0.2-0.3 by chance.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import transfer_matrix as tm
import shared_analysis as sa

OUT = Path(__file__).parent / "results"
N_PERM, SRC = 20, 100_000


def main():
    nodes = tm.get_nodes(); real = json.load(open(OUT / "transfer_matrix.json"))["cells"]; res = {}
    tgt_idx = {n: tm.strat_sub(len(v["yte"]), v["yte"], tm.TGT_CAP, 0) for n, v in nodes.items()}
    for a in tm.NAMES:
        for b in tm.NAMES:
            key = f"{a}->{b}"; feats = real[key].get("features", [])
            if "runs" not in real[key]: continue
            ta = tm.make_transform("asis", nodes[a]["Xtr"][feats]); tb = tm.make_transform("asis", nodes[b]["Xtr"][feats])
            yte = nodes[b]["yte"].iloc[tgt_idx[b]].to_numpy(); Xb = tb(nodes[b]["Xte"][feats].iloc[tgt_idx[b]])
            si = tm.strat_sub(len(nodes[a]["ytr"]), nodes[a]["ytr"], SRC, 0); Xa = ta(nodes[a]["Xtr"][feats].iloc[si]); ya = nodes[a]["ytr"].iloc[si].to_numpy()
            rng = np.random.RandomState(0); mcc, f1 = [], []
            for p in range(N_PERM):
                clf = DecisionTreeClassifier(max_depth=10, random_state=p).fit(Xa, rng.permutation(ya))
                m = eu.metrics(yte, clf.predict_proba(Xb)[:, 1] if len(clf.classes_) == 2 else np.zeros(len(yte))); mcc.append(m["mcc"]); f1.append(m["f1"])
            r = real[key]["runs"]["asis|DT"]["mean"]["mcc"]; mcc = np.array(mcc)
            res[key] = dict(n_features=len(feats), real_mcc=r, null_mcc_mean=float(mcc.mean()), null_mcc_sd=float(mcc.std(ddof=1)), null_f1_mean=float(np.mean(f1)),
                            z=float((r - mcc.mean()) / max(mcc.std(ddof=1), 1e-9)), p_one_sided=float((mcc >= r).mean()))
            print(f"  {key:30} real MCC {r:+.3f} | null {mcc.mean():+.3f} ± {mcc.std(ddof=1):.3f} | z {res[key]['z']:+.1f}", flush=True)
            (OUT / "transfer_null.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
