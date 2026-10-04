"""Item 3: legitimate group splits for CIC-IDS2017 from raw provenance.

Keys: source IP (flagged degenerate), (src,dst) pair, Flow ID/session, destination host, and time blocks (1/5/15 min, with a +-1 block guard).
A fold is *valid* when test has >= MIN_ATTACK_GROUPS... (here: both classes, test attack rate within TOL of overall, test size ~20%).
Random group assignment with rejection sampling; zero group overlap asserted. Default DT; train capped at TRAIN_CAP rows.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import raw_loaders

OUT = Path(__file__).parent / "results"
TEST_FRAC, TOL, N_FOLDS, MAX_TRY, TRAIN_CAP = 0.2, 0.25, 5, 400, 1_000_000


def make_fold(codes, y, rng, guard_codes=None):
    """codes: integer group id per row. Returns (tr, te, info) or None if no valid assignment found."""
    n = len(y); ya = y.to_numpy()
    ug, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    atk = np.bincount(inv, weights=ya).astype(float)
    overall = ya.mean(); target = TEST_FRAC * n
    for attempt in range(MAX_TRY):
        perm = rng.permutation(len(ug))
        cum = np.cumsum(cnt[perm]); k = np.searchsorted(cum, target) + 1
        sel = perm[:k]
        t_rows, t_atk = cnt[sel].sum(), atk[sel].sum()
        rate = t_atk / t_rows
        if t_atk >= 50 and t_rows - t_atk >= 50 and abs(rate - overall) <= TOL * overall:
            in_test = np.zeros(len(ug), bool); in_test[sel] = True
            te_mask = in_test[inv]
            return te_mask, dict(n_groups=int(len(ug)), n_test_groups=int(len(sel)), attempts=attempt + 1, test_attack_rate=float(rate),
                                 test_attack_groups=int((atk[sel] > 0).sum()))
    return None


def run_fold(X, y, te_mask, drop_mask=None, seed=0):
    tr_mask = ~te_mask if drop_mask is None else ~(te_mask | drop_mask)
    tr = np.flatnonzero(tr_mask); te = np.flatnonzero(te_mask)
    if len(tr) > TRAIN_CAP:
        tr = np.random.RandomState(seed).choice(tr, TRAIN_CAP, replace=False)
    Xtr, Xte = X.iloc[tr], X.iloc[te]; med = Xtr.median().fillna(0)
    clf = DecisionTreeClassifier(random_state=seed).fit(Xtr.fillna(med), y.iloc[tr])
    prob = clf.predict_proba(Xte.fillna(med))[:, 1]
    m = eu.metrics(y.iloc[te], prob); m["n_train"] = int(len(tr)); m["n_test"] = int(len(te))
    return m, tr, te


def main():
    d = raw_loaders.cicids2017(); X, y, ex = d["X"], d["y"], d["extra"]
    n = len(y); print(f"IDS2017 n={n} attack={y.mean():.4f}", flush=True)
    src, dst, flow, ts = d["group"], ex["dst"], ex["flow"], ex["ts"]
    keys = {"source_ip": pd.factorize(src)[0], "src_dst_pair": pd.factorize(src.astype(str) + ">" + dst.astype(str))[0],
            "flow_id_session": pd.factorize(flow)[0], "destination_host": pd.factorize(dst)[0]}
    tsec = (ts.astype("int64") // 10**9).to_numpy()
    for mins in (1, 5, 15):
        blk = tsec // (mins * 60); keys[f"timeblock_{mins}min"] = pd.factorize(blk)[0]; keys[f"_blk_{mins}"] = blk
    res = dict(dataset="cicids2017", n=n, overall_attack_rate=float(y.mean()), tol=TOL, keys={})
    for kname, codes in [(k, v) for k, v in keys.items() if not k.startswith("_")]:
        t0 = time.time(); rng = np.random.RandomState(0); folds = []; failed = 0
        for f in range(N_FOLDS):
            r = make_fold(codes, y, rng)
            if r is None:
                failed += 1; continue
            te_mask, info = r
            guard = None
            if kname.startswith("timeblock"):
                mins = int(kname.split("_")[1].replace("min", "")); blk = keys[f"_blk_{mins}"]
                tb = set(np.unique(blk[te_mask])); near = set(b + o for b in tb for o in (-1, 0, 1))
                guard = np.isin(blk, list(near)) & ~te_mask
            tr_overlap = set(codes[te_mask]) & set(codes[~te_mask if guard is None else ~(te_mask | guard)])
            assert not tr_overlap, f"group overlap {len(tr_overlap)} for {kname}"
            m, tr, te = run_fold(X, y, te_mask, guard, seed=f)
            m.update(info); m["overlap_groups"] = 0; m["guard_rows_dropped"] = int(guard.sum()) if guard is not None else 0
            folds.append(m)
        row = dict(valid_folds=len(folds), failed_folds=failed, folds=folds)
        if folds:
            row["summary"] = eu.summarize(folds, ["f1", "mcc", "ap", "fpr", "recall"])
        res["keys"][kname] = row
        s = row.get("summary")
        print(f"  [{kname}] valid {len(folds)}/{N_FOLDS}" + (f" F1 {s['f1']['mean']:.4f}±{s['f1']['sd']:.4f} MCC {s['mcc']['mean']:.4f} AP {s['ap']['mean']:.4f} FPR {s['fpr']['mean']:.4f} testAtk {np.mean([f['test_attack_rate'] for f in folds]):.3f}" if s else " NO VALID FOLD (attack structure incompatible with this key)") + f" [{time.time()-t0:.0f}s]", flush=True)
        (OUT / "ids2017_groups.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
