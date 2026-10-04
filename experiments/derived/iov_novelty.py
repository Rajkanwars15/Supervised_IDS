"""Item 5: CIC-IoV novelty experiment.

Rows are 8 CAN frames x 17 bits (136 bit-features) and ~99.7% duplicates, so training/prediction use the unique vectors
with multiplicity (sample_weight = row counts; identical to row-level training for a decision tree).
Novelty levels, all measured against the TRAINING rows of the split:
  exact_unseen        : vector absent from train (Hamming distance >= 1)
  hamming>=k (k=2,3,5): nearest training vector differs in >= k of 136 bits
  frame_novel         : at least one of the 8 frame patterns (per position) never seen in train
Splits: (S1) standard stratified row split (80/20); (S2) vector-disjoint split, 80/20 of unique vectors;
        (S3) vector-disjoint split with 50/50 of unique vectors (less strict: larger novel test, smaller train).
Min-size rule for any claim: n_test_rows >= 2000 and n_pos >= 100, else the cell is labelled 'descriptive only'.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import raw_loaders

OUT = Path(__file__).parent / "results"
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
MIN_N, MIN_POS = 2000, 100


def min_hamming(Ute, Utr, chunk=500):
    out = np.empty(len(Ute), np.int32)
    for i in range(0, len(Ute), chunk):
        x = Ute[i:i + chunk][:, None, :] ^ Utr[None, :, :]
        out[i:i + chunk] = POP[x].sum(axis=2).min(axis=1)
    return out


def fit_models(Ub, u_tr, y_tr, w_tr):
    Xtr = Ub[u_tr]
    dt = DecisionTreeClassifier(random_state=0).fit(Xtr, y_tr, sample_weight=w_tr)
    gb = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.1, n_jobs=4, verbose=-1, random_state=0).fit(Xtr, y_tr, sample_weight=w_tr)
    return dict(DT=dt, LGBM=gb)


def evaluate_subset(models, Ub, rows_u, y_rows, mask, label):
    """rows_u: unique-vector index per test row; mask: boolean over test rows."""
    n, npos = int(mask.sum()), int(y_rows[mask].sum())
    cell = dict(n_rows=n, n_pos=npos, claimable=bool(n >= MIN_N and npos >= MIN_POS))
    if n < 20 or npos == 0 or npos == n:
        cell["note"] = "too few rows / single class"; return cell
    ur = rows_u[mask]; yy = y_rows[mask]
    for name, m in models.items():
        prob_u = m.predict_proba(Ub)[:, 1]  # score all unique vectors once
        prob = prob_u[ur]
        r = eu.metrics(yy, prob); r["ci"] = eu.boot_ci(yy, prob, keys=("f1", "mcc", "ap", "fpr"), n=300)
        cell[name] = r
    f = cell.get("DT", {})
    print(f"    {label:28} n={n:>8} pos={npos:>7} {'' if cell['claimable'] else '(descriptive only)'} DT F1 {f.get('f1', float('nan')):.4f} MCC {f.get('mcc', float('nan')):.4f} AP {f.get('ap', float('nan')):.4f}", flush=True)
    return cell


def main():
    t0 = time.time(); d = raw_loaders.iov(); X, y = d["X"], d["y"].to_numpy()
    bits = X.to_numpy(np.uint8); n = len(y)
    packed = np.packbits(bits, axis=1)
    U, first, inv = np.unique(packed, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
    Ub = bits[first].astype(np.float32)  # unique vectors as bit features
    frames = bits.reshape(n, 8, 17)
    print(f"IoV rows={n} unique vectors={len(U)} ({100*len(U)/n:.3f}%)", flush=True)
    res = dict(dataset="cic-iov-2024", n=n, n_unique=int(len(U)), min_rule=dict(n=MIN_N, pos=MIN_POS), splits={})

    def split_s1():
        return train_test_split(np.arange(n), test_size=0.2, random_state=42, stratify=y)

    def split_vec(test_frac, seed=42):
        rng = np.random.RandomState(seed); ids = rng.permutation(len(U)); k = int(test_frac * len(U))
        te_set = np.zeros(len(U), bool); te_set[ids[:k]] = True
        te = np.flatnonzero(te_set[inv]); tr = np.flatnonzero(~te_set[inv])
        assert not (set(inv[tr]) & set(inv[te]))
        return tr, te

    for sname, (tr, te) in {"S1_row_stratified_80_20": split_s1(), "S2_vector_disjoint_80_20": split_vec(0.2), "S3_vector_disjoint_50_50": split_vec(0.5)}.items():
        print(f"== {sname}: train rows {len(tr)} test rows {len(te)} test attack {y[te].mean():.4f}", flush=True)
        agg = pd.DataFrame({"u": inv[tr], "y": y[tr]}).groupby(["u", "y"]).size().reset_index(name="w")
        models = fit_models(Ub, agg.u.to_numpy(), agg.y.to_numpy(), agg.w.to_numpy())
        u_tr = np.unique(inv[tr]); u_te = np.unique(inv[te])
        mh_unique = {u: h for u, h in zip(u_te, min_hamming(U[u_te], U[u_tr]))}
        mh_rows = np.array([mh_unique[u] for u in inv[te]]) if len(te) else np.array([])
        # frame-level novelty: per-position frame patterns present in train
        fr_tr = [set(map(bytes, np.packbits(frames[tr][:, k, :], axis=1))) for k in range(8)] if len(tr) < 3_000_000 else None
        fr_te = np.zeros(len(te), bool)
        for k in range(8):
            p = np.packbits(frames[te][:, k, :], axis=1)
            fr_te |= np.array([bytes(r) not in fr_tr[k] for r in p])
        sp = dict(n_train=int(len(tr)), n_test=int(len(te)), test_attack_rate=float(y[te].mean()), levels={})
        ytest = y[te]; rows_u = inv[te]
        for lvl, mask in [("all_test_rows", np.ones(len(te), bool)), ("exact_unseen(d>=1)", mh_rows >= 1), ("hamming>=2", mh_rows >= 2),
                          ("hamming>=3", mh_rows >= 3), ("hamming>=5", mh_rows >= 5), ("frame_novel(any frame)", fr_te)]:
            sp["levels"][lvl] = evaluate_subset(models, Ub, rows_u, ytest, mask, lvl)
        res["splits"][sname] = sp
        (OUT / "iov_novelty.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
