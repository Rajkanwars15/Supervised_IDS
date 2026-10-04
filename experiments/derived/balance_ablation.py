"""Secondary robustness experiment: how sensitive are the conclusions to how class imbalance is handled in TRAINING?

Training strategies (everything else identical: split, model class + hyper-parameters, seeds, features):
  standard : train on the (capped) training rows as they are
  sampler  : torchsampler.ImbalancedDatasetSampler applied to the training indices only (weights = 1/class count, len(train) draws
             WITH replacement), i.e. the training distribution is re-balanced to ~50/50; test rows are never touched
  weighted : class-weighted loss (class_weight='balanced') on the unmodified training rows
Protocols (from compression_noniid.splits_for): random (5 resplits), dedup_unseen (vector-disjoint, 3), time (1), group (entity / 5-min block, 3).
Feature sets: raw and derived. Models: DT (default) and LightGBM(200 trees, 63 leaves). Seeds: fold k uses seed 42+k for model AND sampler.
Metrics: F1, MCC, PR-AUC, precision, attack recall, benign recall / FPR, confusion matrix. A label-shuffled variant (random split, DT) checks
how much of the 'chance' F1 comes from the test prior vs the training distribution.
Note: the sampler changes the training sampling distribution only; it does not change the dataset distribution or "solve" imbalance.
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu
import raw_loaders, derive_raw
import compression_noniid as cn
from torchsampler import ImbalancedDatasetSampler

OUT = Path(__file__).parent / "results"
TRAIN_CAP = 300_000
STRATEGIES = ["standard", "sampler", "weighted"]


def sample_indices(tr, y, seed):
    """torchsampler.ImbalancedDatasetSampler over the training indices only; returns indices into the full frame."""
    torch.manual_seed(seed)
    labels = y.iloc[tr].tolist()
    s = ImbalancedDatasetSampler(dataset=list(range(len(tr))), labels=labels, indices=list(range(len(tr))))
    return np.asarray(tr)[np.fromiter(iter(s), dtype=np.int64, count=len(tr))]


def make_model(model, seed, weighted):
    cw = "balanced" if weighted else None
    if model == "DT":
        return DecisionTreeClassifier(random_state=seed, class_weight=cw)
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.1, n_jobs=4, verbose=-1, random_state=seed, class_weight=cw)


def full_metrics(y, prob):
    y = np.asarray(y); m = eu.metrics(y, prob); pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m.update(precision=tp / max(tp + fp, 1), benign_recall=tn / max(tn + fp, 1), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), pred_attack_rate=float(pred.mean()))
    return m


def fit_predict(strategy, model, F, y, tr, te, seed, y_train_override=None):
    if len(tr) > TRAIN_CAP:
        tr = np.random.RandomState(seed).choice(tr, TRAIN_CAP, replace=False)
    ytr_all = y if y_train_override is None else y_train_override
    rows = sample_indices(tr, ytr_all, seed) if strategy == "sampler" else np.asarray(tr)
    Xtr = F.iloc[rows]; med = F.iloc[tr].median().fillna(0)   # imputation medians from the ORIGINAL training rows (same across strategies)
    clf = make_model(model, seed, strategy == "weighted").fit(Xtr.fillna(med), ytr_all.iloc[rows])
    prob = clf.predict_proba(F.iloc[te].fillna(med))[:, 1] if len(clf.classes_) == 2 else np.zeros(len(te))
    return prob, dict(train_attack_rate=float(ytr_all.iloc[rows].mean()), n_train=int(len(rows)))


def shuffled_check(name, X, y, n_perm=10):
    """Random split, DT, label-permuted training labels (test labels real): F1/MCC under each strategy."""
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    if len(tr) > TRAIN_CAP: tr = np.random.RandomState(0).choice(tr, TRAIN_CAP, replace=False)
    out = {s: [] for s in STRATEGIES}
    for p in range(n_perm):
        ys = y.copy(); ys.iloc[tr] = np.random.RandomState(p).permutation(y.iloc[tr].to_numpy())
        for s in STRATEGIES:
            prob, _ = fit_predict(s, "DT", X, y, tr, te, p, y_train_override=ys)
            out[s].append(full_metrics(y.iloc[te], prob))
    return {s: {k: dict(mean=float(np.nanmean([r[k] for r in rows])), sd=float(np.nanstd([r[k] for r in rows], ddof=1))) for k in ("f1", "mcc", "ap", "fpr", "recall", "pred_attack_rate")} for s, rows in out.items()} | dict(test_prior=float(y.iloc[te].mean()))


def main(names):
    for name in names:
        t0 = time.time(); d = raw_loaders.LOADERS[name](); X, y = d["X"], d["y"]
        D = derive_raw.derived(name, X); H = eu.row_hash(X)
        res = dict(dataset=name, n=len(y), attack_rate=float(y.mean()), n_raw=X.shape[1], n_derived=D.shape[1], train_cap=TRAIN_CAP, protocols={})
        print(f"== {name}: n={len(y):,} attack={y.mean():.3f} raw {X.shape[1]} derived {D.shape[1]}", flush=True)
        for proto, sp in cn.splits_for(name, d, H).items():
            res["protocols"][proto] = {}
            for fsname, F in (("raw", X), ("derived", D)):
                for model in ("DT", "LGBM"):
                    runs = {s: [] for s in STRATEGIES}
                    for k, (tr, te) in enumerate(sp):
                        yt = y.iloc[te]
                        for s in STRATEGIES:
                            prob, info = fit_predict(s, model, F, y, tr, te, 42 + k)
                            m = full_metrics(yt, prob); m.update(info); m["fold"] = k; runs[s].append(m)
                    cell = dict(runs=runs, n_folds=len(sp))
                    for s in ("sampler", "weighted"):   # paired deltas vs standard (same fold, same seed)
                        cell[f"delta_{s}"] = {k: dict(mean=float(np.nanmean([a[k] - b[k] for a, b in zip(runs[s], runs["standard"])])),
                                                      sd=float(np.nanstd([a[k] - b[k] for a, b in zip(runs[s], runs["standard"])], ddof=1)) if len(sp) > 1 else float("nan"))
                                              for k in ("f1", "mcc", "ap", "precision", "recall", "fpr")}
                    res["protocols"][proto][f"{fsname}|{model}"] = cell
                    a, b, c = (np.mean([r["f1"] for r in runs[s]]) for s in STRATEGIES)
                    fa, fb, fc = (np.mean([r["fpr"] for r in runs[s]]) for s in STRATEGIES)
                    ra, rb, rc = (np.mean([r["recall"] for r in runs[s]]) for s in STRATEGIES)
                    print(f"  {proto:22} {fsname:7}{model:5} F1 std {a:.4f} | sampler {b:.4f} | weighted {c:.4f}   recall {ra:.3f}/{rb:.3f}/{rc:.3f}   FPR {fa:.4f}/{fb:.4f}/{fc:.4f}", flush=True)
            (OUT / f"balance_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        res["shuffled_label_check"] = {"raw": shuffled_check(name, X, y)}
        sh = res["shuffled_label_check"]["raw"]
        print("  shuffled-label (raw, DT, 10 perms): " + " | ".join(f"{s}: F1 {sh[s]['f1']['mean']:.3f} MCC {sh[s]['mcc']['mean']:+.3f} predAtk {sh[s]['pred_attack_rate']['mean']:.2f}" for s in STRATEGIES) + f"  (test prior {sh['test_prior']:.3f})", flush=True)
        (OUT / f"balance_{name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"   [{name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(raw_loaders.LOADERS))
