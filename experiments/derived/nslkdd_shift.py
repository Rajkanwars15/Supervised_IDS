"""Item 6: why does NSL-KDD go from ~0.99 (random split) to ~0.80 (benchmark KDDTrain+/KDDTest+)?

Compares: class prior, attack-family distribution (novel families), per-feature shift (KS), adversarial-validation AUC
(train-vs-test separability), family-wise recall, and a counterfactual decomposition of the F1/MCC gap:
  random split  ->  benchmark restricted to seen-family test rows  ->  benchmark with family mix re-weighted to the random split
  ->  full benchmark test.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
import eval_utils as eu

DATA = Path(__file__).parents[2] / "data/NSL-KDD-Dataset"
OUT = Path(__file__).parent / "results"


def read(f):
    d = pd.read_csv(DATA / f, header=None)
    fam = d[41].astype(str); y = (fam != "normal").astype(int)
    X = d.drop(columns=[1, 2, 3, 41, 42]); X.columns = X.columns.astype(str)
    return X.astype("float32"), y, fam


def adv_auc(A, B, seed=0):
    X = pd.concat([A, B]); z = np.r_[np.zeros(len(A)), np.ones(len(B))]
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=100, n_jobs=4, verbose=-1, random_state=seed), X, z, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(z, p))


def wmetrics(y, pred, w=None):
    return dict(f1=float(f1_score(y, pred, sample_weight=w)), mcc=float(matthews_corrcoef(y, pred, sample_weight=w)))


def main():
    Xtr, ytr, ftr = read("KDDTrain+.txt"); Xte, yte, fte = read("KDDTest+.txt"); X21, y21, f21 = read("KDDTest-21.txt")
    res = dict(n_train=len(Xtr), n_test=len(Xte), n_test21=len(X21))
    # (a) class prior
    Xall = pd.concat([Xtr, Xte], ignore_index=True); yall = pd.concat([ytr, yte], ignore_index=True); fall = pd.concat([ftr, fte], ignore_index=True)
    itr, ite = train_test_split(np.arange(len(yall)), test_size=0.2, random_state=42, stratify=yall)
    res["class_prior"] = dict(train=float(ytr.mean()), benchmark_test=float(yte.mean()), test21=float(y21.mean()), pooled=float(yall.mean()), random_test=float(yall.iloc[ite].mean()))
    # (b) families
    train_fams = set(ftr[ytr == 1]); test_atk = fte[yte == 1]
    novel_mask_te = (yte == 1) & ~fte.isin(train_fams)
    res["families"] = dict(n_train_families=len(train_fams), n_test_families=int(test_atk.nunique()),
        novel_test_families=sorted(set(test_atk) - train_fams), n_novel_test_attack_rows=int(novel_mask_te.sum()),
        share_of_test_attack_rows_novel=float(novel_mask_te.sum() / (yte == 1).sum()), share_of_all_test_rows_novel=float(novel_mask_te.mean()),
        train_family_share=ftr[ytr == 1].value_counts(normalize=True).head(8).round(4).to_dict(), test_family_share=test_atk.value_counts(normalize=True).head(8).round(4).to_dict())
    # (c) covariate shift
    ks = {c: ks_2samp(np.log1p(Xtr[c].clip(lower=0)), np.log1p(Xte[c].clip(lower=0))).statistic for c in Xtr.columns}
    res["feature_ks_top10"] = dict(sorted(ks.items(), key=lambda kv: -kv[1])[:10])
    res["adversarial_auc"] = dict(train_vs_benchmark_test=adv_auc(Xtr, Xte),
        random_train_vs_random_test=adv_auc(Xall.iloc[itr], Xall.iloc[ite]),
        train_vs_benchmark_test_benign_only=adv_auc(Xtr[ytr == 0], Xte[yte == 0]), train_vs_benchmark_test_attack_only=adv_auc(Xtr[ytr == 1], Xte[yte == 1]))
    # (d) models + family-wise recall on the benchmark split
    med = Xtr.median().fillna(0)
    out = {}
    for name, mk in (("DT", lambda: DecisionTreeClassifier(random_state=42)), ("LGBM", lambda: lgb.LGBMClassifier(n_estimators=200, n_jobs=4, verbose=-1, random_state=42))):
        clf = mk().fit(Xtr.fillna(med), ytr); prob = clf.predict_proba(Xte.fillna(med))[:, 1]; pred = (prob >= 0.5).astype(int)
        fam_recall = {f: float((pred[(fte == f).to_numpy()] == 1).mean()) for f in sorted(set(test_atk)) if (fte == f).sum() >= 20}
        # random-split reference (same model class)
        clf_r = mk().fit(Xall.iloc[itr].fillna(med), yall.iloc[itr]); prob_r = clf_r.predict_proba(Xall.iloc[ite].fillna(med))[:, 1]; pred_r = (prob_r >= 0.5).astype(int)
        yb = yte.to_numpy(); ben = yb == 0
        seen_fam_atk = (yb == 1) & ~novel_mask_te.to_numpy()
        keep_seen = ben | seen_fam_atk                       # drop novel-family attacks, keep all benign
        # weights: match random-split attack-family mix for seen families and its class prior
        pr = fall.iloc[ite][yall.iloc[ite] == 1].value_counts(normalize=True); pb = fte[seen_fam_atk].value_counts(normalize=True)
        w = np.zeros(len(yb)); fam_arr = fte.to_numpy()
        for i in np.flatnonzero(seen_fam_atk): w[i] = pr.get(fam_arr[i], 0.0) / max(pb.get(fam_arr[i], 1e-9), 1e-9)
        atk_prior_rand = float(yall.iloc[ite].mean())
        wa = w[seen_fam_atk].sum(); wb = ben.sum()
        w[seen_fam_atk] *= (atk_prior_rand / (1 - atk_prior_rand)) * wb / wa  # attack:benign weight ratio = random prior ratio
        w[ben] = 1.0
        out[name] = dict(
            random_split=wmetrics(yall.iloc[ite], pred_r),
            benchmark_full=wmetrics(yb, pred),
            benchmark_seen_families_only=wmetrics(yb[keep_seen], pred[keep_seen]),
            benchmark_seen_families_reweighted_to_random_mix=wmetrics(yb[keep_seen], pred[keep_seen], w[keep_seen]),
            recall_seen_family_attacks=float((pred[seen_fam_atk] == 1).mean()), recall_novel_family_attacks=float((pred[novel_mask_te.to_numpy()] == 1).mean()),
            benign_fpr=float((pred[ben] == 1).mean()), family_recall=fam_recall,
            benchmark_test21=wmetrics(y21.to_numpy(), (clf.predict_proba(X21.fillna(med))[:, 1] >= 0.5).astype(int)))
        # exact-seen rows
        seen = np.isin(eu.row_hash(Xte), eu.row_hash(Xtr))
        out[name]["f1_on_exact_seen_rows"] = wmetrics(yb[seen], pred[seen]) if seen.sum() > 20 and 0 < yb[seen].sum() < seen.sum() else None
        out[name]["n_exact_seen"] = int(seen.sum())
        g = out[name]
        print(f"  [{name}] random F1 {g['random_split']['f1']:.4f} MCC {g['random_split']['mcc']:.4f} | benchmark F1 {g['benchmark_full']['f1']:.4f} MCC {g['benchmark_full']['mcc']:.4f} | seen-families-only F1 {g['benchmark_seen_families_only']['f1']:.4f} | reweighted F1 {g['benchmark_seen_families_reweighted_to_random_mix']['f1']:.4f} | recall seen {g['recall_seen_family_attacks']:.3f} novel {g['recall_novel_family_attacks']:.3f} FPR {g['benign_fpr']:.3f}", flush=True)
    res["models"] = out
    # partition check
    res["partition_check"] = dict(n_test=len(yte), benign=int((yte == 0).sum()), seen_family_attacks=int(((yte == 1) & ~novel_mask_te).sum()), novel_family_attacks=int(novel_mask_te.sum()))
    assert res["partition_check"]["benign"] + res["partition_check"]["seen_family_attacks"] + res["partition_check"]["novel_family_attacks"] == len(yte)
    (OUT / "nslkdd_shift.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: res[k] for k in ("class_prior", "adversarial_auc")}, indent=1), res["families"]["share_of_test_attack_rows_novel"], res["families"]["novel_test_families"])


if __name__ == "__main__":
    main()
