"""Raw vs derived-feature decision trees per dataset (same splits as the existing experiments)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from derive_features import REGISTRY

DATA = Path(__file__).parents[2] / "data"
SILVER = {"unsw-nb15": "unsw-nb15-silver/UNSW-NB15", "nsl-kdd": "nsl-kdd-silver/NSL-KDD",
          "cicids2017": "cic-ids2017-silver/CIC-IDS2017", "cic-iov-2024": "cic-iov-2024-silver/CIC-IOV-2024"}
DEPTHS = [3, 5, 7, 10, None]


def load(name):
    if name == "sensornetguard":
        df = pd.read_csv(DATA / "sensornetguard_data.csv")
        y = df.pop("Is_Malicious")
        X = df.drop(columns=["Node_ID", "Timestamp", "IP_Address"])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return Xtr, Xte, ytr, yte
    tr, te = (pd.read_csv(DATA / f"{SILVER[name]}_{s}_Binary.csv") for s in ("Train", "Test"))
    tr.columns, te.columns = tr.columns.astype(str), te.columns.astype(str)
    return tr.drop(columns="is_attack"), te.drop(columns="is_attack"), tr["is_attack"], te["is_attack"]


def prep(Xtr, Xte):
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)
    med = Xtr.median().fillna(0)
    return Xtr.fillna(med), Xte.fillna(med)


def score(Xtr, ytr, Xte, yte, depth):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return dict(depth=depth, acc=accuracy_score(yte, p), f1=f1_score(yte, p),
                auc=roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), leaves=int(clf.tree_.n_leaves),
                top=max(zip(clf.feature_importances_, Xtr.columns))[1])


def main(names):
    out = {}
    for n in names:
        Xtr, Xte, ytr, yte = load(n)
        Dtr, Dte = REGISTRY[n](Xtr), REGISTRY[n](Xte)
        assert len(Dtr) == len(Xtr) and len(Dte) == len(Xte)  # no row collapsing
        Dtr = Dtr.dropna(axis=1, how="all"); Dte = Dte[Dtr.columns]
        out[n] = {"n_derived": Dtr.shape[1], "n_raw": Xtr.shape[1], "raw": [], "derived": []}
        for tag, (a, b) in {"raw": prep(Xtr, Xte), "derived": prep(Dtr, Dte)}.items():
            for d in DEPTHS:
                out[n][tag].append(score(a, ytr, b, yte, d))
        print(n, f"raw={out[n]['n_raw']}f derived={out[n]['n_derived']}f")
        for r, d in zip(out[n]["raw"], out[n]["derived"]):
            print(f"  depth={str(r['depth']):>4}  raw F1 {r['f1']:.4f} ({r['leaves']:>5} leaves, {r['top']})"
                  f" | derived F1 {d['f1']:.4f} ({d['leaves']:>5} leaves, {d['top']})")
    res = Path(__file__).parent / "results"; res.mkdir(exist_ok=True)
    (res / "derived_vs_raw.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or ["sensornetguard", "unsw-nb15", "nsl-kdd", "cicids2017", "cic-iov-2024"])
