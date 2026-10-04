"""Row-aligned raw loaders that keep provenance: returns dict(X, y, group, time, family) per dataset.

X uses the same feature definitions as the silver experiments where possible (differences noted per loader).
time: sortable numeric (None if the dataset has no temporal info); group: entity key whose rows must not straddle
a train/test boundary (None if unavailable); family: attack-type string ('benign' for y==0).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).parents[2] / "data"


def _out(X, y, group=None, time=None, family=None, note=""):
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32").reset_index(drop=True)
    n = len(X)
    mk = lambda v: None if v is None else pd.Series(np.asarray(v)).reset_index(drop=True)
    d = dict(X=X, y=pd.Series(np.asarray(y)).astype(int).reset_index(drop=True), group=mk(group), time=mk(time), family=mk(family), note=note)
    for k in ("group", "time", "family"):
        assert d[k] is None or len(d[k]) == n
    assert len(d["y"]) == n
    return d


def sensornetguard():
    df = pd.read_csv(DATA / "sensornetguard_data.csv")
    y = df.pop("Is_Malicious")
    t = pd.to_datetime(df["Timestamp"], format="%d-%m-%y %H:%M").astype("int64")
    X = df.drop(columns=["Node_ID", "Timestamp", "IP_Address"])
    return _out(X, y, group=None, time=t, note="synthetic; Node_ID is unique per row (no group structure); Timestamp as time")


def cicids2017():
    fs = sorted(glob.glob(str(DATA / "CIC IDS2017/TrafficLabelling /*.csv")))
    parts = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False, encoding="latin1")
        d.columns = [c.strip() for c in d.columns]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    blank = d["Label"].isna() | d["Flow ID"].isna()  # fully empty trailing lines in the raw CSVs, not flows
    print(f"   [cicids2017 loader] dropping {int(blank.sum())} blank lines of {len(d)}", flush=True)
    d = d[~blank].reset_index(drop=True)
    lab = d.pop("Label").astype(str).str.strip()
    ts = pd.to_datetime(d["Timestamp"], format="mixed", errors="coerce")
    grp = d["Source IP"]
    X = d.drop(columns=["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    y = (lab.str.upper() != "BENIGN").astype(int)
    fam = np.where(y == 1, lab, "benign")
    return _out(X, y, group=grp, time=ts.astype("int64"), family=fam,
                note="Destination Port kept as feature (as in silver); duplicates NOT removed; row count differs from silver if silver dropped NaN/inf rows")


def farmflow():
    fs = sorted((DATA / "farm-flow/Datasets").glob("*/*_Farm-Flows.csv"))
    parts = []
    for i, f in enumerate(fs):
        d = pd.read_csv(f, low_memory=False); d["_month"] = i; d["_row"] = np.arange(len(d)); parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    y = d.pop("is_attack"); fam = d.pop("traffic").astype(str)
    grp = d["id.orig_h"]; time = d["_month"] * 10_000_000 + d["_row"]  # file order within month as time proxy
    X = d.select_dtypes("number").drop(columns=["_month", "_row", "id.orig_p", "id.resp_p"], errors="ignore")
    return _out(X, y, group=grp, time=time, family=np.where(y == 1, fam, "benign"),
                note="no timestamp column: month + row order used as time proxy; ports excluded from features")


def iov():
    fs = sorted(glob.glob(str(DATA / "CICIoV2024/binary_*.csv")))
    parts = []
    for f in fs:
        d = pd.read_csv(f); d["_src"] = Path(f).stem.replace("binary_", ""); d["_row"] = np.arange(len(d)); parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    y = d["label"].map(lambda v: 0 if str(v).lower() in ("benign", "0") else 1) if d["label"].dtype == object else (d["label"] != 0).astype(int)
    fam = np.where(y == 1, d["_src"], "benign")
    X = d[[c for c in d.columns if c.startswith("DATA_")]]
    return _out(X, y, group=None, time=d["_row"], family=fam,
                note="DATA_* only (as silver); CAN ID bits ID0-16 excluded; time = row order within each source file")


def nslkdd():
    cols = [None] * 43
    d = pd.concat([pd.read_csv(DATA / f"NSL-KDD-Dataset/{f}", header=None) for f in ("KDDTrain+.txt", "KDDTest+.txt")], ignore_index=True)
    lab = d[41].astype(str)
    X = d.drop(columns=[1, 2, 3, 41, 42]); X.columns = X.columns.astype(str)
    y = (lab != "normal").astype(int)
    return _out(X, y, family=np.where(y == 1, lab, "benign"), note="no time/group info; categoricals 1-3 dropped as in silver; train+test pooled")


def unsw_raw():
    names = pd.read_csv(DATA / "UNSW-NB15/NUSW-NB15_features.csv", encoding="latin1").iloc[:, 1].str.strip().tolist()
    d = pd.concat([pd.read_csv(DATA / f"UNSW-NB15/UNSW-NB15_{i}.csv", header=None, names=names, low_memory=False) for i in range(1, 5)], ignore_index=True)
    y = d.pop("Label"); fam = d.pop("attack_cat").fillna("benign").astype(str).str.strip()
    grp = d["srcip"]; t = pd.to_numeric(d["Stime"], errors="coerce")
    X = d.drop(columns=["srcip", "sport", "dstip", "dsport", "proto", "state", "service", "Stime", "Ltime"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    return _out(X, y, group=grp, time=t, family=np.where(y == 1, fam, "benign"),
                note="raw 4-file UNSW (2.5M rows, ~42 numeric features) - NOT the 257k train/test-set partition used in silver")


LOADERS = dict(sensornetguard=sensornetguard, cicids2017=cicids2017, **{"farm-flow": farmflow}, **{"cic-iov-2024": iov},
               **{"nsl-kdd": nslkdd}, **{"unsw-nb15": unsw_raw})
