"""Derived-feature adapters for the provenance-aware raw_loaders frames (so raw-vs-derived can be run on non-IID protocols).

derived(name, X) -> DataFrame, same rows/order as X. Mirrors robust_per_dataset.get(): flow datasets use the two overlapping
derivations (a_* from derive_features.REGISTRY, b_* from shared_analysis.derive); Farm-Flow uses shared_analysis.schema.
"""
import numpy as np, pandas as pd
import derive_features as df_
import shared_analysis as sa

UNSW_RENAME = {"Spkts": "spkts", "Dpkts": "dpkts", "Sintpkt": "sinpkt", "Dintpkt": "dinpkt", "Sload": "sload", "Dload": "dload",
               "Sjit": "sjit", "Djit": "djit", "smeansz": "smean", "dmeansz": "dmean"}


def _clean(D):
    D = D.replace([np.inf, -np.inf], np.nan).astype("float32")
    D = D.dropna(axis=1, how="all")
    return D.loc[:, D.nunique() > 1]


def derived(name, X):
    if name == "sensornetguard":
        return _clean(df_.REGISTRY[name](X))
    if name == "cic-iov-2024":
        return _clean(df_.derive_iov(X))
    if name == "farm-flow":
        p = X.orig_pkts + X.resp_pkts
        return _clean(sa.schema(X.orig_ip_bytes, X.resp_ip_bytes, X.orig_pkts, X.resp_pkts, X.flow_duration, sa.ratio(X.flow_RST_flag_count, p)))
    if name == "unsw-nb15":
        Xs = X.rename(columns=UNSW_RENAME)
        a = df_.derive_unsw(Xs).add_prefix("a_")
        b = sa.schema(Xs.sbytes, Xs.dbytes, Xs.spkts, Xs.dpkts, Xs.dur).add_prefix("b_")
        return _clean(pd.concat([a, b], axis=1))
    if name in ("nsl-kdd", "cicids2017"):
        a = df_.REGISTRY[name](X).add_prefix("a_"); b = sa.derive(name, X).add_prefix("b_")
        return _clean(pd.concat([a, b], axis=1))
    raise ValueError(name)
