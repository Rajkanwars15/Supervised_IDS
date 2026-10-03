"""Row-wise derivation of a shared, scale-free feature schema (SensorNetGuard-style rates/ratios).

Every derive_* function returns a frame with the same index/length as its input (no row dropped
or aggregated). Features a dataset cannot support are NaN (median-imputed downstream).
"""
import numpy as np
import pandas as pd

SCHEMA = ["pkt_rate", "byte_rate", "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio", "mean_pkt_size",
          "loss_rate", "error_rate", "iat_mean", "log_duration"]


def _div(a, b):
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    out = a / b.where(b > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _frame(index, **cols):
    out = pd.DataFrame(index=index, columns=SCHEMA, dtype=float)
    for k, v in cols.items():
        out[k] = v
    return out


def derive_sensornetguard(d):
    return _frame(d.index, pkt_rate=d["Packet_Rate"], byte_rate=d["Data_Throughput"],
                  loss_rate=d["Packet_Drop_Rate"], error_rate=d["Error_Rate"])


def derive_unsw(d):
    pk, by = d["spkts"] + d["dpkts"], d["sbytes"] + d["dbytes"]
    return _frame(d.index, pkt_rate=_div(pk, d["dur"]), byte_rate=_div(by, d["dur"]),
                  fwd_bwd_pkt_ratio=_div(d["spkts"], d["dpkts"]),
                  fwd_bwd_byte_ratio=_div(d["sbytes"], d["dbytes"]),
                  mean_pkt_size=_div(by, pk), loss_rate=_div(d["sloss"] + d["dloss"], pk),
                  iat_mean=(d["sinpkt"] + d["dinpkt"]) / 2, log_duration=np.log1p(d["dur"]))


def derive_nsl(d):
    # silver columns are NSL-KDD indices: 0 duration, 4 src_bytes, 5 dst_bytes, 24 serror_rate, 26 rerror_rate
    dur, sb, db = d["0"], d["4"], d["5"]
    return _frame(d.index, byte_rate=_div(sb + db, dur), fwd_bwd_byte_ratio=_div(sb, db),
                  error_rate=(d["24"] + d["26"]) / 2, log_duration=np.log1p(dur))


def derive_ids2017(d):
    fp, bp = d["Total Fwd Packets"], d["Total Backward Packets"]
    fb, bb = d["Total Length of Fwd Packets"], d["Total Length of Bwd Packets"]
    dur = d["Flow Duration"].clip(lower=0) / 1e6  # microseconds -> seconds
    return _frame(d.index, pkt_rate=_div(fp + bp, dur), byte_rate=_div(fb + bb, dur),
                  fwd_bwd_pkt_ratio=_div(fp, bp), fwd_bwd_byte_ratio=_div(fb, bb),
                  mean_pkt_size=_div(fb + bb, fp + bp),
                  error_rate=_div(d["RST Flag Count"] + d["FIN Flag Count"], fp + bp),
                  iat_mean=d["Flow IAT Mean"] / 1e6, log_duration=np.log1p(dur))


def derive_iov(d):
    # 8 CAN frames x 17 bit-columns (DATA_<frame><bit>): per-row bit statistics, no flow semantics.
    bits = d[[c for c in d.columns if c.startswith("DATA_")]].to_numpy(float).reshape(len(d), 8, 17)
    return pd.DataFrame({"bit_density": bits.mean((1, 2)), "frame_density_std": bits.mean(2).std(1),
                         "frame_delta": np.abs(np.diff(bits, axis=1)).mean((1, 2)),
                         "bit_position_std": bits.mean(1).std(1)}, index=d.index)


REGISTRY = {"sensornetguard": derive_sensornetguard, "unsw-nb15": derive_unsw, "nsl-kdd": derive_nsl,
            "cicids2017": derive_ids2017, "cic-iov-2024": derive_iov}
