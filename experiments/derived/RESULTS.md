# Derived-feature & per-dataset decision-tree experiments

All numbers are generated from `experiments/derived/results/*.json` by `write_report.py`. Scripts: `derive_features.py`, `derived_dt_experiment.py`, `shared_analysis.py`, `robust_per_dataset.py`, `summarize_robust.py`.

## Summary

- A decision tree per dataset works. A single shared tree across datasets does not (Part 2).
- Derived rate/ratio features reach close to the raw-feature baseline with 4-18 features instead of 17-78 (Part 1), row-for-row (no rows collapsed).
- Per-dataset tuning (feature set × depth × min-leaf × criterion × class-weight, chosen by CV on train only) gives small gains only where the baseline had room: NSL-KDD official (+2.5 F1 pts) and UNSW-NB15 official (+0.5). Elsewhere it is at the ceiling.
- NSL-KDD's official split is a distribution-shift problem: random re-splits give ~0.993 F1 vs ~0.80 on the official split.
- Kyoto was removed: its silver train/test sets are 100% attack (label mapping appears inverted) and columns 14-16 look like IDS-detection flags (leakage).

## Part 3 (final): robust per-dataset results, 5 iterations

Protocol: **official** = the dataset's own train/test split (iterations vary only tuning/tree seed, so intervals understate true uncertainty); **random** = 5 different stratified 80/20 re-splits of the pooled data (the honest variance estimate). Cells are mean ± 95% t-interval over 5 iterations. Baseline = untuned DecisionTree on raw features, same split.

| Dataset | Protocol | Baseline F1 | Tuned F1 | Tuned precision | Tuned recall | Tuned MCC | Tuned AUC | Tuned AP |
|---|---|---|---|---|---|---|---|---|
| cic-iov-2024 | official | 0.9999 ± 0.0000 | **0.9999 ± 0.0000** | 1.0000 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| cic-iov-2024 | random | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| cicids2017 | official | 0.9972 ± 0.0000 | **0.9978 ± 0.0001** | 0.9964 ± 0.0004 | 0.9993 ± 0.0003 | 0.9973 ± 0.0001 | 0.9996 ± 0.0001 | 0.9985 ± 0.0004 |
| cicids2017 | random | 0.9972 ± 0.0002 | **0.9972 ± 0.0019** | 0.9949 ± 0.0038 | 0.9995 ± 0.0001 | 0.9965 ± 0.0023 | 0.9997 ± 0.0001 | 0.9989 ± 0.0002 |
| farm-flow | random | 0.9991 ± 0.0000 | **0.9990 ± 0.0000** | 1.0000 ± 0.0000 | 0.9980 ± 0.0001 | 0.9546 ± 0.0013 | 0.9987 ± 0.0004 | 0.9999 ± 0.0000 |
| nsl-kdd | official | 0.7772 ± 0.0068 | **0.8025 ± 0.0057** | 0.9638 ± 0.0026 | 0.6875 ± 0.0092 | 0.6589 ± 0.0058 | 0.8268 ± 0.0036 | 0.8407 ± 0.0027 |
| nsl-kdd | random | 0.9930 ± 0.0006 | **0.9932 ± 0.0010** | 0.9932 ± 0.0013 | 0.9932 ± 0.0012 | 0.9869 ± 0.0020 | 0.9944 ± 0.0015 | 0.9913 ± 0.0023 |
| sensornetguard | random | 0.9866 ± 0.0058 | **0.9834 ± 0.0118** | 0.9876 ± 0.0106 | 0.9794 ± 0.0239 | 0.9826 ± 0.0123 | 0.9894 ± 0.0119 | 0.9682 ± 0.0219 |
| unsw-nb15 | official | 0.8836 ± 0.0006 | **0.8886 ± 0.0024** | 0.8150 ± 0.0044 | 0.9769 ± 0.0057 | 0.7406 ± 0.0060 | 0.9556 ± 0.0037 | 0.9435 ± 0.0044 |
| unsw-nb15 | random | 0.9517 ± 0.0008 | **0.9540 ± 0.0009** | 0.9598 ± 0.0017 | 0.9483 ± 0.0012 | 0.8740 ± 0.0026 | 0.9850 ± 0.0039 | 0.9887 ± 0.0041 |

### Confidence metrics (tuned model)

Brier = mean squared error of predicted probability (lower better). ECE = expected calibration error over 10 confidence bins (lower better). Mean conf = average confidence in the predicted class. High-conf = rows where confidence ≥ 0.9 (coverage = fraction of rows; accuracy on those rows). F1 bootstrap CI = 95% interval from 1000 multinomial resamples of the test confusion matrix, averaged over iterations.

| Dataset | Protocol | Brier | ECE | Mean conf | High-conf coverage | High-conf accuracy | F1 bootstrap CI (mean of iters) |
|---|---|---|---|---|---|---|---|
| cic-iov-2024 | official | 0.0000 | 0.0000 | 1.0000 | 1.000 | 1.0000 | 0.9999 - 1.0000 |
| cic-iov-2024 | random | 0.0000 | 0.0000 | 1.0000 | 1.000 | 1.0000 | 0.9999 - 1.0000 |
| cicids2017 | official | 0.0008 | 0.0006 | 0.9997 | 1.000 | 0.9994 | 0.9976 - 0.9980 |
| cicids2017 | random | 0.0010 | 0.0008 | 0.9997 | 0.999 | 0.9992 | 0.9969 - 0.9974 |
| farm-flow | random | 0.0020 | 0.0019 | 0.9999 | 1.000 | 0.9980 | 0.9989 - 0.9991 |
| nsl-kdd | official | 0.1925 | 0.1925 | 0.9998 | 1.000 | 0.8075 | 0.7967 - 0.8080 |
| nsl-kdd | random | 0.0062 | 0.0058 | 0.9991 | 0.997 | 0.9944 | 0.9923 - 0.9941 |
| sensornetguard | random | 0.0016 | 0.0016 | 1.0000 | 1.000 | 0.9984 | 0.9639 - 0.9971 |
| unsw-nb15 | official | 0.0942 | 0.0756 | 0.9408 | 0.814 | 0.9491 | 0.8866 - 0.8906 |
| unsw-nb15 | random | 0.0403 | 0.0132 | 0.9547 | 0.852 | 0.9897 | 0.9524 - 0.9556 |

### Selected configurations per iteration

| Dataset | Protocol | Feature set (count) | Most common params |
|---|---|---|---|
| cic-iov-2024 | official | raw (5) | `{"class_weight": null, "criterion": "gini", "max_depth": null, "min_samples_leaf": 1}` (5/5) |
| cic-iov-2024 | random | raw (5) | `{"class_weight": null, "criterion": "gini", "max_depth": null, "min_samples_leaf": 1}` (3/5) |
| cicids2017 | official | raw+derived (1), raw (4) | `{"class_weight": "balanced", "criterion": "entropy", "max_depth": 20, "min_samples_leaf": 1}` (4/5) |
| cicids2017 | random | raw (5) | `{"class_weight": "balanced", "criterion": "entropy", "max_depth": 20, "min_samples_leaf": 1}` (4/5) |
| farm-flow | random | raw+derived (5) | `{"class_weight": "balanced", "criterion": "entropy", "max_depth": 12, "min_samples_leaf": 1}` (5/5) |
| nsl-kdd | official | raw (1), raw+derived (4) | `{"class_weight": null, "criterion": "entropy", "max_depth": null, "min_samples_leaf": 1}` (2/5) |
| nsl-kdd | random | raw (4), raw+derived (1) | `{"class_weight": "balanced", "criterion": "entropy", "max_depth": 20, "min_samples_leaf": 1}` (2/5) |
| sensornetguard | random | raw (3), raw+derived (2) | `{"class_weight": null, "criterion": "gini", "max_depth": null, "min_samples_leaf": 1}` (3/5) |
| unsw-nb15 | official | raw (2), raw+derived (3) | `{"class_weight": null, "criterion": "entropy", "max_depth": 20, "min_samples_leaf": 20}` (2/5) |
| unsw-nb15 | random | raw+derived (4), raw (1) | `{"class_weight": null, "criterion": "entropy", "max_depth": 20, "min_samples_leaf": 20}` (4/5) |

### Per-iteration tuned F1

| Dataset | Protocol | it0 | it1 | it2 | it3 | it4 |
|---|---|---|---|---|---|---|
| cic-iov-2024 | official | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| cic-iov-2024 | random | 0.9999 | 1.0000 | 1.0000 | 0.9999 | 1.0000 |
| cicids2017 | official | 0.9979 | 0.9979 | 0.9978 | 0.9977 | 0.9978 |
| cicids2017 | random | 0.9981 | 0.9978 | 0.9945 | 0.9976 | 0.9978 |
| farm-flow | random | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| nsl-kdd | official | 0.7988 | 0.8035 | 0.8028 | 0.8095 | 0.7981 |
| nsl-kdd | random | 0.9931 | 0.9937 | 0.9935 | 0.9940 | 0.9918 |
| sensornetguard | random | 0.9898 | 0.9684 | 0.9896 | 0.9897 | 0.9794 |
| unsw-nb15 | official | 0.8879 | 0.8895 | 0.8885 | 0.8861 | 0.8912 |
| unsw-nb15 | random | 0.9532 | 0.9532 | 0.9543 | 0.9546 | 0.9547 |

## Part 1: derived features vs raw (single run, official/80-20 split)

Derived features (rates, ratios, sizes, error/loss rates) computed row-for-row. F1 by tree depth (None = unlimited).

| Dataset | Raw → derived features | Depth | Raw F1 | Derived F1 | Raw top feature | Derived top feature |
|---|---|---|---|---|---|---|
| cic-iov-2024 | 136 → 4 | 3 | 0.8998 | 0.7734 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | 5 | 0.9907 | 0.8914 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | 7 | 0.9992 | 0.9840 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | 10 | 0.9999 | 0.9911 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | None | 0.9999 | 0.9911 | DATA_310 | frame_delta |

## Part 2: shared features and cross-dataset transfer (negative result)

Flow datasets: UNSW-NB15, NSL-KDD, CIC-IDS2017, Farm-Flow (from raw monthly CSVs). SensorNetGuard and IoV excluded (different domains).

- Features available in all four: `log_duration`, `log_src_bytes`, `log_dst_bytes`, `log_total_bytes`, `src_byte_share`, `log_byte_rate`.
- Consistent-direction shortlist (≥3 datasets, |AUC−0.5|>0.1, same sign): `log_src_bytes`.
- A depth-6 tree identifies which dataset a *benign* row came from with 92.5% accuracy on the all-4 features (78.5% on the shortlist; chance 25%): the domains are not aligned.

Signed AUC−0.5 per feature (positive: attacks have higher values):

| Feature | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow | Benign KS shift |
|---|---|---|---|---|---|
| log_duration | -0.14 | -0.04 | +0.04 | -0.44 | 0.64 |
| log_src_bytes | -0.21 | -0.40 | -0.14 | -0.46 | 0.59 |
| log_dst_bytes | -0.29 | -0.40 | +0.04 | -0.46 | 0.40 |
| log_total_bytes | -0.24 | -0.43 | +0.01 | -0.47 | 0.47 |
| src_byte_share | +0.31 | +0.41 | -0.30 | +0.39 | 0.47 |
| log_byte_rate | +0.17 | -0.05 | -0.08 | -0.28 | 0.63 |
| log_mean_pkt_size | -0.15 | n/a | +0.02 | -0.31 | 0.54 |
| src_pkt_share | +0.33 | n/a | +0.00 | +0.31 | 0.29 |
| log_pkt_rate | +0.18 | n/a | -0.06 | -0.29 | 0.41 |
| error_rate | n/a | +0.38 | -0.00 | +0.05 | 0.05 |

Leave-one-dataset-out F1 (train on the other three):

| Feature set | Transform | Depth | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow |
|---|---|---|---|---|---|---|
| all4 | asis | 5 | 0.10 | 0.32 | 0.36 | 0.12 |
| all4 | asis | 10 | 0.08 | 0.59 | 0.24 | 0.34 |
| all4 | quantile | 5 | 0.02 | 0.41 | 0.29 | 0.90 |
| all4 | quantile | 10 | 0.19 | 0.71 | 0.27 | 0.72 |
| shortlist | asis | 5 | 0.07 | 0.17 | 0.37 | 0.37 |
| shortlist | asis | 10 | 0.05 | 0.19 | 0.32 | 0.38 |
| shortlist | quantile | 5 | 0.58 | 0.17 | 0.42 | 0.80 |
| shortlist | quantile | 10 | 0.05 | 0.08 | 0.36 | 0.80 |

Conclusion: no stable shared decision tree; the dominant feature differs per dataset (SensorNetGuard/NSL-KDD `error_rate`, UNSW-NB15 `pkt_rate`, CIC-IDS2017 `fwd_bwd_byte_ratio`), and a pooled tree mostly learns dataset identity.

## Caveats

- Farm-Flow raw monthly CSVs are 97.9% attack vs ~50/50 in the silver split; its numbers are not comparable to earlier silver baselines. The `traffic` column (attack-type label) was excluded from features.
- Official-split intervals only reflect tuning/seed variance; trees are near-deterministic given a fixed split.
- CIC-IDS2017 and UNSW-NB15 tuning used a ≤100k-row subsample with 3-fold CV on train rows only (no test leakage).
- Byte columns are not the same quantity across datasets (IP bytes vs payload/flow bytes), which limits cross-dataset comparison.
- Protocol/flag/TTL columns were dropped from the silver files; rebuilding from raw could expose more shared features.
