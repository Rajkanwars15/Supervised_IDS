# Derived features, evaluation protocol and cross-dataset transfer in network IDS (decision-tree study)

Every number below is generated from `experiments/derived/results/*.json` by `write_report.py`. Scripts: `derive_features.py`, `derived_dt_experiment.py`, `robust_per_dataset.py`, `shared_analysis.py`, `leakage_audit.py`, `raw_loaders.py`, `split_ladder.py`, `compression_table.py`, `baselines.py`.

## Summary

- **RQ1 (compression).** Dataset-specific derived rate/ratio features cut inputs by 56-97% and keep 96-100% of the raw-feature F1 at unlimited depth (random splits, paired, 5 resplits): SensorNetGuard 99.4%, Farm-Flow 99.9%, UNSW-NB15 98.9%, NSL-KDD 97.9%, CIC-IoV 99.1%, CIC-IDS2017 96.2%. The compression is of the *input*, not of the *model*: derived trees are often as large or larger (e.g. CIC-IDS2017 4,491 vs 1,437 nodes).
- **RQ2 (generalisation).** Compact representations do not transfer across datasets: the dominant feature differs per dataset, signs of the same feature flip, a tree identifies the source dataset of a benign row 92.5% of the time from the six shared features, and leave-one-dataset-out F1 is 0.02-0.90 (mostly < 0.4).
- **RQ3 (protocol).** The evaluation protocol changes conclusions more than the model does. Examples: CIC-IDS2017 F1 0.997 (random) → 0.849 (time) → 0.061 (group by source IP, degenerate test) and 0.43 macro-recall on held-out attack families; CIC-IoV 1.000 → 0.927 (time) and 0.33 macro-recall on held-out families; NSL-KDD 0.993 (random) vs 0.78 (benchmark) with 0.38 macro-recall on held-out families.
- **Leakage.** UNSW-NB15 is ~40% duplicate rows: its random-split F1 of 0.951 is 0.869 on test rows unseen in training, so much of the random-vs-benchmark gap is duplicate leakage. CIC-IoV-2024's bit features are 99.7% duplicates (99.8% of test rows have an exact copy in train), i.e. near-perfect scores reflect lookup. Duplicates do **not** explain NSL-KDD's gap (a family/distribution shift) or CIC-IDS2017.
- **Classifier-independence.** RF, XGBoost and LightGBM show the same pattern as the decision tree on raw vs derived features (Section 4); ensembles add about 1 F1 point or less over the decision tree.
- Kyoto was removed: its silver train/test sets are 100% attack (label mapping appears inverted) and columns 14-16 look like IDS-detection flags (leakage).

## 0. What "official" split means here (correction)

| Dataset | Split used as "official" | Genuinely a published benchmark split? |
|---|---|---|
| UNSW-NB15 | UNSW_NB15 training-set / testing-set (175k/82k) | yes |
| NSL-KDD | KDDTrain+ / KDDTest+ | yes |
| Farm-Flow | Farm-Flow Train/Test CSVs (balanced; raw monthly files are 97.9% attack) | yes, but class balance differs from raw |
| CIC-IDS2017 | our own `train_test_split(0.2, seed 42, stratify)` in preprocessing | **no** |
| CIC-IoV-2024 | our own `train_test_split(0.2, seed 42, stratify)` in preprocessing | **no** |
| SensorNetGuard | n/a (single CSV; our 80/20) | no |

Only UNSW-NB15 and NSL-KDD give a benchmark-vs-random comparison. Silver files also dropped provenance (IDS2017 IPs/timestamps/Flow ID; IoV CAN-ID bits and file identity; UNSW IPs/time), so group/time/family protocols were re-run from the raw files (`raw_loaders.py`), whose feature sets differ slightly from silver (notes in Section 3).

## 1. Leakage and dataset-artefact audit

Default-parameter decision tree on raw features. *Seen* = test row whose exact (rounded) feature vector appears in train. *Unseen F1* = F1 on test rows not seen in train. *Drop-top-3* removes the three most important features and retrains. *Shuffled* = F1 with permuted training labels (≈ class prior if the pipeline is sound).

| Dataset | Split | Dup. train % | Test seen in train % | Label-conflict rows % | F1 all | F1 unseen | F1 drop-top-1 | F1 drop-top-3 | F1 shuffled | Top feature (importance) |
|---|---|---|---|---|---|---|---|---|---|---|
| sensornetguard | random | 0.0 | 0.0 | 0.00 | 0.9898 | 0.9898 | 0.9789 | 0.9206 | 0.0476 | Error_Rate (0.83) |
| nsl-kdd | benchmark | 5.1 | 6.8 | 0.02 | 0.7754 | 0.7452 | 0.7556 | 0.7297 | 0.5054 | 4 (0.75) |
| nsl-kdd | random | 4.7 | 7.9 | 0.15 | 0.9925 | 0.9925 | 0.9900 | 0.9844 | 0.4666 | 4 (0.70) |
| unsw-nb15 | benchmark | 42.6 | 10.5 | 0.57 | 0.8846 | 0.8695 | 0.8860 | 0.8873 | 0.6057 | sttl (0.68) |
| unsw-nb15 | random | 39.2 | 45.3 | 0.68 | 0.9512 | 0.8694 | 0.9515 | 0.9520 | 0.7088 | sttl (0.54) |
| farm-flow | random | 37.3 | 52.9 | 0.00 | 0.9991 | 0.9980 | 0.9990 | 0.9991 | 0.9805 | bwd_data_pkts_tot (0.55) |
| cic-iov-2024 | own 80/20 (seed 42) | 99.7 | 99.8 | 1.65 | 0.9999 | n/a | 0.9999 | 0.9999 | 0.0000 | DATA_310 (0.44) |
| cic-iov-2024 | random | 99.7 | 99.8 | 1.66 | 0.9999 | n/a | 0.9999 | 0.9999 | 0.0000 | DATA_310 (0.44) |
| cicids2017 | own 80/20 (seed 42) | 10.3 | 13.7 | 0.24 | 0.9969 | 0.9971 | 0.9968 | 0.9969 | 0.1693 | Bwd Packet Length Std (0.38) |
| cicids2017 | random | 10.9 | 14.4 | 0.23 | 0.9970 | 0.9973 | 0.9970 | 0.9970 | 0.1700 | Bwd Packet Length Std (0.38) |

Reading: UNSW-NB15 random vs benchmark (F1 0.951 vs 0.885) shrinks to 0.869 vs 0.870 once seen rows are excluded - the gap is duplicate contamination. CIC-IoV's "unseen" subset is too small/single-class for an F1. Only SensorNetGuard loses more than ~1 F1 point when its top three features are dropped (0.990 → 0.921); elsewhere the signal is redundant across features, so no single leaked column explains the scores. Not audited: IP/port identifier-only predictors beyond the group-split results in Section 2, and flow/session overlap in silver files (provenance dropped).

## 2. RQ3 - evaluation-protocol ladder

Default decision tree, raw features from `raw_loaders.py`. random = 5 stratified resplits; group = 5 GroupShuffleSplits by source IP; time = earliest 80% → latest 20% (single deterministic run; IoV within each source file; Farm-Flow row order as proxy); family = leave-one-attack-family-out (benign 80/20 fixed), reporting recall on the held-out family. N/A = the dataset has no such information.

| Dataset (n, features) | Random F1 | Group F1 | Time F1 | Family macro-recall | Family weighted-recall | Mean benign FPR | Loader note |
|---|---|---|---|---|---|---|---|
| sensornetguard (10,000; 17) | 0.9938 ± 0.0085 | N/A (no group key) | 0.9758 | N/A | N/A | N/A | synthetic; Node_ID is unique per row (no group structure); Timestamp as time |
| nsl-kdd (148,517; 38) | 0.9930 ± 0.0005 | N/A (no group key) | N/A (no timestamp) | 0.378 | 0.797 | 0.0061 | no time/group info; categoricals 1-3 dropped as in silver; train+test pooled |
| unsw-nb15 (2,540,047; 38) | 0.9718 ± 0.0009 | 0.9404 ± 0.0535 | 0.9644 | 0.809 | 0.614 | 0.0041 | raw 4-file UNSW (2.5M rows, ~42 numeric features) - NOT the 257k train/test-set partition used in silver |
| farm-flow (1,309,887; 85) | 0.9991 ± 0.0000 | 0.9890 ± 0.0100 | N/A (single-class test) | 0.769 | 0.905 | 0.0408 | no timestamp column: month + row order used as time proxy; ports excluded from features |
| cic-iov-2024 (1,408,219; 136) | 1.0000 ± 0.0000 | N/A (no group key) | 0.9274 | 0.331 | 0.459 | 0.0000 | DATA_* only (as silver); CAN ID bits ID0-16 excluded; time = row order within each source file |
| cicids2017 (2,830,743; 79) | 0.9973 ± 0.0001 | 0.0611 ± 0.0575 | 0.8486 | 0.425 | 0.297 | 0.0008 | Destination Port kept as feature (as in silver); duplicates NOT removed; row count differs from silver if silver dropped NaN/inf rows |

Per-family recall when the family is held out (worst first):

- **nsl-kdd**: snmpgetattack 0.00, snmpguess 0.00, mailbomb 0.00, warezclient 0.01, back 0.01, processtable 0.04, smurf 0.17, warezmaster 0.25
- **unsw-nb15**: Fuzzers 0.22, Generic 0.55, Reconnaissance 0.82, Analysis 0.83, Shellcode 0.85, Exploits 0.90, Worms 0.97, DoS 0.98
- **farm-flow**: Arp_Spoofing 0.06, BotNet_DDOS 0.53, HTTP_Flood 0.60, UDP_Flood 0.96, Port_Scanning 1.00, TCP_Flood 1.00, ICMP_Flood 1.00, MQTT_Flood 1.00
- **cic-iov-2024**: spoofing-GAS 0.00, spoofing-STEERING_WHEEL 0.00, spoofing-RPM 0.45, DoS 0.60, spoofing-SPEED 0.60
- **cicids2017**: Bot 0.00, FTP-Patator 0.00, PortScan 0.01, SSH-Patator 0.06, DoS Hulk 0.29, DoS Slowhttptest 0.35, DDoS 0.63, DoS GoldenEye 0.66

Caveats: CIC-IDS2017's group split is degenerate (a handful of attacker IPs generate nearly all attacks, so the group-held-out test set is ~0.2% attack; F1 0.06 reflects attacker-identity shift, not a clean error rate). Farm-Flow's time test slice is single-class (attack-only). Time-split numbers have no resampling interval.

## 3. RQ1 - compression: raw vs derived features

Random 80/20, 5 paired resplits (same split/seed for raw and derived), default decision tree with depth cap, pools capped at 500k rows. Retention = derived / raw. Model size = joblib bytes; latency = median ms per 1,000 rows over 9 timed predicts (resolution ~0.1 ms, single process).

| Dataset | Depth cap | Raw → derived features | Input reduction | Raw F1 | Derived F1 | F1 retention | MCC retention | AP retention | Nodes raw → derived | Model KB raw → derived | Latency raw → derived (ms/1k rows) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sensornetguard | 5 | 17 → 4 | 76% | 0.9856 | 0.9803 | 99.5% | 99.4% | 99.9% | 16 → 21 | 3 → 3 | 0.18 → 0.16 |
| sensornetguard | 10 | 17 → 4 | 76% | 0.9866 | 0.9804 | 99.4% | 99.3% | 98.8% | 19 → 35 | 3 → 4 | 0.18 → 0.15 |
| sensornetguard | unlimited | 17 → 4 | 76% | 0.9866 | 0.9804 | 99.4% | 99.3% | 98.8% | 19 → 35 | 3 → 4 | 0.18 → 0.15 |
| nsl-kdd | 5 | 38 → 11 | 71% | 0.9709 | 0.9391 | 96.7% | 93.4% | 95.6% | 61 → 63 | 7 → 7 | 0.06 → 0.05 |
| nsl-kdd | 10 | 38 → 11 | 71% | 0.9907 | 0.9691 | 97.8% | 95.7% | 96.9% | 459 → 457 | 38 → 37 | 0.07 → 0.05 |
| nsl-kdd | unlimited | 38 → 11 | 71% | 0.9930 | 0.9722 | 97.9% | 95.9% | 97.4% | 1709 → 1022 | 135 → 82 | 0.08 → 0.06 |
| unsw-nb15 | 5 | 39 → 17 | 56% | 0.9298 | 0.9102 | 97.9% | 90.2% | 99.0% | 39 → 54 | 5 → 6 | 0.07 → 0.05 |
| unsw-nb15 | 10 | 39 → 17 | 56% | 0.9473 | 0.9383 | 99.1% | 96.4% | 99.7% | 559 → 683 | 46 → 55 | 0.08 → 0.07 |
| unsw-nb15 | unlimited | 39 → 17 | 56% | 0.9517 | 0.9410 | 98.9% | 96.5% | 99.2% | 16747 → 21551 | 1311 → 1686 | 0.11 → 0.09 |
| farm-flow | 5 | 87 → 10 | 89% | 0.9982 | 0.9988 | 100.1% | 103.2% | 100.1% | 40 → 49 | 7 → 6 | 0.11 → 0.03 |
| farm-flow | 10 | 87 → 10 | 89% | 0.9991 | 0.9989 | 100.0% | 99.1% | 100.0% | 215 → 231 | 20 → 20 | 0.12 → 0.04 |
| farm-flow | unlimited | 87 → 10 | 89% | 0.9990 | 0.9985 | 99.9% | 97.4% | 99.9% | 1006 → 2363 | 82 → 186 | 0.12 → 0.04 |
| cic-iov-2024 | 5 | 136 → 4 | 97% | 0.9913 | 0.8896 | 89.7% | 88.2% | 95.6% | 35 → 33 | 6 → 4 | 0.14 → 0.04 |
| cic-iov-2024 | 10 | 136 → 4 | 97% | 0.9996 | 0.9913 | 99.2% | 99.1% | 100.0% | 61 → 65 | 8 → 7 | 0.14 → 0.04 |
| cic-iov-2024 | unlimited | 136 → 4 | 97% | 0.9999 | 0.9913 | 99.1% | 99.0% | 100.0% | 73 → 65 | 9 → 7 | 0.14 → 0.04 |
| cicids2017 | 5 | 78 → 18 | 77% | 0.9571 | 0.8967 | 93.7% | 92.2% | 94.7% | 44 → 55 | 7 → 6 | 0.09 → 0.05 |
| cicids2017 | 10 | 78 → 18 | 77% | 0.9917 | 0.9468 | 95.5% | 94.3% | 99.1% | 165 → 479 | 16 → 39 | 0.09 → 0.06 |
| cicids2017 | unlimited | 78 → 18 | 77% | 0.9963 | 0.9586 | 96.2% | 95.3% | 98.3% | 1437 → 4491 | 116 → 353 | 0.12 → 0.08 |

Notes: derived sets are dataset-specific (SensorNetGuard 4, NSL-KDD 11, UNSW-NB15 17, Farm-Flow 10, CIC-IoV 4 bit-statistics, CIC-IDS2017 18); for the flow datasets they contain two overlapping derivations (`a_*`, `b_*`). The input reduction does not shrink trees: unlimited-depth derived trees are larger on CIC-IDS2017 (≈3×), UNSW-NB15 and Farm-Flow, and smaller only on NSL-KDD and CIC-IoV. Inference latency is negligible for all (below measurement resolution).

## 4. Stronger baselines on the same protocol

Fixed hyper-parameters (RF 100 trees; XGBoost 200 trees depth 6; LightGBM 200 trees, 63 leaves; DT default), train capped at 300k rows. Mean F1 over 5 resplits (random) or 5 model seeds on the fixed benchmark split. Cells: raw / derived.

| Dataset | Protocol | DT | RF | XGBoost | LightGBM |
|---|---|---|---|---|---|
| sensornetguard | random | 0.9866 / 0.9804 | 0.9979 / 0.9834 | 0.9969 / 0.9845 | 0.9875 / 0.9876 |
| nsl-kdd | benchmark | 0.7772 / 0.7976 | 0.7548 / 0.7500 | 0.7814 / 0.8041 | 0.7672 / 0.7424 |
| nsl-kdd | random | 0.9930 / 0.9722 | 0.9947 / 0.9728 | 0.9947 / 0.9713 | 0.9957 / 0.9720 |
| unsw-nb15 | benchmark | 0.8836 / 0.8752 | 0.8925 / 0.8904 | 0.8951 / 0.8868 | 0.8938 / 0.8913 |
| unsw-nb15 | random | 0.9517 / 0.9410 | 0.9620 / 0.9469 | 0.9577 / 0.9459 | 0.9611 / 0.9474 |
| farm-flow | random | 0.9989 / 0.9985 | 0.9992 / 0.9985 | 0.9993 / 0.9989 | 0.9993 / 0.9988 |
| cic-iov-2024 | random | 0.9999 / 0.9913 | 0.9999 / 0.9913 | 0.9999 / 0.9913 | 0.9999 / 0.9913 |
| cicids2017 | random | 0.9962 / 0.9582 | 0.9963 / 0.9594 | 0.9978 / 0.9606 | 0.9976 / 0.9617 |

Benchmark-split rows have no sampling variance (seeds only). On NSL-KDD's benchmark split derived features *help* DT and XGBoost but hurt LightGBM and RF, so model class matters there; on the other datasets the raw→derived pattern is the same across all four classifiers.

## 5. RQ2 - shared features and cross-dataset transfer (flow datasets)

UNSW-NB15, NSL-KDD, CIC-IDS2017 (silver splits) and Farm-Flow (raw monthly files, stratified 80/20). SensorNetGuard and IoV excluded (different domains).

- Features available in all four: `log_duration`, `log_src_bytes`, `log_dst_bytes`, `log_total_bytes`, `src_byte_share`, `log_byte_rate`; consistent-direction shortlist (≥3 datasets, |AUC−0.5|>0.1, same sign): `log_src_bytes`.
- A depth-6 tree identifies which dataset a *benign* row came from with 92.5% accuracy on the all-4 features (78.5% on the shortlist; chance 25%).

Signed AUC−0.5 (positive: attacks have higher values) and mean pairwise KS shift between benign distributions:

| Feature | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow | Benign KS |
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

Single run, single seed. Byte columns are not the same quantity across datasets (IP bytes vs payload/flow bytes).

## 6. Tuned per-dataset decision trees, 5 iterations, with confidence metrics

Tuning (feature set × depth × min-leaf × criterion × class-weight) by 3-fold CV on a ≤100k train subsample only. **Benchmark** rows: fixed split, iterations vary seed/tuning only, so the ± reflects model-selection uncertainty, **not** test-set sampling (use the bootstrap F1 interval below for that - and it is not an independent replication). **Random** rows: 5 different stratified resplits (sampling uncertainty). 'own 80/20' = our fixed seed-42 split, not a published benchmark.

| Dataset | Protocol | Baseline F1 | Tuned F1 | Tuned precision | Tuned recall | Tuned MCC | Tuned AUC | Tuned AP |
|---|---|---|---|---|---|---|---|---|
| cic-iov-2024 | own 80/20 (seed 42) | 0.9999 ± 0.0000 | **0.9999 ± 0.0000** | 1.0000 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| cic-iov-2024 | random | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| cicids2017 | own 80/20 (seed 42) | 0.9972 ± 0.0000 | **0.9978 ± 0.0001** | 0.9964 ± 0.0004 | 0.9993 ± 0.0003 | 0.9973 ± 0.0001 | 0.9996 ± 0.0001 | 0.9985 ± 0.0004 |
| cicids2017 | random | 0.9972 ± 0.0002 | **0.9972 ± 0.0019** | 0.9949 ± 0.0038 | 0.9995 ± 0.0001 | 0.9965 ± 0.0023 | 0.9997 ± 0.0001 | 0.9989 ± 0.0002 |
| farm-flow | random | 0.9991 ± 0.0000 | **0.9990 ± 0.0000** | 1.0000 ± 0.0000 | 0.9980 ± 0.0001 | 0.9546 ± 0.0013 | 0.9987 ± 0.0004 | 0.9999 ± 0.0000 |
| nsl-kdd | benchmark | 0.7772 ± 0.0068 | **0.8025 ± 0.0057** | 0.9638 ± 0.0026 | 0.6875 ± 0.0092 | 0.6589 ± 0.0058 | 0.8268 ± 0.0036 | 0.8407 ± 0.0027 |
| nsl-kdd | random | 0.9930 ± 0.0006 | **0.9932 ± 0.0010** | 0.9932 ± 0.0013 | 0.9932 ± 0.0012 | 0.9869 ± 0.0020 | 0.9944 ± 0.0015 | 0.9913 ± 0.0023 |
| sensornetguard | random | 0.9866 ± 0.0058 | **0.9834 ± 0.0118** | 0.9876 ± 0.0106 | 0.9794 ± 0.0239 | 0.9826 ± 0.0123 | 0.9894 ± 0.0119 | 0.9682 ± 0.0219 |
| unsw-nb15 | benchmark | 0.8836 ± 0.0006 | **0.8886 ± 0.0024** | 0.8150 ± 0.0044 | 0.9769 ± 0.0057 | 0.7406 ± 0.0060 | 0.9556 ± 0.0037 | 0.9435 ± 0.0044 |
| unsw-nb15 | random | 0.9517 ± 0.0008 | **0.9540 ± 0.0009** | 0.9598 ± 0.0017 | 0.9483 ± 0.0012 | 0.8740 ± 0.0026 | 0.9850 ± 0.0039 | 0.9887 ± 0.0041 |

Confidence metrics (tuned model): Brier, ECE (10 bins), mean confidence, high-confidence (≥0.9) coverage/accuracy, and the 95% bootstrap interval of the test-set F1 (averaged over iterations).

| Dataset | Protocol | Brier | ECE | Mean conf | High-conf coverage | High-conf accuracy | F1 bootstrap CI |
|---|---|---|---|---|---|---|---|
| cic-iov-2024 | own 80/20 (seed 42) | 0.0000 | 0.0000 | 1.0000 | 1.000 | 1.0000 | 0.9999 - 1.0000 |
| cic-iov-2024 | random | 0.0000 | 0.0000 | 1.0000 | 1.000 | 1.0000 | 0.9999 - 1.0000 |
| cicids2017 | own 80/20 (seed 42) | 0.0008 | 0.0006 | 0.9997 | 1.000 | 0.9994 | 0.9976 - 0.9980 |
| cicids2017 | random | 0.0010 | 0.0008 | 0.9997 | 0.999 | 0.9992 | 0.9969 - 0.9974 |
| farm-flow | random | 0.0020 | 0.0019 | 0.9999 | 1.000 | 0.9980 | 0.9989 - 0.9991 |
| nsl-kdd | benchmark | 0.1925 | 0.1925 | 0.9998 | 1.000 | 0.8075 | 0.7967 - 0.8080 |
| nsl-kdd | random | 0.0062 | 0.0058 | 0.9991 | 0.997 | 0.9944 | 0.9923 - 0.9941 |
| sensornetguard | random | 0.0016 | 0.0016 | 1.0000 | 1.000 | 0.9984 | 0.9639 - 0.9971 |
| unsw-nb15 | benchmark | 0.0942 | 0.0756 | 0.9408 | 0.814 | 0.9491 | 0.8866 - 0.8906 |
| unsw-nb15 | random | 0.0403 | 0.0132 | 0.9547 | 0.852 | 0.9897 | 0.9524 - 0.9556 |

Selected configurations: cic-iov-2024|official: raw×5; cic-iov-2024|random: raw×5; cicids2017|official: raw+derived×1,raw×4; cicids2017|random: raw×5; farm-flow|random: raw+derived×5; nsl-kdd|official: raw×1,raw+derived×4; nsl-kdd|random: raw×4,raw+derived×1; sensornetguard|random: raw×3,raw+derived×2; unsw-nb15|official: raw×2,raw+derived×3; unsw-nb15|random: raw+derived×4,raw×1.

## 7. Derived vs raw, single run (official/80-20 splits)

| Dataset | Raw → derived features | Depth | Raw F1 | Derived F1 | Raw top feature | Derived top feature |
|---|---|---|---|---|---|---|
| cic-iov-2024 | 136 → 4 | 3 | 0.8998 | 0.7734 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | 5 | 0.9907 | 0.8914 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | 7 | 0.9992 | 0.9840 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | 10 | 0.9999 | 0.9911 | DATA_310 | frame_delta |
| cic-iov-2024 | 136 → 4 | None | 0.9999 | 0.9911 | DATA_310 | frame_delta |

## Limitations

- Farm-Flow raw monthly CSVs are 97.9% attack vs ~50/50 in the silver split; absolute numbers differ between the two. The `traffic` column (attack-type label) was excluded from features.
- Derived feature sets are hand-built and dataset-specific; the flow-dataset sets use two overlapping derivations, so 'derived' is not minimal.
- Raw UNSW-NB15 (2.5M rows, four files) differs from the 257k train/test partition used for the benchmark rows; ladder numbers are not directly comparable to Section 6's UNSW benchmark.
- Silver preprocessing dropped CAN-ID bits (IoV), IPs/ports/timestamps (IDS2017, UNSW), so identifier-leakage tests are limited to group/time/family splits.
- Time splits are single deterministic runs (no interval); Farm-Flow and IoV time proxies are row order, not timestamps.
- CIC-IDS2017 group split is degenerate (see Section 2). Hyper-parameters for RF/XGBoost/LightGBM were fixed, not tuned. SensorNetGuard appears synthetic.
- Latency was measured on a shared laptop CPU, single process; differences below ~0.1 ms/1k rows are not resolvable.
