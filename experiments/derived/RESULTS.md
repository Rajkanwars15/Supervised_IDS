# Derived features, evaluation protocol and cross-dataset transfer in network IDS (decision-tree study)

Every number below is generated from `experiments/derived/results/*.json` by `write_report.py`. Scripts: `derive_features.py`, `derived_dt_experiment.py`, `robust_per_dataset.py`, `shared_analysis.py`, `leakage_audit.py`, `raw_loaders.py`, `split_ladder.py`, `compression_table.py`, `baselines.py`.

## Summary

**Scope.** Six datasets (SensorNetGuard, NSL-KDD, UNSW-NB15, Farm-Flow, CIC-IoV-2024, CIC-IDS2017; Kyoto removed), decision trees with RF/XGBoost/LightGBM as baselines, raw vs hand-derived rate/ratio features, evaluated under random, duplicate-aware, temporal, group, attack-family and cross-dataset protocols. Fixed metric set: F1, MCC, PR-AUC, benign FPR, family macro-recall.

- **RQ1 - input compression holds under IID-like protocols but weakens under shift, and it is not model compression.** Derived features cut inputs by 56-97% and retain 95-100% of raw F1 on random splits and, except for CIC-IoV, on vector-disjoint splits (Sections 3, 13). Under harder protocols retention drops: CIC-IoV vector-disjoint 75% (tree) / 90% (LightGBM); CIC-IDS2017 time split 94% (tree) / 78% (LightGBM) in Section 13 but a paired ΔF1 of −0.20 [−0.20, −0.20] in Section 16, because the IDS2017 time-split F1 depends strongly on the training subsample (raw tree F1 0.849 with the full train set, 0.754 with a 600k cap, 0.867 with a 300k cap) - treat IDS2017 time-split retention as unresolved - and 5-minute time-block groups 88-91%. Fitted trees are not smaller: at unlimited depth derived/raw node ratios are 0.58 (NSL-KDD), 0.89 (IoV), 1.30 (UNSW), 1.57 (SensorNetGuard), 2.31 (Farm-Flow), 3.2 (CIC-IDS2017); inference latency is 0.02-0.08 µs/row either way (Section 14).
- **RQ2 - no cross-dataset generalisation.** On harmonised features, within-dataset MCC is 0.59-0.97 while off-diagonal transfer has median MCC 0.00 (Section 15). Against a 20-permutation null, only 1 of 20 off-diagonal cells beats chance (SensorNetGuard→NSL-KDD, two features whose semantics differ) and one is significantly worse than chance (UNSW→Farm-Flow). A tree identifies the source dataset of a benign row 92.5% of the time (Section 5).
- **RQ3 - evaluation protocol dominates.** CIC-IDS2017 F1 0.997 (random) → 0.849 (time) → 0.936/0.821 (5/15-minute time-block groups; source-IP groups are degenerate) with family macro-recall 0.43; CIC-IoV 1.000 (random) → 0.927 (time) → 0.63 (tree) / 0.90 (LightGBM) on a 313k-row vector-disjoint test set, macro-recall 0.33; NSL-KDD 0.993 (random) vs 0.78 (benchmark), macro-recall 0.38 (Sections 9-12, 17).
- **Leakage and chance checks.** Farm-Flow's shuffled-label F1 (~0.98) is the 97.9% attack prior: over 30 permutations MCC is −0.0001 ± 0.005 and AUC 0.500, before or after feature derivation; removing the ports and the five most important features leaves real-label MCC at 0.950 → 0.949 and shuffled-label MCC at 0 (Section 8). A label-shuffled run with the balanced sampler drops Farm-Flow's F1 from 0.987 to 0.684 while MCC stays ≈ 0, confirming a test-prior effect, not leakage (Section 18). Exact-duplicate contamination is large for CIC-IoV (99.7% duplicate rows) and moderate for UNSW-NB15 (21% in the raw files; random-split F1 0.972 vs 0.860 on unseen rows).
- **NSL-KDD.** The 0.993 → 0.776 gap is mostly family-mix and novel-family shift, not duplication: 29% of benchmark-test attack rows come from 17 families absent in training; dropping them gives F1 0.854, matching the random split's family mix gives 0.960; adversarial-validation AUC is 0.90 (0.50 for a random split) (Section 12).
- **Class-balancing in training (secondary experiment).** `ImbalancedDatasetSampler` and class-weighted loss rarely change IID conclusions (ΔF1 on the IID split between −0.022 and 0.000); on SensorNetGuard (4.9% attack) and UNSW-NB15 the sampler raises benign false alarms (FPR +0.0021 and +0.0024 for the tree) and lowers F1. Compression retention is stable across strategies on most datasets but not under shift (CIC-IDS2017 time split 77% / 80% / 95% for standard / sampler / weighted) (Section 18).
- **Classifier-independence.** RF, XGBoost and LightGBM follow the same raw-vs-derived pattern and add about 1 F1 point or less over the decision tree on IID splits (Section 4), but model class matters a lot on novel vectors (IoV: 0.63 vs 0.90).
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
## 8. Chance-level and label-permutation checks (items 1-2)

Stratified label permutation (prior preserved), full pipeline refit, default decision tree, pool capped at 200k rows. Chance references for the attack class: F1(all-attack) = 2p/(1+p), MCC = 0, AP = p, AUC = 0.5. **Pass** = |mean MCC| ≤ 0.01 and |mean AUC − 0.5| ≤ 0.01.

| Dataset | Features | Attack prior | Real F1 | Real MCC | Shuffled F1 | Shuffled MCC (mean ± SD) | Shuffled AUC | Shuffled AP | Chance F1 | #perms | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| farm-flow raw (87 incl. ports), permute after derivation | | 0.979 | 0.9989 | 0.9498 | 0.9794 | -0.0001 ± 0.0049 | 0.5001 | 0.9789 | 0.9894 | 30 | yes |
| farm-flow raw, permute BEFORE derivation + refit | | 0.979 | n/a | n/a | 0.9795 | +0.0014 ± 0.0052 | 0.5008 | 0.9790 | 0.9894 | 30 | yes |
| farm-flow derived, permute after | | 0.979 | 0.9986 | 0.9327 | 0.9873 | +0.0025 ± 0.0065 | 0.4957 | 0.9787 | 0.9894 | 30 | yes |
| sensornetguard raw | | 0.049 | 0.9898 | 0.9893 | 0.0519 | -0.0023 ± 0.0198 | 0.4985 | 0.0497 | 0.0929 | 20 | yes |
| sensornetguard derived | | 0.049 | 0.9846 | 0.9838 | 0.0457 | -0.0071 ± 0.0231 | 0.4966 | 0.0497 | 0.0929 | 20 | yes |
| nsl-kdd raw | | 0.481 | 0.9928 | 0.9861 | 0.4879 | +0.0013 ± 0.0057 | 0.5008 | 0.4818 | 0.6497 | 20 | yes |
| nsl-kdd derived | | 0.481 | 0.9718 | 0.9452 | 0.3531 | +0.0002 ± 0.0054 | 0.5006 | 0.4819 | 0.6497 | 20 | yes |
| unsw-nb15 raw | | 0.639 | 0.9488 | 0.8581 | 0.6824 | +0.0007 ± 0.0058 | 0.5002 | 0.6390 | 0.7798 | 20 | yes |
| unsw-nb15 derived | | 0.639 | 0.9390 | 0.8315 | 0.7095 | -0.0003 ± 0.0058 | 0.5002 | 0.6392 | 0.7798 | 20 | yes |
| cic-iov-2024 raw | | 0.131 | 0.9999 | 0.9999 | 0.0009 | +0.0002 ± 0.0047 | 0.5004 | 0.1309 | 0.2317 | 20 | yes |
| cic-iov-2024 derived | | 0.131 | 0.9906 | 0.9893 | 0.0008 | +0.0000 ± 0.0044 | 0.5002 | 0.1311 | 0.2317 | 20 | yes |
| cicids2017 raw | | 0.197 | 0.9951 | 0.9939 | 0.2045 | +0.0011 ± 0.0059 | 0.5002 | 0.1971 | 0.3291 | 20 | yes |
| cicids2017 derived | | 0.197 | 0.9589 | 0.9488 | 0.1785 | +0.0005 ± 0.0042 | 0.4998 | 0.1969 | 0.3291 | 20 | yes |

F1 alone is not a chance check when the prior is extreme: Farm-Flow's raw data is 97.9% attack, so a no-skill classifier gets F1 ≈ 0.98-0.99 while MCC and AUC sit at 0 and 0.5. The earlier audit column "F1 shuffled" (0.98 for Farm-Flow) is therefore the class prior, not leakage.

### Farm-Flow identifier / top-feature ablation

Importance-ranked removal on the identifier-free model. Real-label metrics on the standard test fold and on a 50/50 undersampled test fold (prior removed); shuffled-label MCC over 20 permutations. Note: the earlier Farm-Flow experiments (audit, robust, compression, baselines) used **87 numeric features including `id.orig_p` and `id.resp_p`** (L0); L1 removes them.

| Feature set | #features | Real F1 | Real MCC | Real AP | Balanced-test MCC | Balanced-test BA | Shuffled MCC | Shuffled F1 |
|---|---|---|---|---|---|---|---|---|
| L0 all numeric (incl. ports, as used in earlier experiments) | 87 | 0.9989 | 0.9498 | 0.9991 | 0.9579 | 0.9785 | +0.0002 | 0.9793 |
| L1 identifiers removed (id.* ports/IPs) | 85 | 0.9991 | 0.9568 | 0.9993 | 0.9706 | 0.9851 | +0.0002 | 0.9867 |
| L1 + drop top-1 (bwd_data_pkts_tot) | 84 | 0.9989 | 0.9488 | 0.9991 | 0.9590 | 0.9791 | -0.0002 | 0.9866 |
| L1 + drop top-3 (bwd_data_pkts_tot,orig_pkts,orig_ip_bytes) | 82 | 0.9989 | 0.9488 | 0.9991 | 0.9601 | 0.9797 | +0.0008 | 0.9870 |
| L1 + drop top-5 (bwd_data_pkts_tot,orig_pkts,orig_ip_bytes,bwd_last_window_size,flow_ACK_flag_count) | 80 | 0.9989 | 0.9487 | 0.9991 | 0.9578 | 0.9785 | +0.0005 | 0.9870 |

Top features (identifier-free model): bwd_data_pkts_tot, orig_pkts, orig_ip_bytes, bwd_last_window_size, flow_ACK_flag_count, flow_FIN_flag_count, resp_pkts, bwd_header_size_min. Reading: removing identifiers and the five most important features leaves real-label MCC essentially unchanged while the shuffled-label MCC stays at 0, so Farm-Flow's separability is redundant across many flow statistics (consistent with synthetic attack traffic), not carried by one leaked column or the ports.

## 9. Duplicate-aware evaluation ladder (item 4)

Same fitted model scored on subsets of one test set. **seen** = exact feature vector present in train; **unseen** = absent. **dedup-group** = identical vectors form one group and whole groups go to train or test (vector-disjoint; test rows keep their multiplicity). Label-conflict ceiling = F1 of an oracle predicting the majority label per identical vector. Columns are F1 / MCC / PR-AUC / benign FPR (mean over resplits; ± SD for F1).

| Dataset | Model | Dup. rows % | Ceiling F1 | IID random (contaminated) | seen share of test | IID - seen subset (F1) | IID - unseen subset (F1) | Dedup-group split | Benchmark split |
|---|---|---|---|---|---|---|---|---|---|
| sensornetguard | DT | 0.0 | 1.0000 | 0.994 / 0.994 / 0.988 / 0.000 (F1 ±0.008) | 0.000 | n/a | 0.994 ± 0.008 | 0.987 / 0.986 / 0.974 / 0.001 (F1 ±0.009) | n/a |
| nsl-kdd | DT | 5.5 | 0.9991 | 0.993 / 0.987 / 0.990 / 0.007 (F1 ±0.000) | 0.079 | 0.991 ± 0.001 | 0.993 ± 0.001 | 0.994 / 0.988 / 0.991 / 0.006 (F1 ±0.000) | 0.777 / 0.634 / 0.829 / 0.027 (F1 ±0.006) |
| unsw-nb15 | DT | 20.6 | 0.9989 | 0.972 / 0.968 / 0.950 / 0.004 (F1 ±0.001) | 0.229 | 0.998 ± 0.000 | 0.860 ± 0.004 | 0.968 / 0.963 / 0.943 / 0.005 (F1 ±0.001) | 0.884 / 0.727 / 0.814 / 0.254 (F1 ±0.001) |
| unsw-nb15 | RF | 20.6 | 0.9989 | 0.972 / 0.968 / 0.998 / 0.004 (F1 ±0.000) | 0.175 | 0.999 ± 0.000 | 0.913 ± 0.001 | 0.971 / 0.967 / 0.998 / 0.004 (F1 ±0.001) | 0.893 / 0.751 / 0.980 / 0.274 (F1 ±0.000) |
| unsw-nb15 | XGB | 20.6 | 0.9989 | 0.972 / 0.968 / 0.998 / 0.004 (F1 ±0.001) | 0.175 | 0.999 ± 0.000 | 0.913 ± 0.001 | 0.971 / 0.967 / 0.998 / 0.004 (F1 ±0.001) | 0.895 / 0.757 / 0.988 / 0.261 (F1 ±0.000) |
| unsw-nb15 | LGBM | 20.6 | 0.9989 | 0.973 / 0.969 / 0.998 / 0.004 (F1 ±0.001) | 0.175 | 0.999 ± 0.000 | 0.914 ± 0.001 | 0.972 / 0.968 / 0.998 / 0.004 (F1 ±0.001) | 0.893 / 0.753 / 0.988 / 0.263 (F1 ±0.000) |
| farm-flow | DT | 79.7 | 1.0000 | 0.999 / 0.955 / 0.999 / 0.045 (F1 ±0.000) | 0.807 | 1.000 ± 0.000 | 0.995 ± 0.000 | 0.998 / 0.955 / 0.998 / 0.045 (F1 ±0.001) | n/a |
| cic-iov-2024 | DT | 99.7 | 1.0000 | 1.000 / 1.000 / 1.000 / 0.000 (F1 ±0.000) | 0.998 | 1.000 ± 0.000 | n/a | 0.729 / 0.728 / 0.635 / 0.005 (F1 ±0.138) | n/a |
| cic-iov-2024 | RF | 99.7 | 1.0000 | 1.000 / 1.000 / 1.000 / 0.000 (F1 ±0.000) | 0.997 | 1.000 ± 0.000 | n/a | 0.854 / 0.852 / 0.924 / 0.000 (F1 ±0.136) | n/a |
| cic-iov-2024 | XGB | 99.7 | 1.0000 | 1.000 / 1.000 / 1.000 / 0.000 (F1 ±0.000) | 0.997 | 1.000 ± 0.000 | n/a | 0.786 / 0.781 / 0.903 / 0.004 (F1 ±0.144) | n/a |
| cic-iov-2024 | LGBM | 99.7 | 1.0000 | 1.000 / 1.000 / 1.000 / 0.000 (F1 ±0.000) | 0.997 | 1.000 ± 0.000 | n/a | 0.774 / 0.768 / 0.897 / 0.004 (F1 ±0.129) | n/a |
| cicids2017 | DT | 11.7 | 0.9993 | 0.997 / 0.997 / 0.995 / 0.001 (F1 ±0.000) | 0.144 | 0.997 ± 0.000 | 0.997 ± 0.000 | 0.989 / 0.986 / 0.982 / 0.001 (F1 ±0.019) | n/a |

Temporal and group protocols (decision tree; single run for time, mean over group resplits; F1 / MCC / PR-AUC / FPR; test attack rate in brackets):

| Dataset | Time split | Group split |
|---|---|---|
| sensornetguard | 0.976 / 0.975 / 0.954 / 0.0011 [atk 0.05] | N/A (no group key) |
| nsl-kdd | N/A (no timestamp) | N/A (no group key) |
| unsw-nb15 | 0.964 / 0.956 / 0.939 / 0.0092 [atk 0.20] | 0.940 / 0.935 / 0.893 / 0.006 (F1 ±0.054) |
| farm-flow | N/A (single-class test slice) | 0.989 / 0.756 / 0.984 / 0.282 (F1 ±0.010) |
| cic-iov-2024 | 0.927 / 0.920 / 0.985 / 0.0000 [atk 0.13] | N/A (no group key) |
| cicids2017 | 0.849 / 0.778 / 0.839 / 0.0125 [atk 0.42] | see ids2017_groups.json (source-IP groups are degenerate) |

## 10. CIC-IDS2017 group splits from raw provenance (item 3)

Raw flows with Flow ID, IPs and timestamps (n = 2,830,743; attack rate 0.197). A fold is **valid** if the test side has ≥ 50 attack and ≥ 50 benign rows and an attack rate within ±25% (relative) of overall; group overlap between train and test is asserted to be zero (time-block folds also drop ±1 block of guard rows from train). Attacks come from only 10 source IPs, 555,641 of 557,646 attack flows from 172.16.0.1 (all to 192.168.10.50), so entity keys that follow the attacker box cannot produce a balanced held-out set.

| Group key | Valid folds | F1 (mean ± SD) | MCC | PR-AUC | Recall | Benign FPR | Test attack rate | #groups | Guard rows dropped |
|---|---|---|---|---|---|---|---|---|---|
| source_ip | 0/5 | no valid fold (degenerate: attacks concentrated in one entity) | | | | | | | |
| src_dst_pair | 0/5 | no valid fold (degenerate: attacks concentrated in one entity) | | | | | | | |
| flow_id_session | 5/5 | 0.9969 ± 0.0002 | 0.9961 | 0.9943 | 0.9971 | 0.0008 | 0.197 | 1,085,071 | 0 |
| destination_host | 0/5 | no valid fold (degenerate: attacks concentrated in one entity) | | | | | | | |
| timeblock_1min | 5/5 | 0.9682 ± 0.0486 | 0.9624 | 0.9523 | 0.9474 | 0.0014 | 0.185 | 2,454 | 819,931 |
| timeblock_5min | 5/5 | 0.9356 ± 0.0745 | 0.9267 | 0.9056 | 0.8913 | 0.0016 | 0.199 | 496 | 845,067 |
| timeblock_15min | 5/5 | 0.8213 ± 0.0955 | 0.7979 | 0.7596 | 0.7233 | 0.0070 | 0.221 | 171 | 716,523 |

Reading: the earlier source-IP group F1 of 0.061 was a degenerate split (kept in `ladder_cicids2017.json` for transparency). Entity-level keys that are not tied to the attacker box (Flow-ID/session, time blocks) give valid folds; performance falls monotonically as the time block widens (more temporal separation), the legitimate group-generalisation estimate.

## 11. CIC-IoV novelty experiment (item 5)

1,408,219 rows but only 3,575 unique 136-bit vectors; training uses unique vectors with multiplicity weights (equivalent to row-level training for a decision tree). Claimable only if a cell has ≥ 2000 test rows and ≥ 100 positives; otherwise it is descriptive. Novelty is measured against the training rows of each split: exact-unseen (Hamming ≥ 1), Hamming ≥ k bits from every training vector, and frame-novel (some of the 8 CAN-frame patterns never seen at that position). 95% CIs are test-set bootstrap (not replication).

| Split | Novelty level | n rows | n positives | Claimable | DT F1 [CI] | DT MCC | DT PR-AUC | DT benign FPR | LGBM F1 | LGBM MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| S1_row_stratified_80_20 | all_test_rows | 281,644 | 36,896 | yes | 0.9999 [1.000, 1.000] | 0.9999 | 0.9999 | 0.0000 | 1.0000 | 1.0000 |
| S1_row_stratified_80_20 | exact_unseen(d>=1) | 551 | 0 | no | too few rows / single class | | | | | |
| S1_row_stratified_80_20 | hamming>=2 | 139 | 0 | no | too few rows / single class | | | | | |
| S1_row_stratified_80_20 | hamming>=3 | 0 | 0 | no | too few rows / single class | | | | | |
| S1_row_stratified_80_20 | hamming>=5 | 0 | 0 | no | too few rows / single class | | | | | |
| S1_row_stratified_80_20 | frame_novel(any frame) | 1 | 0 | no | too few rows / single class | | | | | |
| S2_vector_disjoint_80_20 | all_test_rows | 313,102 | 27,375 | yes | 0.6275 [0.624, 0.632] | 0.6034 | 0.4323 | 0.0756 | 0.8999 | 0.8966 |
| S2_vector_disjoint_80_20 | exact_unseen(d>=1) | 313,102 | 27,375 | yes | 0.6275 [0.624, 0.632] | 0.6034 | 0.4323 | 0.0756 | 0.8999 | 0.8966 |
| S2_vector_disjoint_80_20 | hamming>=2 | 286,860 | 27,357 | yes | 0.6277 [0.624, 0.632] | 0.5994 | 0.4339 | 0.0832 | 0.9002 | 0.8962 |
| S2_vector_disjoint_80_20 | hamming>=3 | 246,402 | 27,357 | yes | 0.6277 [0.624, 0.632] | 0.5906 | 0.4368 | 0.0986 | 0.9002 | 0.8946 |
| S2_vector_disjoint_80_20 | hamming>=5 | 197,816 | 27,357 | yes | 0.6277 [0.624, 0.632] | 0.5743 | 0.4417 | 0.1267 | 0.9002 | 0.8918 |
| S2_vector_disjoint_80_20 | frame_novel(any frame) | 247,355 | 22,387 | yes | 0.5674 [0.563, 0.572] | 0.5370 | 0.3675 | 0.0960 | 0.8753 | 0.8726 |
| S3_vector_disjoint_50_50 | all_test_rows | 643,064 | 79,766 | yes | 0.4706 [0.468, 0.473] | 0.4497 | 0.3049 | 0.3042 | 0.8145 | 0.8112 |
| S3_vector_disjoint_50_50 | exact_unseen(d>=1) | 643,064 | 79,766 | yes | 0.4706 [0.468, 0.473] | 0.4497 | 0.3049 | 0.3042 | 0.8145 | 0.8112 |
| S3_vector_disjoint_50_50 | hamming>=2 | 598,661 | 79,748 | yes | 0.4706 [0.469, 0.473] | 0.4403 | 0.3052 | 0.3302 | 0.8147 | 0.8098 |
| S3_vector_disjoint_50_50 | hamming>=3 | 505,626 | 79,748 | yes | 0.4706 [0.469, 0.472] | 0.4129 | 0.3060 | 0.4023 | 0.8147 | 0.8058 |
| S3_vector_disjoint_50_50 | hamming>=5 | 380,940 | 79,748 | yes | 0.4967 [0.495, 0.499] | 0.3809 | 0.3301 | 0.5115 | 0.8147 | 0.7967 |
| S3_vector_disjoint_50_50 | frame_novel(any frame) | 530,424 | 69,783 | yes | 0.4394 [0.437, 0.442] | 0.4060 | 0.2791 | 0.3672 | 0.8330 | 0.8272 |

## 12. NSL-KDD: why the benchmark split is harder than a random split (item 6)

- **Class prior**: attack share train 0.465, benchmark test 0.569 (KDDTest-21: 0.818), random-split test 0.481.
- **Families**: 22 attack families in train, 37 in test; **17 test families are absent from train** (apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm), covering 29.2% of test attack rows (3,750 rows).
- **Covariate shift** (adversarial validation AUC, LightGBM 5-fold; 0.5 = indistinguishable): train vs benchmark test 0.897 (benign only 0.795, attack only 0.971) versus 0.496 for a random split. Top shifted features (KS, log1p): 39 (0.23), 37 (0.20), 38 (0.20), 24 (0.19), 25 (0.18).

Counterfactual decomposition (F1 / MCC on the test set; same trained model per row):

| Model | Random split | Benchmark (full) | Benchmark, novel-family attacks removed | …and re-weighted to random-split family mix & prior | Recall seen-family attacks | Recall novel-family attacks | Benign FPR |
|---|---|---|---|---|---|---|---|
| DT | 0.993 / 0.986 | 0.776 / 0.632 | 0.854 / 0.760 | 0.960 / 0.924 | 0.767 | 0.354 | 0.027 |
| LGBM | 0.995 / 0.991 | 0.764 / 0.620 | 0.856 / 0.764 | 0.966 / 0.934 | 0.770 | 0.294 | 0.026 |

Partition check: 9,711 benign + 9,083 seen-family attacks + 3,750 novel-family attacks = 22,544 test rows. The re-weighting is approximate (importance weights on seen-family attack rows to match the random split's family mix and class prior). Reading: removing novel-family attacks recovers part of the gap, matching the family mix recovers most of the rest, and a residual remains that is covariate shift within shared families.

## 13. RQ1 under non-IID protocols: F1 / MCC / PR-AUC retention of derived features (item 7)

Raw vs derived on identical splits (train capped at 600k rows). Retention = derived / raw (mean over folds). ΔF1 = mean paired difference (derived − raw) with the mean of the per-fold paired-bootstrap 95% limits (test-row resampling; approximate pooled CI). Protocols: random (5 resplits), dedup-unseen (vector-disjoint, 3), time (1 run), group (entity or 5-min time block, 3 folds) where valid.

| Dataset (raw → derived) | Protocol | Model | Raw F1 | Derived F1 | F1 retention | MCC retention | PR-AUC retention | ΔF1 [paired 95% CI] | Benign FPR raw → derived |
|---|---|---|---|---|---|---|---|---|---|
| sensornetguard (17 → 4) | random | DT | 0.9938 | 0.9804 | 98.6% | 98.6% | 97.4% | -0.0134 [-0.0345, +0.0076] | 0.0004 → 0.0009 |
| sensornetguard (17 → 4) | random | LGBM | 0.9949 | 0.9835 | 98.9% | 98.8% | 99.9% | -0.0113 [-0.0340, +0.0050] | 0.0003 → 0.0009 |
| sensornetguard (17 → 4) | dedup_unseen | DT | 0.9864 | 0.9788 | 99.2% | 99.2% | 98.5% | -0.0077 [-0.0305, +0.0138] | 0.0011 → 0.0012 |
| sensornetguard (17 → 4) | dedup_unseen | LGBM | 0.9981 | 0.9910 | 99.3% | 99.3% | 100.0% | -0.0071 [-0.0220, +0.0061] | 0.0000 → 0.0002 |
| sensornetguard (17 → 4) | time | DT | 0.9758 | 0.9806 | 100.5% | 100.5% | 101.0% | +0.0047 [+0.0000, +0.0159] | 0.0011 → 0.0005 |
| sensornetguard (17 → 4) | time | LGBM | 0.9952 | 0.9856 | 99.0% | 99.0% | 99.9% | -0.0095 [-0.0296, +0.0054] | 0.0000 → 0.0011 |
| nsl-kdd (38 → 11) | random | DT | 0.9930 | 0.9721 | 97.9% | 95.9% | 97.4% | -0.0209 [-0.0228, -0.0189] | 0.0069 → 0.0426 |
| nsl-kdd (38 → 11) | random | LGBM | 0.9957 | 0.9720 | 97.6% | 95.3% | 97.0% | -0.0238 [-0.0257, -0.0220] | 0.0032 → 0.0426 |
| nsl-kdd (38 → 11) | dedup_unseen | DT | 0.9939 | 0.9722 | 97.8% | 95.7% | 97.3% | -0.0218 [-0.0237, -0.0198] | 0.0064 → 0.0427 |
| nsl-kdd (38 → 11) | dedup_unseen | LGBM | 0.9961 | 0.9719 | 97.6% | 95.3% | 96.9% | -0.0243 [-0.0260, -0.0222] | 0.0033 → 0.0427 |
| unsw-nb15 (38 → 17) | random | DT | 0.9669 | 0.9611 | 99.4% | 99.3% | 99.6% | -0.0058 [-0.0069, -0.0047] | 0.0049 → 0.0055 |
| unsw-nb15 (38 → 17) | random | LGBM | 0.9744 | 0.9671 | 99.2% | 99.1% | 99.9% | -0.0073 [-0.0082, -0.0065] | 0.0034 → 0.0046 |
| unsw-nb15 (38 → 17) | dedup_unseen | DT | 0.9644 | 0.9586 | 99.4% | 99.3% | 99.7% | -0.0059 [-0.0070, -0.0048] | 0.0052 → 0.0059 |
| unsw-nb15 (38 → 17) | dedup_unseen | LGBM | 0.9732 | 0.9658 | 99.2% | 99.1% | 99.9% | -0.0073 [-0.0082, -0.0065] | 0.0038 → 0.0050 |
| unsw-nb15 (38 → 17) | time | DT | 0.9639 | 0.9600 | 99.6% | 99.5% | 100.2% | -0.0038 [-0.0046, -0.0029] | 0.0093 → 0.0093 |
| unsw-nb15 (38 → 17) | time | LGBM | 0.9709 | 0.9676 | 99.7% | 99.6% | 99.9% | -0.0032 [-0.0039, -0.0024] | 0.0079 → 0.0084 |
| unsw-nb15 (38 → 17) | group(entity) | DT | 0.9200 | 0.9045 | 98.3% | 98.1% | 97.9% | -0.0154 [-0.0175, -0.0131] | 0.0069 → 0.0087 |
| unsw-nb15 (38 → 17) | group(entity) | LGBM | 0.9384 | 0.9236 | 98.4% | 98.3% | 99.0% | -0.0149 [-0.0165, -0.0130] | 0.0049 → 0.0061 |
| farm-flow (85 → 10) | random | DT | 0.9990 | 0.9984 | 99.9% | 97.2% | 100.0% | -0.0005 [-0.0006, -0.0004] | 0.0476 → 0.0757 |
| farm-flow (85 → 10) | random | LGBM | 0.9993 | 0.9987 | 99.9% | 97.6% | 100.0% | -0.0005 [-0.0006, -0.0004] | 0.0222 → 0.0236 |
| farm-flow (85 → 10) | dedup_unseen | DT | 0.9988 | 0.9979 | 99.9% | 96.2% | 99.9% | -0.0009 [-0.0011, -0.0008] | 0.0469 → 0.0771 |
| farm-flow (85 → 10) | dedup_unseen | LGBM | 0.9991 | 0.9983 | 99.9% | 97.0% | 100.0% | -0.0008 [-0.0010, -0.0007] | 0.0223 → 0.0181 |
| farm-flow (85 → 10) | group(entity) | DT | 0.9806 | 0.9843 | 100.4% | 114.6% | 100.7% | +0.0037 [+0.0030, +0.0041] | 0.4072 → 0.2920 |
| farm-flow (85 → 10) | group(entity) | LGBM | 0.9866 | 0.9895 | 100.3% | 112.4% | 100.0% | +0.0029 [+0.0024, +0.0033] | 0.4248 → 0.2289 |
| cic-iov-2024 (136 → 4) | random | DT | 1.0000 | 0.9914 | 99.1% | 99.0% | 100.0% | -0.0086 [-0.0093, -0.0080] | 0.0000 → 0.0026 |
| cic-iov-2024 (136 → 4) | random | LGBM | 1.0000 | 0.9914 | 99.1% | 99.0% | 100.0% | -0.0086 [-0.0093, -0.0080] | 0.0000 → 0.0026 |
| cic-iov-2024 (136 → 4) | dedup_unseen | DT | 0.7913 | 0.5962 | 75.3% | 76.7% | 68.3% | -0.1951 [-0.2002, -0.1896] | 0.0043 → 0.0197 |
| cic-iov-2024 (136 → 4) | dedup_unseen | LGBM | 0.8269 | 0.7400 | 89.5% | 91.8% | 90.8% | -0.0869 [-0.0926, -0.0812] | 0.0045 → 0.0113 |
| cic-iov-2024 (136 → 4) | time | DT | 0.9274 | 0.8294 | 89.4% | 88.8% | 77.7% | -0.0980 [-0.1006, -0.0958] | 0.0000 → 0.0044 |
| cic-iov-2024 (136 → 4) | time | LGBM | 0.8435 | 0.8295 | 98.3% | 97.7% | 81.9% | -0.0140 [-0.0149, -0.0133] | 0.0000 → 0.0044 |
| cicids2017 (79 → 18) | random | DT | 0.9966 | 0.9601 | 96.3% | 95.4% | 98.4% | -0.0365 [-0.0373, -0.0357] | 0.0009 → 0.0141 |
| cicids2017 (79 → 18) | random | LGBM | 0.9978 | 0.9624 | 96.4% | 95.6% | 99.5% | -0.0355 [-0.0363, -0.0347] | 0.0007 → 0.0128 |
| cicids2017 (79 → 18) | dedup_unseen | DT | 0.9962 | 0.9467 | 95.0% | 93.9% | 98.1% | -0.0496 [-0.0505, -0.0486] | 0.0009 → 0.0118 |
| cicids2017 (79 → 18) | dedup_unseen | LGBM | 0.9977 | 0.9504 | 95.3% | 94.2% | 99.6% | -0.0473 [-0.0482, -0.0464] | 0.0008 → 0.0104 |
| cicids2017 (79 → 18) | time | DT | 0.7543 | 0.7078 | 93.8% | 93.6% | 102.2% | -0.0465 [-0.0476, -0.0455] | 0.0112 → 0.0087 |
| cicids2017 (79 → 18) | time | LGBM | 0.8682 | 0.6756 | 77.8% | 74.5% | 97.5% | -0.1925 [-0.1938, -0.1911] | 0.0002 → 0.0078 |
| cicids2017 (79 → 18) | group(5min time-block) | DT | 0.9372 | 0.8509 | 90.8% | 88.8% | 91.6% | -0.0863 [-0.0876, -0.0850] | 0.0006 → 0.0103 |
| cicids2017 (79 → 18) | group(5min time-block) | LGBM | 0.9396 | 0.8300 | 88.3% | 85.7% | 96.6% | -0.1096 [-0.1108, -0.1083] | 0.0005 → 0.0164 |

## 14. Capacity and model complexity: input compression vs model compression (item 8)

One fixed 80/20 split (pool ≤ 500k rows), decision tree with depth cap. Latency = median µs per row over 15 repeats of a 100k-row batch (IQR in brackets), single process; peak = tracemalloc peak during `predict` on that batch.

| Dataset | Depth cap | Features raw → derived | Nodes raw → derived | Leaves raw → derived | Model KB raw → derived | Latency µs/row raw → derived | Peak predict MB raw → derived | F1 raw → derived | MCC raw → derived |
|---|---|---|---|---|---|---|---|---|---|
| sensornetguard | 5 | 17 → 4 | 15 → 19 | 8 → 10 | 3 → 3 | 0.050 [0.049-0.051] → 0.048 [0.047-0.049] | 0.1 → 0.1 | 0.9898 → 0.9794 | 0.9893 → 0.9783 |
| sensornetguard | 10 | 17 → 4 | 21 → 33 | 11 → 17 | 4 → 4 | 0.050 [0.050-0.051] → 0.052 [0.051-0.062] | 0.1 → 0.1 | 0.9898 → 0.9846 | 0.9893 → 0.9838 |
| sensornetguard | unlimited | 17 → 4 | 21 → 33 | 11 → 17 | 4 → 4 | 0.050 [0.050-0.052] → 0.049 [0.049-0.050] | 0.1 → 0.1 | 0.9898 → 0.9846 | 0.9893 → 0.9838 |
| nsl-kdd | 5 | 38 → 11 | 61 → 63 | 31 → 32 | 7 → 7 | 0.036 [0.035-0.037] → 0.032 [0.031-0.033] | 1.0 → 1.0 | 0.9712 → 0.9396 | 0.9442 → 0.8820 |
| nsl-kdd | 10 | 38 → 11 | 439 → 445 | 220 → 223 | 36 → 37 | 0.044 [0.043-0.047] → 0.042 [0.040-0.043] | 1.0 → 1.0 | 0.9906 → 0.9686 | 0.9819 → 0.9391 |
| nsl-kdd | unlimited | 38 → 11 | 1,719 → 1,001 | 860 → 501 | 136 → 80 | 0.048 [0.048-0.049] → 0.049 [0.048-0.049] | 1.0 → 1.0 | 0.9928 → 0.9718 | 0.9861 → 0.9452 |
| unsw-nb15 | 5 | 39 → 17 | 39 → 53 | 20 → 27 | 5 → 6 | 0.036 [0.035-0.042] → 0.036 [0.035-0.038] | 1.7 → 1.7 | 0.9308 → 0.9112 | 0.8181 → 0.7383 |
| unsw-nb15 | 10 | 39 → 17 | 543 → 691 | 272 → 346 | 45 → 56 | 0.047 [0.046-0.050] → 0.047 [0.046-0.049] | 1.7 → 1.7 | 0.9475 → 0.9397 | 0.8573 → 0.8304 |
| unsw-nb15 | unlimited | 39 → 17 | 16,673 → 21,621 | 8,337 → 10,811 | 1305 → 1691 | 0.071 [0.070-0.077] → 0.076 [0.075-0.081] | 1.7 → 1.7 | 0.9509 → 0.9417 | 0.8636 → 0.8381 |
| farm-flow | 5 | 87 → 10 | 37 → 51 | 19 → 26 | 6 → 6 | 0.033 [0.032-0.034] → 0.021 [0.020-0.021] | 3.2 → 3.2 | 0.9982 → 0.9988 | 0.9155 → 0.9470 |
| farm-flow | 10 | 87 → 10 | 203 → 243 | 102 → 122 | 19 → 21 | 0.041 [0.040-0.049] → 0.022 [0.022-0.030] | 3.2 → 3.2 | 0.9991 → 0.9989 | 0.9561 → 0.9494 |
| farm-flow | unlimited | 87 → 10 | 1,005 → 2,325 | 503 → 1,163 | 82 → 183 | 0.046 [0.044-0.054] → 0.022 [0.022-0.023] | 3.2 → 3.2 | 0.9990 → 0.9983 | 0.9503 → 0.9176 |
| cic-iov-2024 | 5 | 136 → 4 | 35 → 33 | 18 → 17 | 6 → 4 | 0.052 [0.051-0.055] → 0.030 [0.030-0.030] | 3.2 → 3.2 | 0.9912 → 0.8906 | 0.9899 → 0.8741 |
| cic-iov-2024 | 10 | 136 → 4 | 61 → 65 | 31 → 33 | 8 → 7 | 0.062 [0.061-0.071] → 0.031 [0.031-0.031] | 3.2 → 3.2 | 0.9994 → 0.9914 | 0.9993 → 0.9901 |
| cic-iov-2024 | unlimited | 136 → 4 | 73 → 65 | 37 → 33 | 9 → 7 | 0.062 [0.061-0.064] → 0.031 [0.031-0.032] | 3.2 → 3.2 | 0.9999 → 0.9914 | 0.9999 → 0.9901 |
| cicids2017 | 5 | 78 → 18 | 41 → 55 | 21 → 28 | 7 → 6 | 0.040 [0.039-0.041] → 0.027 [0.026-0.028] | 3.2 → 3.2 | 0.9581 → 0.8985 | 0.9477 → 0.8744 |
| cicids2017 | 10 | 78 → 18 | 159 → 487 | 80 → 244 | 16 → 40 | 0.046 [0.045-0.047] → 0.040 [0.039-0.041] | 3.2 → 3.2 | 0.9918 → 0.9482 | 0.9898 → 0.9355 |
| cicids2017 | unlimited | 78 → 18 | 1,419 → 4,567 | 710 → 2,284 | 114 → 359 | 0.069 [0.068-0.072] → 0.059 [0.059-0.061] | 3.2 → 3.2 | 0.9960 → 0.9596 | 0.9950 → 0.9497 |

Verdict at unlimited depth (derived / raw):

| Dataset | Input dimension reduction | Node ratio | Serialized-size ratio | Latency ratio | Model-level compression? |
|---|---|---|---|---|---|
| sensornetguard | 76% | 1.57× | 1.17× | 0.99× | no (larger tree) |
| nsl-kdd | 71% | 0.58× | 0.59× | 1.01× | yes (smaller tree) |
| unsw-nb15 | 56% | 1.30× | 1.30× | 1.07× | no (larger tree) |
| farm-flow | 89% | 2.31× | 2.24× | 0.48× | no (larger tree) |
| cic-iov-2024 | 97% | 0.89× | 0.73× | 0.50× | no (about the same) |
| cicids2017 | 77% | 3.22× | 3.14× | 0.86× | no (larger tree) |

Conclusion: semantic derivation compresses the **input representation**; it does not systematically shrink the fitted tree, its serialized size or inference cost.

## 15. Cross-dataset transfer matrix on harmonised features (item 9)

Rows = train dataset, columns = test dataset. Each cell uses only the harmonised features computable in **both** datasets (count in the feature table). Source train ≤ 300k, target test ≤ 200k (stratified), 3 source subsamples. `quantile` = per-dataset quantile transform fitted label-free on each dataset's own train. SensorNetGuard contributes only 3 node-health rate features (partial node; cells are descriptive). Diagonal = within-dataset on the same features.

**As-is features, decision tree (depth 10): MCC (F1)**

| train \ test | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow | sensornetguard |
|---|---|---|---|---|---|
| **unsw-nb15** | +0.65 ± 0.00 (0.85) | -0.04 ± 0.42 (0.64) | -0.27 ± 0.10 (0.18) | -0.22 ± 0.02 (0.27) | +0.00 ± 0.00 (0.09) |
| **nsl-kdd** | -0.25 ± 0.02 (0.14) | +0.59 ± 0.00 (0.75) | -0.06 ± 0.21 (0.25) | +0.03 ± 0.04 (0.46) | +0.11 ± 0.00 (0.14) |
| **cicids2017** | -0.01 ± 0.02 (0.00) | +0.52 ± 0.03 (0.69) | +0.93 ± 0.00 (0.95) | +0.01 ± 0.00 (0.01) | +0.00 ± 0.00 (0.00) |
| **farm-flow** | +0.19 ± 0.03 (0.71) | +0.06 ± 0.03 (0.73) | +0.07 ± 0.11 (0.33) | +0.94 ± 0.00 (1.00) | +0.00 ± 0.00 (0.09) |
| **sensornetguard** | -0.13 ± 0.00 (0.01) | +0.61 ± 0.00 (0.75) | -0.07 ± 0.00 (0.22) | -0.04 ± 0.00 (0.34) | +0.97 ± 0.00 (0.98) |

**Quantile-transformed features, LightGBM: MCC (F1)**

| train \ test | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow | sensornetguard |
|---|---|---|---|---|---|
| **unsw-nb15** | +0.74 ± 0.00 (0.89) | -0.04 ± 0.00 (0.35) | +0.14 ± 0.00 (0.37) | +0.16 ± 0.00 (0.73) | +0.10 ± 0.00 (0.11) |
| **nsl-kdd** | +0.43 ± 0.00 (0.70) | +0.64 ± 0.00 (0.79) | -0.03 ± 0.00 (0.25) | +0.17 ± 0.00 (0.75) | +0.07 ± 0.00 (0.10) |
| **cicids2017** | +0.04 ± 0.06 (0.08) | -0.04 ± 0.32 (0.22) | +0.95 ± 0.00 (0.96) | -0.01 ± 0.03 (0.12) | -0.15 ± 0.00 (0.00) |
| **farm-flow** | +0.05 ± 0.02 (0.71) | -0.01 ± 0.01 (0.73) | +0.04 ± 0.00 (0.33) | +0.94 ± 0.00 (1.00) | +0.00 ± 0.00 (0.09) |
| **sensornetguard** | +0.04 ± 0.00 (0.04) | +0.01 ± 0.00 (0.00) | -0.01 ± 0.00 (0.00) | +0.01 ± 0.00 (0.02) | +0.97 ± 0.00 (0.97) |

**As-is features, LightGBM: MCC (F1)**

| train \ test | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow | sensornetguard |
|---|---|---|---|---|---|
| **unsw-nb15** | +0.74 ± 0.00 (0.89) | -0.24 ± 0.00 (0.26) | -0.06 ± 0.00 (0.23) | -0.26 ± 0.00 (0.15) | -0.60 ± 0.00 (0.01) |
| **nsl-kdd** | -0.11 ± 0.00 (0.04) | +0.63 ± 0.00 (0.78) | +0.24 ± 0.00 (0.42) | +0.08 ± 0.00 (0.37) | +0.18 ± 0.00 (0.19) |
| **cicids2017** | +0.01 ± 0.01 (0.00) | +0.09 ± 0.19 (0.21) | +0.95 ± 0.00 (0.96) | +0.00 ± 0.00 (0.00) | +0.00 ± 0.00 (0.00) |
| **farm-flow** | +0.14 ± 0.05 (0.70) | +0.05 ± 0.03 (0.73) | -0.07 ± 0.04 (0.31) | +0.94 ± 0.00 (1.00) | +0.00 ± 0.00 (0.09) |
| **sensornetguard** | +0.15 ± 0.00 (0.64) | +0.61 ± 0.00 (0.75) | +0.00 ± 0.00 (0.00) | +0.05 ± 0.00 (0.18) | +0.97 ± 0.00 (0.97) |

**Quantile-transformed features, decision tree: MCC (F1)**

| train \ test | unsw-nb15 | nsl-kdd | cicids2017 | farm-flow | sensornetguard |
|---|---|---|---|---|---|
| **unsw-nb15** | +0.65 ± 0.00 (0.85) | -0.38 ± 0.48 (0.32) | -0.01 ± 0.02 (0.31) | +0.19 ± 0.05 (0.76) | +0.10 ± 0.00 (0.11) |
| **nsl-kdd** | +0.46 ± 0.00 (0.72) | +0.59 ± 0.00 (0.75) | -0.02 ± 0.06 (0.26) | +0.06 ± 0.02 (0.39) | +0.16 ± 0.00 (0.13) |
| **cicids2017** | -0.03 ± 0.06 (0.21) | +0.27 ± 0.42 (0.63) | +0.93 ± 0.00 (0.95) | +0.05 ± 0.02 (0.39) | -0.01 ± 0.03 (0.08) |
| **farm-flow** | +0.10 ± 0.02 (0.71) | -0.03 ± 0.07 (0.72) | +0.02 ± 0.07 (0.33) | +0.94 ± 0.00 (1.00) | +0.01 ± 0.01 (0.09) |
| **sensornetguard** | -0.10 ± 0.00 (0.02) | +0.01 ± 0.00 (0.00) | -0.03 ± 0.00 (0.00) | +0.02 ± 0.00 (0.03) | +0.98 ± 0.00 (0.98) |

Harmonised feature count per cell: unsw-nb15->nsl-kdd: 6; unsw-nb15->cicids2017: 9; unsw-nb15->farm-flow: 9; unsw-nb15->sensornetguard: 2; nsl-kdd->unsw-nb15: 6; nsl-kdd->cicids2017: 7; nsl-kdd->farm-flow: 7; nsl-kdd->sensornetguard: 2; cicids2017->unsw-nb15: 9; cicids2017->nsl-kdd: 7; cicids2017->farm-flow: 10; cicids2017->sensornetguard: 3; farm-flow->unsw-nb15: 9; farm-flow->nsl-kdd: 7; farm-flow->cicids2017: 10; farm-flow->sensornetguard: 3; sensornetguard->unsw-nb15: 2; sensornetguard->nsl-kdd: 2; sensornetguard->cicids2017: 3; sensornetguard->farm-flow: 3.

**Permutation null for off-diagonal cells (as-is DT; 20 permutations of the source training labels).** A single shuffled run is not a valid null under distribution shift - with few features a noise-fit tree can reach |MCC| ≈ 0.2-0.35 by chance - so real MCC is compared with the null distribution.

| Transfer | Features | Real MCC | Null MCC (mean ± SD) | z | One-sided p |
|---|---|---|---|---|---|
| unsw-nb15->nsl-kdd | 6 | -0.044 | +0.027 ± 0.225 | -0.3 | 0.60 |
| unsw-nb15->cicids2017 | 9 | -0.265 | -0.128 ± 0.185 | -0.7 | 0.80 |
| unsw-nb15->farm-flow | 9 | -0.217 | -0.013 ± 0.019 | -11.0 | 1.00 |
| unsw-nb15->sensornetguard | 2 | +0.000 | -0.035 ± 0.093 | +0.4 | 0.75 |
| nsl-kdd->unsw-nb15 | 6 | -0.246 | +0.010 ± 0.264 | -1.0 | 0.85 |
| nsl-kdd->cicids2017 | 7 | -0.064 | +0.029 ± 0.267 | -0.3 | 0.40 |
| nsl-kdd->farm-flow | 7 | +0.030 | -0.022 ± 0.083 | +0.6 | 0.15 |
| nsl-kdd->sensornetguard | 2 | +0.108 | +0.020 ± 0.194 | +0.5 | 0.10 |
| cicids2017->unsw-nb15 | 9 | -0.006 | +0.018 ± 0.131 | -0.2 | 0.50 |
| cicids2017->nsl-kdd | 7 | +0.518 | +0.033 ± 0.345 | +1.4 | 0.15 |
| cicids2017->farm-flow | 10 | +0.008 | +0.008 ± 0.020 | +0.0 | 0.30 |
| cicids2017->sensornetguard | 3 | +0.000 | +0.065 ± 0.114 | -0.6 | 0.85 |
| farm-flow->unsw-nb15 | 9 | +0.187 | +0.009 ± 0.125 | +1.4 | 0.10 |
| farm-flow->nsl-kdd | 7 | +0.064 | +0.013 ± 0.165 | +0.3 | 0.20 |
| farm-flow->cicids2017 | 10 | +0.072 | -0.099 ± 0.200 | +0.9 | 0.10 |
| farm-flow->sensornetguard | 3 | +0.000 | +0.001 ± 0.002 | -0.3 | 1.00 |
| sensornetguard->unsw-nb15 | 2 | -0.132 | +0.005 ± 0.112 | -1.2 | 0.90 |
| sensornetguard->nsl-kdd | 2 | +0.612 | -0.034 ± 0.311 | +2.1 | 0.00 |
| sensornetguard->cicids2017 | 3 | -0.069 | +0.001 ± 0.111 | -0.6 | 0.70 |
| sensornetguard->farm-flow | 3 | -0.041 | -0.023 ± 0.132 | -0.1 | 0.55 |

Summary (as-is DT): diagonal MCC 0.59-0.97; off-diagonal MCC median +0.00, range -0.27 to +0.61; 2 of 20 off-diagonal cells exceed MCC 0.5.

## 16. Paired comparisons with bootstrap 95% CIs (item 10)

Identical test rows for both arms; 300 bootstrap resamples of the test rows. These CIs capture **test-set sampling** for a single split (random seed 42, benchmark, or time) and are not independent replications; mean ± SD is reserved for repeated random resplits (Sections 6 and 9).

**RQ1 - derived minus raw (ΔF1, ΔMCC, ΔPR-AUC), per model**

| Dataset | Split | n test | Model | ΔF1 [95% CI] | ΔMCC [95% CI] | ΔPR-AUC [95% CI] |
|---|---|---|---|---|---|---|
| sensornetguard | random_seed42 | 2,000 | DT | -0.0052 [-0.0219, +0.0132] | -0.0055 [-0.0230, +0.0137] | -0.0098 [-0.0421, +0.0265] |
| sensornetguard | random_seed42 | 2,000 | RF | -0.0104 [-0.0283, +0.0000] | -0.0109 [-0.0293, +0.0000] | -0.0013 [-0.0041, +0.0000] |
| sensornetguard | random_seed42 | 2,000 | XGB | -0.0208 [-0.0408, -0.0050] | -0.0218 [-0.0426, -0.0053] | -0.0006 [-0.0018, +0.0000] |
| sensornetguard | random_seed42 | 2,000 | LGBM | -0.0155 [-0.0419, +0.0057] | -0.0163 [-0.0437, +0.0059] | -0.0010 [-0.0028, +0.0000] |
| sensornetguard | time | 2,000 | DT | +0.0047 [+0.0000, +0.0159] | +0.0050 [+0.0000, +0.0166] | +0.0093 [+0.0000, +0.0311] |
| sensornetguard | time | 2,000 | RF | -0.0145 [-0.0319, +0.0000] | -0.0153 [-0.0336, +0.0000] | -0.0003 [-0.0012, +0.0000] |
| sensornetguard | time | 2,000 | XGB | -0.0048 [-0.0226, +0.0093] | -0.0051 [-0.0236, +0.0097] | -0.0005 [-0.0023, +0.0008] |
| sensornetguard | time | 2,000 | LGBM | -0.0095 [-0.0310, +0.0059] | -0.0101 [-0.0323, +0.0061] | -0.0007 [-0.0028, +0.0000] |
| nsl-kdd | random_seed42 | 29,704 | DT | -0.0210 [-0.0231, -0.0190] | -0.0409 [-0.0446, -0.0370] | -0.0258 [-0.0292, -0.0224] |
| nsl-kdd | random_seed42 | 29,704 | RF | -0.0227 [-0.0244, -0.0209] | -0.0442 [-0.0475, -0.0408] | -0.0308 [-0.0334, -0.0283] |
| nsl-kdd | random_seed42 | 29,704 | XGB | -0.0232 [-0.0252, -0.0214] | -0.0451 [-0.0490, -0.0417] | -0.0316 [-0.0342, -0.0292] |
| nsl-kdd | random_seed42 | 29,704 | LGBM | -0.0243 [-0.0261, -0.0224] | -0.0473 [-0.0507, -0.0434] | -0.0303 [-0.0329, -0.0279] |
| nsl-kdd | benchmark | 22,544 | DT | +0.0222 [+0.0169, +0.0283] | +0.0159 [+0.0093, +0.0235] | +0.0012 [-0.0037, +0.0056] |
| nsl-kdd | benchmark | 22,544 | RF | -0.0040 [-0.0087, +0.0002] | -0.0161 [-0.0224, -0.0109] | -0.0514 [-0.0548, -0.0477] |
| nsl-kdd | benchmark | 22,544 | XGB | +0.0227 [+0.0184, +0.0275] | +0.0191 [+0.0133, +0.0248] | -0.0154 [-0.0175, -0.0132] |
| nsl-kdd | benchmark | 22,544 | LGBM | -0.0248 [-0.0297, -0.0198] | -0.0385 [-0.0448, -0.0328] | -0.0277 [-0.0306, -0.0249] |
| unsw-nb15 | random_seed42 | 508,010 | DT | -0.0053 [-0.0064, -0.0042] | -0.0060 [-0.0073, -0.0048] | -0.0016 [-0.0038, +0.0004] |
| unsw-nb15 | random_seed42 | 508,010 | RF | -0.0072 [-0.0081, -0.0063] | -0.0082 [-0.0093, -0.0072] | -0.0023 [-0.0026, -0.0021] |
| unsw-nb15 | random_seed42 | 508,010 | XGB | -0.0059 [-0.0067, -0.0049] | -0.0068 [-0.0076, -0.0056] | -0.0013 [-0.0015, -0.0012] |
| unsw-nb15 | random_seed42 | 508,010 | LGBM | -0.0065 [-0.0074, -0.0057] | -0.0075 [-0.0085, -0.0065] | -0.0015 [-0.0018, -0.0011] |
| unsw-nb15 | time | 508,010 | DT | -0.0040 [-0.0049, -0.0031] | -0.0050 [-0.0061, -0.0039] | +0.0010 [-0.0006, +0.0026] |
| unsw-nb15 | time | 508,010 | RF | -0.0058 [-0.0067, -0.0052] | -0.0073 [-0.0083, -0.0064] | -0.0021 [-0.0023, -0.0019] |
| unsw-nb15 | time | 508,010 | XGB | -0.0044 [-0.0052, -0.0037] | -0.0055 [-0.0064, -0.0046] | -0.0010 [-0.0011, -0.0008] |
| unsw-nb15 | time | 508,010 | LGBM | -0.0048 [-0.0055, -0.0040] | -0.0059 [-0.0068, -0.0050] | -0.0010 [-0.0011, -0.0009] |
| unsw-nb15 | benchmark | 82,332 | DT | -0.0109 [-0.0127, -0.0090] | -0.0271 [-0.0317, -0.0225] | +0.0038 [+0.0013, +0.0065] |
| unsw-nb15 | benchmark | 82,332 | RF | -0.0025 [-0.0041, -0.0011] | -0.0079 [-0.0119, -0.0043] | -0.0112 [-0.0125, -0.0100] |
| unsw-nb15 | benchmark | 82,332 | XGB | -0.0083 [-0.0097, -0.0070] | -0.0208 [-0.0243, -0.0175] | -0.0039 [-0.0043, -0.0035] |
| unsw-nb15 | benchmark | 82,332 | LGBM | -0.0021 [-0.0036, -0.0006] | -0.0056 [-0.0094, -0.0018] | -0.0031 [-0.0036, -0.0027] |
| farm-flow | random_seed42 | 261,978 | DT | -0.0005 [-0.0006, -0.0004] | -0.0229 [-0.0283, -0.0172] | -0.0004 [-0.0006, -0.0003] |
| farm-flow | random_seed42 | 261,978 | RF | -0.0007 [-0.0008, -0.0006] | -0.0343 [-0.0388, -0.0289] | -0.0000 [-0.0001, -0.0000] |
| farm-flow | random_seed42 | 261,978 | XGB | -0.0004 [-0.0005, -0.0004] | -0.0183 [-0.0219, -0.0148] | -0.0000 [-0.0000, -0.0000] |
| farm-flow | random_seed42 | 261,978 | LGBM | -0.0005 [-0.0005, -0.0004] | -0.0205 [-0.0245, -0.0166] | -0.0000 [-0.0000, -0.0000] |
| cic-iov-2024 | random_seed42 | 281,644 | DT | -0.0089 [-0.0095, -0.0082] | -0.0102 [-0.0109, -0.0094] | -0.0005 [-0.0006, -0.0004] |
| cic-iov-2024 | random_seed42 | 281,644 | RF | -0.0089 [-0.0095, -0.0082] | -0.0102 [-0.0109, -0.0094] | -0.0005 [-0.0005, -0.0004] |
| cic-iov-2024 | random_seed42 | 281,644 | XGB | -0.0089 [-0.0095, -0.0082] | -0.0102 [-0.0109, -0.0094] | -0.0005 [-0.0005, -0.0004] |
| cic-iov-2024 | random_seed42 | 281,644 | LGBM | -0.0089 [-0.0095, -0.0082] | -0.0102 [-0.0109, -0.0094] | -0.0005 [-0.0005, -0.0004] |
| cic-iov-2024 | time | 281,647 | DT | -0.0980 [-0.1006, -0.0958] | -0.1030 [-0.1056, -0.1007] | -0.2199 [-0.2237, -0.2165] |
| cic-iov-2024 | time | 281,647 | RF | -0.0979 [-0.1006, -0.0958] | -0.1029 [-0.1056, -0.1007] | -0.2350 [-0.2391, -0.2314] |
| cic-iov-2024 | time | 281,647 | XGB | -0.0979 [-0.1006, -0.0958] | -0.1029 [-0.1056, -0.1007] | -0.1195 [-0.1217, -0.1172] |
| cic-iov-2024 | time | 281,647 | LGBM | -0.0140 [-0.0149, -0.0133] | -0.0195 [-0.0207, -0.0186] | -0.1596 [-0.1629, -0.1563] |
| cicids2017 | random_seed42 | 566,149 | DT | -0.0367 [-0.0375, -0.0358] | -0.0457 [-0.0467, -0.0447] | -0.0166 [-0.0175, -0.0157] |
| cicids2017 | random_seed42 | 566,149 | RF | -0.0355 [-0.0363, -0.0347] | -0.0442 [-0.0452, -0.0432] | -0.0088 [-0.0093, -0.0083] |
| cicids2017 | random_seed42 | 566,149 | XGB | -0.0363 [-0.0371, -0.0355] | -0.0453 [-0.0462, -0.0443] | -0.0058 [-0.0060, -0.0057] |
| cicids2017 | random_seed42 | 566,149 | LGBM | -0.0357 [-0.0366, -0.0350] | -0.0445 [-0.0455, -0.0437] | -0.0055 [-0.0056, -0.0053] |
| cicids2017 | time | 566,149 | DT | -0.2029 [-0.2044, -0.2015] | -0.2195 [-0.2211, -0.2182] | -0.0521 [-0.0529, -0.0513] |
| cicids2017 | time | 566,149 | RF | -0.2152 [-0.2167, -0.2137] | -0.2332 [-0.2348, -0.2318] | -0.1001 [-0.1011, -0.0991] |
| cicids2017 | time | 566,149 | XGB | -0.1609 [-0.1624, -0.1597] | -0.1596 [-0.1611, -0.1585] | -0.0261 [-0.0266, -0.0257] |
| cicids2017 | time | 566,149 | LGBM | -0.2104 [-0.2121, -0.2089] | -0.2266 [-0.2283, -0.2254] | -0.0329 [-0.0334, -0.0324] |

**Model comparison - ensemble minus decision tree (raw features)**

| Dataset | Split | Model | ΔF1 [95% CI] | ΔMCC [95% CI] | ΔPR-AUC [95% CI] |
|---|---|---|---|---|---|
| sensornetguard | random_seed42 | RF | +0.0051 [+0.0000, +0.0197] | +0.0054 [+0.0000, +0.0204] | +0.0102 [+0.0000, +0.0385] |
| sensornetguard | random_seed42 | XGB | +0.0051 [+0.0000, +0.0197] | +0.0054 [+0.0000, +0.0204] | +0.0102 [-0.0000, +0.0385] |
| sensornetguard | random_seed42 | LGBM | +0.0000 [-0.0129, +0.0176] | +0.0000 [-0.0134, +0.0182] | +0.0102 [-0.0000, +0.0385] |
| sensornetguard | time | RF | +0.0242 [+0.0046, +0.0484] | +0.0255 [+0.0048, +0.0506] | +0.0462 [+0.0086, +0.0924] |
| sensornetguard | time | XGB | +0.0144 [-0.0052, +0.0384] | +0.0153 [-0.0054, +0.0403] | +0.0458 [+0.0084, +0.0924] |
| sensornetguard | time | LGBM | +0.0193 [-0.0047, +0.0469] | +0.0204 [-0.0049, +0.0494] | +0.0462 [+0.0086, +0.0924] |
| nsl-kdd | random_seed42 | RF | +0.0024 [+0.0016, +0.0032] | +0.0047 [+0.0031, +0.0061] | +0.0105 [+0.0092, +0.0120] |
| nsl-kdd | random_seed42 | XGB | +0.0017 [+0.0008, +0.0026] | +0.0033 [+0.0016, +0.0051] | +0.0107 [+0.0093, +0.0122] |
| nsl-kdd | random_seed42 | LGBM | +0.0031 [+0.0022, +0.0040] | +0.0059 [+0.0044, +0.0077] | +0.0107 [+0.0094, +0.0122] |
| nsl-kdd | benchmark | RF | -0.0239 [-0.0277, -0.0200] | -0.0260 [-0.0311, -0.0215] | +0.1286 [+0.1249, +0.1325] |
| nsl-kdd | benchmark | XGB | +0.0058 [+0.0029, +0.0088] | +0.0074 [+0.0039, +0.0111] | +0.1463 [+0.1422, +0.1508] |
| nsl-kdd | benchmark | LGBM | -0.0084 [-0.0121, -0.0052] | -0.0093 [-0.0136, -0.0055] | +0.1456 [+0.1413, +0.1501] |
| unsw-nb15 | random_seed42 | RF | +0.0080 [+0.0070, +0.0090] | +0.0092 [+0.0080, +0.0103] | +0.0634 [+0.0613, +0.0652] |
| unsw-nb15 | random_seed42 | XGB | +0.0076 [+0.0067, +0.0086] | +0.0088 [+0.0077, +0.0098] | +0.0636 [+0.0615, +0.0654] |
| unsw-nb15 | random_seed42 | LGBM | +0.0081 [+0.0072, +0.0091] | +0.0093 [+0.0082, +0.0104] | +0.0636 [+0.0615, +0.0655] |
| unsw-nb15 | time | RF | +0.0072 [+0.0065, +0.0080] | +0.0089 [+0.0080, +0.0099] | +0.0646 [+0.0632, +0.0661] |
| unsw-nb15 | time | XGB | +0.0081 [+0.0073, +0.0089] | +0.0100 [+0.0090, +0.0111] | +0.0649 [+0.0635, +0.0664] |
| unsw-nb15 | time | LGBM | +0.0078 [+0.0071, +0.0085] | +0.0097 [+0.0089, +0.0106] | +0.0649 [+0.0635, +0.0664] |
| unsw-nb15 | benchmark | RF | +0.0087 [+0.0074, +0.0102] | +0.0235 [+0.0202, +0.0273] | +0.1660 [+0.1632, +0.1691] |
| unsw-nb15 | benchmark | XGB | +0.0112 [+0.0097, +0.0129] | +0.0290 [+0.0254, +0.0333] | +0.1738 [+0.1708, +0.1769] |
| unsw-nb15 | benchmark | LGBM | +0.0095 [+0.0083, +0.0110] | +0.0247 [+0.0216, +0.0286] | +0.1736 [+0.1706, +0.1767] |
| farm-flow | random_seed42 | RF | +0.0003 [+0.0002, +0.0003] | +0.0133 [+0.0097, +0.0168] | +0.0011 [+0.0010, +0.0013] |
| farm-flow | random_seed42 | XGB | +0.0004 [+0.0003, +0.0004] | +0.0176 [+0.0138, +0.0217] | +0.0012 [+0.0010, +0.0013] |
| farm-flow | random_seed42 | LGBM | +0.0003 [+0.0002, +0.0004] | +0.0140 [+0.0104, +0.0174] | +0.0012 [+0.0010, +0.0013] |
| cic-iov-2024 | random_seed42 | RF | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| cic-iov-2024 | random_seed42 | XGB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| cic-iov-2024 | random_seed42 | LGBM | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] |
| cic-iov-2024 | time | RF | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0151 [+0.0145, +0.0156] |
| cic-iov-2024 | time | XGB | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0032 [-0.0034, -0.0030] |
| cic-iov-2024 | time | LGBM | -0.0839 [-0.0862, -0.0818] | -0.0834 [-0.0855, -0.0813] | +0.0152 [+0.0146, +0.0157] |
| cicids2017 | random_seed42 | RF | +0.0001 [-0.0002, +0.0003] | +0.0001 [-0.0002, +0.0004] | +0.0059 [+0.0055, +0.0064] |
| cicids2017 | random_seed42 | XGB | +0.0020 [+0.0018, +0.0022] | +0.0025 [+0.0022, +0.0028] | +0.0074 [+0.0070, +0.0079] |
| cicids2017 | random_seed42 | LGBM | +0.0019 [+0.0017, +0.0021] | +0.0023 [+0.0021, +0.0026] | +0.0074 [+0.0070, +0.0079] |
| cicids2017 | time | RF | +0.0169 [+0.0165, +0.0173] | +0.0210 [+0.0205, +0.0215] | +0.1173 [+0.1163, +0.1182] |
| cicids2017 | time | XGB | -0.0754 [-0.0763, -0.0745] | -0.0848 [-0.0857, -0.0838] | +0.1114 [+0.1105, +0.1123] |
| cicids2017 | time | LGBM | +0.0126 [+0.0122, +0.0131] | +0.0158 [+0.0153, +0.0164] | +0.1117 [+0.1107, +0.1125] |

## 17. Master result table (decision tree, raw features)

IID = 5 stratified resplits (mean ± SD; duplication-contaminated where noted). Unseen = same IID models scored on test rows whose exact vector is absent from train. Dedup = vector-disjoint group split. Time = earliest 80% → latest 20% (single run). Fair group = the strongest valid entity/time-block group split. Family macro-recall = mean recall over leave-one-family-out runs (original ladder, `ladder_*.json`). Benign FPR on the IID split.

| Dataset | IID F1 | IID MCC | IID PR-AUC | IID benign FPR | Unseen F1 | Dedup F1 | Time F1 | Fair-group F1 | Family macro-recall | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| sensornetguard | 0.9938 ± 0.0085 | 0.9935 ± 0.0089 | 0.9880 ± 0.0163 | 0.0004 ± 0.0004 | 0.994 ± 0.008 | 0.987 ± 0.009 | 0.976 | N/A | N/A | synthetic; no duplicates |
| nsl-kdd | 0.9930 ± 0.0005 | 0.9865 ± 0.0010 | 0.9898 ± 0.0010 | 0.0069 ± 0.0007 | 0.993 ± 0.001 | 0.994 ± 0.000 | N/A (no timestamp) | N/A | 0.378 | benchmark split F1 0.78 (Section 12) |
| unsw-nb15 | 0.9718 ± 0.0009 | 0.9677 ± 0.0010 | 0.9501 ± 0.0015 | 0.0042 ± 0.0001 | 0.860 ± 0.004 | 0.968 ± 0.001 | 0.964 | 0.940 ± 0.054 | 0.809 | raw 2.5M-row files (21% duplicates) |
| farm-flow | 0.9991 ± 0.0000 | 0.9548 ± 0.0015 | 0.9990 ± 0.0000 | 0.0447 ± 0.0021 | 0.995 ± 0.000 | 0.998 ± 0.001 | N/A (single-class test slice) | 0.989 ± 0.010 | 0.769 | 97.9% attack; compare MCC |
| cic-iov-2024 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | n/a | 0.729 ± 0.138 | 0.927 | N/A | 0.331 | † 99.7% duplicate rows; IID is lookup |
| cicids2017 | 0.9973 ± 0.0001 | 0.9966 ± 0.0002 | 0.9951 ± 0.0002 | 0.0008 ± 0.0000 | 0.997 ± 0.000 | 0.989 ± 0.019 | 0.849 | 0.936 (5-min blocks) | 0.425 | ‡ source-IP groups degenerate; 5-min time-block groups used |

## 18. Secondary robustness experiment: class-balancing in training (standard vs ImbalancedDatasetSampler vs class-weighted loss)

Same split, model class and hyper-parameters, seeds and features for every strategy; only the *training* distribution/loss changes, and the test set is never rebalanced. `sampler` = `torchsampler.ImbalancedDatasetSampler` applied to the training indices only (weights 1/class count, len(train) draws with replacement, i.e. ~50/50 training mix); `weighted` = `class_weight='balanced'` on the unmodified training rows. Train rows capped at 300k before sampling. DT and LightGBM (200 trees), raw and derived features, protocols random / vector-disjoint (unseen) / time / group where valid. Δ = strategy − standard, paired per fold and seed. This changes the training sampling distribution; it does not change the dataset distribution and is not a fix for imbalance.

**Headline matrix (random/IID split, decision tree, raw features; mean over 5 resplits)**

| Dataset | Attack share | Standard F1 | Sampler F1 | ΔF1 | Standard MCC | Sampler MCC | ΔMCC | Weighted F1 | ΔF1 (weighted) | Weighted MCC | ΔMCC (weighted) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sensornetguard | 4.9% | 0.9938 | 0.9721 | -0.0217 | 0.9935 | 0.9711 | -0.0224 | 0.9856 | -0.0082 | 0.9850 | -0.0085 |
| nsl-kdd | 48.1% | 0.9930 | 0.9920 | -0.0011 | 0.9865 | 0.9845 | -0.0020 | 0.9929 | -0.0001 | 0.9863 | -0.0003 |
| unsw-nb15 | 12.6% | 0.9645 | 0.9616 | -0.0029 | 0.9593 | 0.9560 | -0.0033 | 0.9631 | -0.0014 | 0.9577 | -0.0016 |
| farm-flow | 97.9% | 0.9988 | 0.9989 | +0.0001 | 0.9450 | 0.9491 | +0.0041 | 0.9988 | -0.0000 | 0.9440 | -0.0009 |
| cic-iov-2024 | 13.1% | 1.0000 | 1.0000 | -0.0000 | 1.0000 | 1.0000 | -0.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| cicids2017 | 19.7% | 0.9960 | 0.9955 | -0.0005 | 0.9951 | 0.9944 | -0.0007 | 0.9960 | -0.0001 | 0.9950 | -0.0001 |

**Attack recall vs benign false-alarm rate** (what the sampler trades): decision tree, raw features. Each cell: recall / FPR / precision.

| Dataset | Protocol | Standard | Sampler | Weighted | ΔF1 sampler [fold SD] | ΔMCC sampler | ΔFPR sampler | ΔFPR weighted |
|---|---|---|---|---|---|---|---|---|
| sensornetguard | random | 0.996 / 0.0004 / 0.992 | 0.992 / 0.0025 / 0.954 | 0.990 / 0.0009 / 0.982 | -0.0217 [0.0251] | -0.0224 | +0.0021 | +0.0005 |
| sensornetguard | dedup_unseen | 0.993 / 0.0011 / 0.980 | 0.985 / 0.0019 / 0.962 | 0.971 / 0.0009 / 0.982 | -0.0130 [0.0314] | -0.0136 | +0.0009 | -0.0002 |
| sensornetguard | time | 0.971 / 0.0011 / 0.981 | 0.981 / 0.0016 / 0.971 | 0.971 / 0.0021 / 0.962 | +0.0002 [n/a] | +0.0002 | +0.0005 | +0.0011 |
| nsl-kdd | random | 0.993 / 0.0069 / 0.993 | 0.992 / 0.0076 / 0.992 | 0.993 / 0.0072 / 0.992 | -0.0011 [0.0006] | -0.0020 | +0.0007 | +0.0003 |
| nsl-kdd | dedup_unseen | 0.995 / 0.0064 / 0.993 | 0.993 / 0.0072 / 0.992 | 0.995 / 0.0063 / 0.993 | -0.0012 [0.0006] | -0.0022 | +0.0008 | -0.0001 |
| unsw-nb15 | random | 0.964 / 0.0051 / 0.965 | 0.974 / 0.0075 / 0.949 | 0.961 / 0.0051 / 0.965 | -0.0029 [0.0008] | -0.0033 | +0.0024 | -0.0001 |
| unsw-nb15 | dedup_unseen | 0.963 / 0.0054 / 0.963 | 0.973 / 0.0075 / 0.950 | 0.960 / 0.0052 / 0.964 | -0.0022 [0.0008] | -0.0025 | +0.0022 | -0.0001 |
| unsw-nb15 | time | 0.965 / 0.0100 / 0.959 | 0.973 / 0.0136 / 0.946 | 0.960 / 0.0091 / 0.962 | -0.0031 [n/a] | -0.0038 | +0.0036 | -0.0009 |
| unsw-nb15 | group(entity) | 0.925 / 0.0068 / 0.913 | 0.945 / 0.0089 / 0.891 | 0.917 / 0.0065 / 0.915 | -0.0022 [0.0013] | -0.0021 | +0.0022 | -0.0003 |
| farm-flow | random | 0.999 / 0.0529 / 0.999 | 0.999 / 0.0327 / 0.999 | 0.999 / 0.0598 / 0.999 | +0.0001 [0.0000] | +0.0041 | -0.0201 | +0.0069 |
| farm-flow | dedup_unseen | 0.999 / 0.0494 / 0.999 | 0.998 / 0.0329 / 0.999 | 0.999 / 0.0563 / 0.999 | -0.0001 [0.0002] | -0.0014 | -0.0165 | +0.0069 |
| farm-flow | group(entity) | 0.988 / 0.3941 / 0.977 | 0.970 / 0.3421 / 0.980 | 0.981 / 0.4097 / 0.976 | -0.0079 [0.0150] | -0.0537 | -0.0520 | +0.0156 |
| cic-iov-2024 | random | 1.000 / 0.0000 / 1.000 | 1.000 / 0.0000 / 1.000 | 1.000 / 0.0000 / 1.000 | -0.0000 [0.0000] | -0.0000 | +0.0000 | +0.0000 |
| cic-iov-2024 | dedup_unseen | 0.677 / 0.0064 / 0.943 | 0.732 / 0.0074 / 0.921 | 0.732 / 0.0053 / 0.938 | +0.0272 [0.0472] | +0.0226 | +0.0010 | -0.0012 |
| cic-iov-2024 | time | 0.865 / 0.0000 / 1.000 | 0.729 / 0.0000 / 1.000 | 0.865 / 0.0000 / 1.000 | -0.0839 [n/a] | -0.0834 | +0.0000 | +0.0000 |
| cicids2017 | random | 0.996 / 0.0010 / 0.996 | 0.997 / 0.0016 / 0.994 | 0.996 / 0.0010 / 0.996 | -0.0005 [0.0003] | -0.0007 | +0.0005 | -0.0000 |
| cicids2017 | dedup_unseen | 0.997 / 0.0011 / 0.995 | 0.997 / 0.0015 / 0.994 | 0.967 / 0.0011 / 0.995 | -0.0005 [0.0001] | -0.0006 | +0.0004 | -0.0000 |
| cicids2017 | time | 0.766 / 0.0005 / 0.999 | 0.759 / 0.0006 / 0.999 | 0.542 / 0.0003 / 0.999 | -0.0048 [n/a] | -0.0059 | +0.0001 | -0.0002 |
| cicids2017 | group(5min time-block) | 0.893 / 0.0007 / 0.997 | 0.787 / 0.0051 / 0.957 | 0.775 / 0.0051 / 0.952 | -0.0948 [0.1605] | -0.0931 | +0.0044 | +0.0044 |

**All configurations (F1 / MCC / PR-AUC; mean over folds)**

| Dataset | Protocol | Features | Model | Standard | Sampler | Weighted |
|---|---|---|---|---|---|---|
| sensornetguard | random | raw | DT | 0.9938 / 0.9935 / 0.9880 | 0.9721 / 0.9711 / 0.9465 | 0.9856 / 0.9850 / 0.9722 |
| sensornetguard | random | raw | LGBM | 0.9949 / 0.9946 / 1.0000 | 0.9858 / 0.9852 / 0.9998 | 0.9939 / 0.9936 / 0.9999 |
| sensornetguard | random | derived | DT | 0.9804 / 0.9794 / 0.9622 | 0.9697 / 0.9683 / 0.9415 | 0.9692 / 0.9677 / 0.9408 |
| sensornetguard | random | derived | LGBM | 0.9835 / 0.9827 / 0.9990 | 0.9805 / 0.9796 / 0.9981 | 0.9816 / 0.9807 / 0.9991 |
| sensornetguard | dedup_unseen | raw | DT | 0.9864 / 0.9858 / 0.9735 | 0.9734 / 0.9722 / 0.9488 | 0.9765 / 0.9754 / 0.9551 |
| sensornetguard | dedup_unseen | raw | LGBM | 0.9981 / 0.9981 / 0.9999 | 0.9689 / 0.9676 / 0.9985 | 0.9889 / 0.9885 / 0.9994 |
| sensornetguard | dedup_unseen | derived | DT | 0.9788 / 0.9778 / 0.9592 | 0.9790 / 0.9780 / 0.9595 | 0.9823 / 0.9816 / 0.9667 |
| sensornetguard | dedup_unseen | derived | LGBM | 0.9910 / 0.9907 / 0.9996 | 0.9876 / 0.9870 / 0.9992 | 0.9910 / 0.9907 / 0.9993 |
| sensornetguard | time | raw | DT | 0.9758 / 0.9745 / 0.9538 | 0.9761 / 0.9748 / 0.9537 | 0.9665 / 0.9647 / 0.9357 |
| sensornetguard | time | raw | LGBM | 0.9952 / 0.9949 / 1.0000 | 0.9904 / 0.9899 / 0.9991 | 0.9952 / 0.9949 / 1.0000 |
| sensornetguard | time | derived | DT | 0.9806 / 0.9796 / 0.9631 | 0.9423 / 0.9391 / 0.8909 | 0.9519 / 0.9493 / 0.9087 |
| sensornetguard | time | derived | LGBM | 0.9856 / 0.9849 / 0.9993 | 0.9662 / 0.9644 / 0.9975 | 0.9856 / 0.9849 / 0.9995 |
| nsl-kdd | random | raw | DT | 0.9930 / 0.9865 / 0.9898 | 0.9920 / 0.9845 / 0.9883 | 0.9929 / 0.9863 / 0.9895 |
| nsl-kdd | random | raw | LGBM | 0.9957 / 0.9918 / 0.9999 | 0.9951 / 0.9907 / 0.9999 | 0.9957 / 0.9917 / 0.9999 |
| nsl-kdd | random | derived | DT | 0.9721 / 0.9460 / 0.9638 | 0.9715 / 0.9447 / 0.9627 | 0.9721 / 0.9460 / 0.9637 |
| nsl-kdd | random | derived | LGBM | 0.9720 / 0.9456 / 0.9696 | 0.9716 / 0.9449 / 0.9693 | 0.9719 / 0.9455 / 0.9696 |
| nsl-kdd | dedup_unseen | raw | DT | 0.9939 / 0.9884 / 0.9912 | 0.9928 / 0.9862 / 0.9894 | 0.9940 / 0.9886 / 0.9913 |
| nsl-kdd | dedup_unseen | raw | LGBM | 0.9961 / 0.9926 / 0.9999 | 0.9956 / 0.9917 / 0.9999 | 0.9963 / 0.9928 / 0.9999 |
| nsl-kdd | dedup_unseen | derived | DT | 0.9722 / 0.9463 / 0.9645 | 0.9716 / 0.9454 / 0.9639 | 0.9722 / 0.9465 / 0.9647 |
| nsl-kdd | dedup_unseen | derived | LGBM | 0.9719 / 0.9458 / 0.9694 | 0.9715 / 0.9450 / 0.9693 | 0.9719 / 0.9458 / 0.9693 |
| unsw-nb15 | random | raw | DT | 0.9645 / 0.9593 / 0.9353 | 0.9616 / 0.9560 / 0.9284 | 0.9631 / 0.9577 / 0.9330 |
| unsw-nb15 | random | raw | LGBM | 0.9725 / 0.9685 / 0.9980 | 0.9636 / 0.9587 / 0.9977 | 0.9669 / 0.9623 / 0.9979 |
| unsw-nb15 | random | derived | DT | 0.9582 / 0.9522 / 0.9325 | 0.9546 / 0.9482 / 0.9276 | 0.9560 / 0.9496 / 0.9322 |
| unsw-nb15 | random | derived | LGBM | 0.9649 / 0.9598 / 0.9967 | 0.9549 / 0.9490 / 0.9963 | 0.9554 / 0.9496 / 0.9966 |
| unsw-nb15 | dedup_unseen | raw | DT | 0.9633 / 0.9579 / 0.9330 | 0.9611 / 0.9554 / 0.9276 | 0.9622 / 0.9567 / 0.9313 |
| unsw-nb15 | dedup_unseen | raw | LGBM | 0.9715 / 0.9674 / 0.9979 | 0.9628 / 0.9578 / 0.9976 | 0.9660 / 0.9612 / 0.9978 |
| unsw-nb15 | dedup_unseen | derived | DT | 0.9570 / 0.9508 / 0.9305 | 0.9545 / 0.9480 / 0.9274 | 0.9549 / 0.9483 / 0.9300 |
| unsw-nb15 | dedup_unseen | derived | LGBM | 0.9644 / 0.9592 / 0.9965 | 0.9546 / 0.9486 / 0.9962 | 0.9548 / 0.9488 / 0.9964 |
| unsw-nb15 | time | raw | DT | 0.9622 / 0.9530 / 0.9329 | 0.9591 / 0.9492 / 0.9254 | 0.9611 / 0.9517 / 0.9319 |
| unsw-nb15 | time | raw | LGBM | 0.9700 / 0.9627 / 0.9978 | 0.9613 / 0.9523 / 0.9976 | 0.9638 / 0.9553 / 0.9977 |
| unsw-nb15 | time | derived | DT | 0.9581 / 0.9480 / 0.9339 | 0.9562 / 0.9455 / 0.9318 | 0.9570 / 0.9465 / 0.9351 |
| unsw-nb15 | time | derived | LGBM | 0.9652 / 0.9568 / 0.9968 | 0.9570 / 0.9472 / 0.9966 | 0.9566 / 0.9467 / 0.9967 |
| unsw-nb15 | group(entity) | raw | DT | 0.9189 / 0.9127 / 0.8528 | 0.9167 / 0.9107 / 0.8483 | 0.9157 / 0.9092 / 0.8476 |
| unsw-nb15 | group(entity) | raw | LGBM | 0.9341 / 0.9291 / 0.9858 | 0.9220 / 0.9183 / 0.9845 | 0.9249 / 0.9206 / 0.9853 |
| unsw-nb15 | group(entity) | derived | DT | 0.8995 / 0.8915 / 0.8323 | 0.9031 / 0.8966 / 0.8380 | 0.8989 / 0.8909 / 0.8333 |
| unsw-nb15 | group(entity) | derived | LGBM | 0.9203 / 0.9142 / 0.9759 | 0.9056 / 0.9019 / 0.9717 | 0.9075 / 0.9036 / 0.9746 |
| farm-flow | random | raw | DT | 0.9988 / 0.9450 / 0.9988 | 0.9989 / 0.9491 / 0.9993 | 0.9988 / 0.9440 / 0.9987 |
| farm-flow | random | raw | LGBM | 0.9992 / 0.9619 / 1.0000 | 0.9991 / 0.9601 / 1.0000 | 0.9992 / 0.9632 / 1.0000 |
| farm-flow | random | derived | DT | 0.9984 / 0.9239 / 0.9985 | 0.9985 / 0.9326 / 0.9992 | 0.9984 / 0.9229 / 0.9985 |
| farm-flow | random | derived | LGBM | 0.9987 / 0.9392 / 1.0000 | 0.9987 / 0.9415 / 1.0000 | 0.9987 / 0.9418 / 1.0000 |
| farm-flow | dedup_unseen | raw | DT | 0.9987 / 0.9498 / 0.9987 | 0.9986 / 0.9483 / 0.9990 | 0.9986 / 0.9451 / 0.9985 |
| farm-flow | dedup_unseen | raw | LGBM | 0.9991 / 0.9636 / 1.0000 | 0.9990 / 0.9620 / 1.0000 | 0.9991 / 0.9644 / 1.0000 |
| farm-flow | dedup_unseen | derived | DT | 0.9979 / 0.9183 / 0.9981 | 0.9980 / 0.9276 / 0.9990 | 0.9978 / 0.9173 / 0.9981 |
| farm-flow | dedup_unseen | derived | LGBM | 0.9982 / 0.9347 / 1.0000 | 0.9983 / 0.9386 / 1.0000 | 0.9983 / 0.9377 / 1.0000 |
| farm-flow | group(entity) | raw | DT | 0.9825 / 0.6292 / 0.9769 | 0.9746 / 0.5756 / 0.9794 | 0.9782 / 0.5739 / 0.9758 |
| farm-flow | group(entity) | raw | LGBM | 0.9875 / 0.7266 / 0.9997 | 0.9883 / 0.7454 / 0.9996 | 0.9854 / 0.6837 / 0.9996 |
| farm-flow | group(entity) | derived | DT | 0.9780 / 0.6201 / 0.9825 | 0.9878 / 0.7621 / 0.9902 | 0.9853 / 0.7049 / 0.9834 |
| farm-flow | group(entity) | derived | LGBM | 0.9873 / 0.7471 / 0.9997 | 0.9902 / 0.8158 / 0.9998 | 0.9909 / 0.8262 / 0.9998 |
| cic-iov-2024 | random | raw | DT | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| cic-iov-2024 | random | raw | LGBM | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 |
| cic-iov-2024 | random | derived | DT | 0.9914 / 0.9901 / 0.9995 | 0.9913 / 0.9901 / 0.9995 | 0.9913 / 0.9901 / 0.9995 |
| cic-iov-2024 | random | derived | LGBM | 0.9914 / 0.9901 / 0.9995 | 0.9913 / 0.9901 / 0.9995 | 0.9913 / 0.9901 / 0.9995 |
| cic-iov-2024 | dedup_unseen | raw | DT | 0.7861 / 0.7780 / 0.6789 | 0.8133 / 0.8006 / 0.7139 | 0.8185 / 0.8089 / 0.7233 |
| cic-iov-2024 | dedup_unseen | raw | LGBM | 0.8269 / 0.8184 / 0.9606 | 0.7835 / 0.7777 / 0.9625 | 0.8269 / 0.8184 / 0.9605 |
| cic-iov-2024 | dedup_unseen | derived | DT | 0.4694 / 0.4940 / 0.3477 | 0.5085 / 0.4518 / 0.3918 | 0.5226 / 0.4979 / 0.3619 |
| cic-iov-2024 | dedup_unseen | derived | LGBM | 0.7183 / 0.7226 / 0.8920 | 0.6856 / 0.6820 / 0.8808 | 0.6905 / 0.6892 / 0.8252 |
| cic-iov-2024 | time | raw | DT | 0.9274 / 0.9205 / 0.9848 | 0.8435 / 0.8371 / 0.8652 | 0.9274 / 0.9205 / 0.9848 |
| cic-iov-2024 | time | raw | LGBM | 0.8435 / 0.8371 / 1.0000 | 0.8435 / 0.8371 / 0.9848 | 0.9274 / 0.9205 / 0.9848 |
| cic-iov-2024 | time | derived | DT | 0.8294 / 0.8175 / 0.7649 | 0.8294 / 0.8176 / 0.7649 | 0.8295 / 0.8176 / 0.7649 |
| cic-iov-2024 | time | derived | LGBM | 0.8295 / 0.8176 / 0.8404 | 0.8294 / 0.8176 / 0.8181 | 0.8295 / 0.8176 / 0.8218 |
| cicids2017 | random | raw | DT | 0.9960 / 0.9951 / 0.9929 | 0.9955 / 0.9944 / 0.9916 | 0.9960 / 0.9950 / 0.9928 |
| cicids2017 | random | raw | LGBM | 0.9975 / 0.9969 / 0.9998 | 0.9974 / 0.9968 / 0.9998 | 0.9976 / 0.9970 / 0.9998 |
| cicids2017 | random | derived | DT | 0.9589 / 0.9489 / 0.9756 | 0.9524 / 0.9410 / 0.9696 | 0.9552 / 0.9444 / 0.9763 |
| cicids2017 | random | derived | LGBM | 0.9619 / 0.9525 / 0.9944 | 0.9563 / 0.9458 / 0.9943 | 0.9569 / 0.9466 / 0.9944 |
| cicids2017 | dedup_unseen | raw | DT | 0.9960 / 0.9951 / 0.9927 | 0.9955 / 0.9944 / 0.9916 | 0.9806 / 0.9765 / 0.9693 |
| cicids2017 | dedup_unseen | raw | LGBM | 0.9971 / 0.9964 / 0.9998 | 0.9969 / 0.9962 / 0.9998 | 0.9971 / 0.9964 / 0.9998 |
| cicids2017 | dedup_unseen | derived | DT | 0.9463 / 0.9339 / 0.9715 | 0.9486 / 0.9365 / 0.9656 | 0.9523 / 0.9410 / 0.9717 |
| cicids2017 | dedup_unseen | derived | LGBM | 0.9488 / 0.9371 / 0.9942 | 0.9546 / 0.9438 / 0.9941 | 0.9558 / 0.9454 / 0.9942 |
| cicids2017 | time | raw | DT | 0.8672 / 0.8078 / 0.8644 | 0.8623 / 0.8018 / 0.8600 | 0.7024 / 0.6360 / 0.7353 |
| cicids2017 | time | raw | LGBM | 0.8798 / 0.8236 / 0.9761 | 0.8813 / 0.8255 / 0.9949 | 0.8776 / 0.8208 / 0.9903 |
| cicids2017 | time | derived | DT | 0.6642 / 0.5882 / 0.8123 | 0.6856 / 0.5979 / 0.7185 | 0.6676 / 0.5867 / 0.7104 |
| cicids2017 | time | derived | LGBM | 0.6694 / 0.5970 / 0.9433 | 0.7818 / 0.6999 / 0.9437 | 0.6954 / 0.6142 / 0.9445 |
| cicids2017 | group(5min time-block) | raw | DT | 0.9369 / 0.9297 / 0.9110 | 0.8421 / 0.8366 / 0.8100 | 0.8281 / 0.8247 / 0.8003 |
| cicids2017 | group(5min time-block) | raw | LGBM | 0.9389 / 0.9320 / 0.9658 | 0.9373 / 0.9300 / 0.9634 | 0.9376 / 0.9301 / 0.9541 |
| cicids2017 | group(5min time-block) | derived | DT | 0.8567 / 0.8325 / 0.8387 | 0.8607 / 0.8339 / 0.8412 | 0.8407 / 0.8074 / 0.8343 |
| cicids2017 | group(5min time-block) | derived | LGBM | 0.8570 / 0.8336 / 0.9322 | 0.8649 / 0.8397 / 0.9269 | 0.8492 / 0.8168 / 0.9232 |

**Does the compression conclusion depend on the training strategy?** Derived/raw F1 and MCC retention by strategy (decision tree; mean over folds).

| Dataset | Protocol | Standard F1 ret. | Sampler F1 ret. | Weighted F1 ret. | Standard MCC ret. | Sampler MCC ret. | Weighted MCC ret. |
|---|---|---|---|---|---|---|---|
| sensornetguard | random | 98.6% | 99.8% | 98.3% | 98.6% | 99.7% | 98.2% |
| sensornetguard | dedup_unseen | 99.2% | 100.6% | 100.6% | 99.2% | 100.6% | 100.6% |
| sensornetguard | time | 100.5% | 96.5% | 98.5% | 100.5% | 96.3% | 98.4% |
| nsl-kdd | random | 97.9% | 97.9% | 97.9% | 95.9% | 96.0% | 95.9% |
| nsl-kdd | dedup_unseen | 97.8% | 97.9% | 97.8% | 95.7% | 95.9% | 95.7% |
| unsw-nb15 | random | 99.3% | 99.3% | 99.3% | 99.3% | 99.2% | 99.1% |
| unsw-nb15 | dedup_unseen | 99.3% | 99.3% | 99.2% | 99.3% | 99.2% | 99.1% |
| unsw-nb15 | time | 99.6% | 99.7% | 99.6% | 99.5% | 99.6% | 99.5% |
| unsw-nb15 | group(entity) | 97.9% | 98.5% | 98.2% | 97.7% | 98.5% | 98.0% |
| farm-flow | random | 100.0% | 100.0% | 100.0% | 97.8% | 98.3% | 97.8% |
| farm-flow | dedup_unseen | 99.9% | 99.9% | 99.9% | 96.7% | 97.8% | 97.1% |
| farm-flow | group(entity) | 99.5% | 101.4% | 100.7% | 98.5% | 132.4% | 122.8% |
| cic-iov-2024 | random | 99.1% | 99.1% | 99.1% | 99.0% | 99.0% | 99.0% |
| cic-iov-2024 | dedup_unseen | 59.7% | 62.5% | 63.8% | 63.5% | 56.4% | 61.6% |
| cic-iov-2024 | time | 89.4% | 98.3% | 89.4% | 88.8% | 97.7% | 88.8% |
| cicids2017 | random | 96.3% | 95.7% | 95.9% | 95.4% | 94.6% | 94.9% |
| cicids2017 | dedup_unseen | 95.0% | 95.3% | 97.1% | 93.9% | 94.2% | 96.4% |
| cicids2017 | time | 76.6% | 79.5% | 95.0% | 72.8% | 74.6% | 92.3% |
| cicids2017 | group(5min time-block) | 91.4% | 102.2% | 101.5% | 89.6% | 99.7% | 97.9% |

**Label-shuffled check by training strategy** (random split, decision tree, raw features, 10 permutations of the training labels; test labels real). With the standard strategy F1 follows the test prior; with the sampler the model predicts the attack class about half the time, so F1 no longer tracks the prior while MCC stays ≈ 0 - evidence that the shuffled-label F1 is a prior effect of the evaluation, not leakage.

| Dataset | Test attack prior | Standard F1 / MCC / predicted-attack rate | Sampler F1 / MCC / predicted-attack rate | Weighted F1 / MCC / predicted-attack rate |
|---|---|---|---|---|
| sensornetguard | 0.049 | 0.085 / +0.035 / 0.06 | 0.068 / -0.001 / 0.12 | 0.060 / +0.010 / 0.05 |
| nsl-kdd | 0.481 | 0.496 / +0.017 / 0.49 | 0.496 / +0.014 / 0.50 | 0.496 / +0.017 / 0.49 |
| unsw-nb15 | 0.126 | 0.103 / -0.033 / 0.14 | 0.217 / +0.058 / 0.28 | 0.237 / +0.109 / 0.17 |
| farm-flow | 0.979 | 0.987 / +0.039 / 0.99 | 0.684 / -0.120 / 0.56 | 0.751 / -0.106 / 0.63 |
| cic-iov-2024 | 0.131 | 0.000 / -0.009 / 0.00 | 0.223 / +0.023 / 0.53 | 0.228 / +0.034 / 0.52 |
| cicids2017 | 0.197 | 0.190 / -0.017 / 0.21 | 0.260 / +0.015 / 0.34 | 0.226 / +0.026 / 0.21 |


## Limitations

- Farm-Flow raw monthly CSVs are 97.9% attack vs ~50/50 in the silver split; absolute numbers differ between the two. The `traffic` column (attack-type label) was excluded from features.
- Derived feature sets are hand-built and dataset-specific; the flow-dataset sets use two overlapping derivations, so 'derived' is not minimal.
- Raw UNSW-NB15 (2.5M rows, four files) differs from the 257k train/test partition used for the benchmark rows; ladder numbers are not directly comparable to Section 6's UNSW benchmark.
- Silver preprocessing dropped CAN-ID bits (IoV), IPs/ports/timestamps (IDS2017, UNSW), so identifier-leakage tests are limited to group/time/family splits.
- Time splits are single deterministic runs (no interval); Farm-Flow and IoV time proxies are row order, not timestamps.
- CIC-IDS2017 group split is degenerate (see Section 2). Hyper-parameters for RF/XGBoost/LightGBM were fixed, not tuned. SensorNetGuard appears synthetic.
- Latency was measured on a laptop CPU, single process, with the other experiment process paused (SIGSTOP); values are ~0.02-0.08 µs/row, and lower derived latency on IoV/Farm-Flow/IDS2017 mostly reflects narrower input arrays, not smaller trees.
- The earlier Farm-Flow raw feature set (Sections 1, 3, 4, 6) included the numeric port columns `id.orig_p`/`id.resp_p` (87 features); Section 8 shows removing them and the top-5 features does not change results, while the provenance loaders (Sections 2, 9, 13) exclude them (85 features).
- `ImbalancedDatasetSampler` is a PyTorch sampler; it is applied here to the training indices of tree/boosting models (draws with replacement), so it duplicates minority rows rather than changing a mini-batch stream. Train rows were capped at 300k for the balance ablation.
- Transfer-matrix cells use only the features computable in both datasets (2-10 features); SensorNetGuard is a partial node whose rates/error rate are node-health quantities, so cells involving it are descriptive. Single-split bootstrap CIs reflect test-set sampling only.
- CIC-IDS2017 time-split results are unstable across training-subsample caps (full / 600k / 300k train rows give raw-tree F1 0.849 / 0.754 / 0.867 in Sections 2, 13, 16); the single deterministic time split has no interval that captures this, so its retention and paired differences should be read as indicative only.
## Appendix: results inventory

Every JSON in `experiments/derived/results/` and the report section that uses it:

| File | Section |
|---|---|
| `audit_cic-iov-2024.json` | 1. Leakage and dataset-artefact audit |
| `audit_cicids2017.json` | 1. Leakage and dataset-artefact audit |
| `audit_farm-flow.json` | 1. Leakage and dataset-artefact audit |
| `audit_nsl-kdd.json` | 1. Leakage and dataset-artefact audit |
| `audit_sensornetguard.json` | 1. Leakage and dataset-artefact audit |
| `audit_unsw-nb15.json` | 1. Leakage and dataset-artefact audit |
| `balance_cic-iov-2024.json` | 18. Class-balance training ablation |
| `balance_cicids2017.json` | 18. Class-balance training ablation |
| `balance_farm-flow.json` | 18. Class-balance training ablation |
| `balance_nsl-kdd.json` | 18. Class-balance training ablation |
| `balance_sensornetguard.json` | 18. Class-balance training ablation |
| `balance_unsw-nb15.json` | 18. Class-balance training ablation |
| `baselines_cic-iov-2024.json` | 4. Stronger baselines |
| `baselines_cicids2017.json` | 4. Stronger baselines |
| `baselines_farm-flow.json` | 4. Stronger baselines |
| `baselines_nsl-kdd.json` | 4. Stronger baselines |
| `baselines_sensornetguard.json` | 4. Stronger baselines |
| `baselines_unsw-nb15.json` | 4. Stronger baselines |
| `compression_cic-iov-2024.json` | 3. Compression table |
| `compression_cicids2017.json` | 3. Compression table |
| `compression_farm-flow.json` | 3. Compression table |
| `compression_latency_cic-iov-2024.json` | 14. Model complexity |
| `compression_latency_cicids2017.json` | 14. Model complexity |
| `compression_latency_farm-flow.json` | 14. Model complexity |
| `compression_latency_nsl-kdd.json` | 14. Model complexity |
| `compression_latency_sensornetguard.json` | 14. Model complexity |
| `compression_latency_unsw-nb15.json` | 14. Model complexity |
| `compression_noniid_cic-iov-2024.json` | 13. Compression under non-IID |
| `compression_noniid_cicids2017.json` | 13. Compression under non-IID |
| `compression_noniid_farm-flow.json` | 13. Compression under non-IID |
| `compression_noniid_nsl-kdd.json` | 13. Compression under non-IID |
| `compression_noniid_sensornetguard.json` | 13. Compression under non-IID |
| `compression_noniid_unsw-nb15.json` | 13. Compression under non-IID |
| `compression_nsl-kdd.json` | 3. Compression table |
| `compression_sensornetguard.json` | 3. Compression table |
| `compression_unsw-nb15.json` | 3. Compression table |
| `derived_vs_raw.json` | 7. Derived vs raw single run |
| `farmflow_ablation.json` | 8. Farm-Flow ablation |
| `final_stats_cic-iov-2024.json` | 16. Paired comparisons |
| `final_stats_cicids2017.json` | 16. Paired comparisons |
| `final_stats_farm-flow.json` | 16. Paired comparisons |
| `final_stats_nsl-kdd.json` | 16. Paired comparisons |
| `final_stats_sensornetguard.json` | 16. Paired comparisons |
| `final_stats_unsw-nb15.json` | 16. Paired comparisons |
| `ids2017_groups.json` | 10. IDS2017 group splits |
| `iov_novelty.json` | 11. IoV novelty |
| `ladder2_cic-iov-2024.json` | 9. Duplicate-aware ladder; 17. master table |
| `ladder2_cicids2017.json` | 9. Duplicate-aware ladder; 17. master table |
| `ladder2_farm-flow.json` | 9. Duplicate-aware ladder; 17. master table |
| `ladder2_nsl-kdd.json` | 9. Duplicate-aware ladder; 17. master table |
| `ladder2_sensornetguard.json` | 9. Duplicate-aware ladder; 17. master table |
| `ladder2_unsw-nb15.json` | 9. Duplicate-aware ladder; 17. master table |
| `ladder_cic-iov-2024.json` | 2. Evaluation-protocol ladder; 17. master table (family-holdout) |
| `ladder_cicids2017.json` | 2. Evaluation-protocol ladder; 17. master table (family-holdout) |
| `ladder_farm-flow.json` | 2. Evaluation-protocol ladder; 17. master table (family-holdout) |
| `ladder_nsl-kdd.json` | 2. Evaluation-protocol ladder; 17. master table (family-holdout) |
| `ladder_sensornetguard.json` | 2. Evaluation-protocol ladder; 17. master table (family-holdout) |
| `ladder_unsw-nb15.json` | 2. Evaluation-protocol ladder; 17. master table (family-holdout) |
| `nslkdd_shift.json` | 12. NSL-KDD shift |
| `permutation_farmflow.json` | 8. Permutation checks |
| `permutation_others.json` | 8. Permutation checks |
| `robust_cic-iov-2024.json` | 6. Tuned trees, 5 iterations |
| `robust_cicids2017.json` | 6. Tuned trees, 5 iterations |
| `robust_farm-flow.json` | 6. Tuned trees, 5 iterations |
| `robust_nsl-kdd.json` | 6. Tuned trees, 5 iterations |
| `robust_sensornetguard.json` | 6. Tuned trees, 5 iterations |
| `robust_unsw-nb15.json` | 6. Tuned trees, 5 iterations |
| `shared_analysis.json` | 5. Shared features / LODO |
| `transfer_matrix.json` | 15. Transfer matrix |
| `transfer_null.json` | 15. Transfer permutation null |

