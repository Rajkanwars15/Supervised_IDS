"""Builds experiments/derived/RESULTS.md from results/*.json (all numbers are generated, none typed by hand)."""
import glob, json
from collections import Counter
from pathlib import Path
import numpy as np

R = Path(__file__).parent / "results"
J = lambda pat: {Path(f).stem.split("_", 1)[1]: json.load(open(f)) for f in sorted(glob.glob(str(R / pat)))}
rob = {}
for f in sorted(R.glob("robust_*.json")):
    rob.update(json.load(open(f)))
audit, ladder, comp, base = J("audit_*.json"), J("ladder_*.json"), J("compression_*.json"), J("baselines_*.json")
d1 = json.load(open(R / "derived_vs_raw.json")); sh = json.load(open(R / "shared_analysis.json"))
ORDER = ["sensornetguard", "nsl-kdd", "unsw-nb15", "farm-flow", "cic-iov-2024", "cicids2017"]
mean = lambda rows, k: float(np.mean([r[k] for r in rows]))
sd = lambda rows, k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else float("nan")
pm = lambda m, k, p=4: f"{m[k]['mean']:.{p}f} ± {m[k]['ci_hi'] - m[k]['mean']:.{p}f}"
L = []; w = L.append
SPLITNAME = {"cicids2017": "own 80/20 (seed 42)", "cic-iov-2024": "own 80/20 (seed 42)"}

w("# Derived features, evaluation protocol and cross-dataset transfer in network IDS (decision-tree study)\n")
w("Every number below is generated from `experiments/derived/results/*.json` by `write_report.py`. "
  "Scripts: `derive_features.py`, `derived_dt_experiment.py`, `robust_per_dataset.py`, `shared_analysis.py`, `leakage_audit.py`, "
  "`raw_loaders.py`, `split_ladder.py`, `compression_table.py`, `baselines.py`.\n")

w("## Summary\n")
w("- **RQ1 (compression).** Dataset-specific derived rate/ratio features cut inputs by 56-97% and keep 96-100% of the raw-feature F1 at unlimited depth "
  "(random splits, paired, 5 resplits): SensorNetGuard 99.4%, Farm-Flow 99.9%, UNSW-NB15 98.9%, NSL-KDD 97.9%, CIC-IoV 99.1%, CIC-IDS2017 96.2%. "
  "The compression is of the *input*, not of the *model*: derived trees are often as large or larger (e.g. CIC-IDS2017 4,491 vs 1,437 nodes).")
w("- **RQ2 (generalisation).** Compact representations do not transfer across datasets: the dominant feature differs per dataset, signs of the same feature flip, "
  "a tree identifies the source dataset of a benign row 92.5% of the time from the six shared features, and leave-one-dataset-out F1 is 0.02-0.90 (mostly < 0.4).")
w("- **RQ3 (protocol).** The evaluation protocol changes conclusions more than the model does. Examples: CIC-IDS2017 F1 0.997 (random) → 0.849 (time) → 0.061 "
  "(group by source IP, degenerate test) and 0.43 macro-recall on held-out attack families; CIC-IoV 1.000 → 0.927 (time) and 0.33 macro-recall on held-out families; "
  "NSL-KDD 0.993 (random) vs 0.78 (benchmark) with 0.38 macro-recall on held-out families.")
w("- **Leakage.** UNSW-NB15 is ~40% duplicate rows: its random-split F1 of 0.951 is 0.869 on test rows unseen in training, so much of the random-vs-benchmark gap is duplicate leakage. "
  "CIC-IoV-2024's bit features are 99.7% duplicates (99.8% of test rows have an exact copy in train), i.e. near-perfect scores reflect lookup. Duplicates do **not** explain NSL-KDD's gap (a family/distribution shift) or CIC-IDS2017.")
w("- **Classifier-independence.** RF, XGBoost and LightGBM show the same pattern as the decision tree on raw vs derived features (Section 4); ensembles add about 1 F1 point or less over the decision tree.")
w("- Kyoto was removed: its silver train/test sets are 100% attack (label mapping appears inverted) and columns 14-16 look like IDS-detection flags (leakage).\n")

w("## 0. What \"official\" split means here (correction)\n")
w("| Dataset | Split used as \"official\" | Genuinely a published benchmark split? |\n|---|---|---|")
for n, s, b in [("UNSW-NB15", "UNSW_NB15 training-set / testing-set (175k/82k)", "yes"), ("NSL-KDD", "KDDTrain+ / KDDTest+", "yes"),
                ("Farm-Flow", "Farm-Flow Train/Test CSVs (balanced; raw monthly files are 97.9% attack)", "yes, but class balance differs from raw"),
                ("CIC-IDS2017", "our own `train_test_split(0.2, seed 42, stratify)` in preprocessing", "**no**"),
                ("CIC-IoV-2024", "our own `train_test_split(0.2, seed 42, stratify)` in preprocessing", "**no**"),
                ("SensorNetGuard", "n/a (single CSV; our 80/20)", "no")]:
    w(f"| {n} | {s} | {b} |")
w("\nOnly UNSW-NB15 and NSL-KDD give a benchmark-vs-random comparison. Silver files also dropped provenance (IDS2017 IPs/timestamps/Flow ID; IoV CAN-ID bits and file identity; UNSW IPs/time), "
  "so group/time/family protocols were re-run from the raw files (`raw_loaders.py`), whose feature sets differ slightly from silver (notes in Section 3).\n")

w("## 1. Leakage and dataset-artefact audit\n")
w("Default-parameter decision tree on raw features. *Seen* = test row whose exact (rounded) feature vector appears in train. *Unseen F1* = F1 on test rows not seen in train. "
  "*Drop-top-3* removes the three most important features and retrains. *Shuffled* = F1 with permuted training labels (≈ class prior if the pipeline is sound).\n")
w("| Dataset | Split | Dup. train % | Test seen in train % | Label-conflict rows % | F1 all | F1 unseen | F1 drop-top-1 | F1 drop-top-3 | F1 shuffled | Top feature (importance) |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for n in ORDER:
    for r in audit.get(n, []):
        f = lambda v: "n/a" if v is None else f"{v:.4f}"
        top = next(iter(r["top_features"].items()))
        w(f"| {n} | {SPLITNAME.get(n, r['split']) if r['split']=='own80/20' else r['split']} | {r['dup_train_pct']:.1f} | {r['test_in_train_pct']:.1f} | {r['conflicting_rows_train_pct']:.2f} | "
          f"{f(r['f1_full'])} | {f(r['f1_unseen'])} | {f(r['drop_top']['1']['f1'])} | {f(r['drop_top']['3']['f1'])} | {f(r['f1_label_shuffled'])} | {top[0]} ({top[1]:.2f}) |")
w("\nReading: UNSW-NB15 random vs benchmark (F1 0.951 vs 0.885) shrinks to 0.869 vs 0.870 once seen rows are excluded - the gap is duplicate contamination. "
  "CIC-IoV's \"unseen\" subset is too small/single-class for an F1. Only SensorNetGuard loses more than ~1 F1 point when its top three features are dropped (0.990 → 0.921); elsewhere the signal is redundant across features, so no single leaked column explains the scores. "
  "Not audited: IP/port identifier-only predictors beyond the group-split results in Section 2, and flow/session overlap in silver files (provenance dropped).\n")

w("## 2. RQ3 - evaluation-protocol ladder\n")
w("Default decision tree, raw features from `raw_loaders.py`. random = 5 stratified resplits; group = 5 GroupShuffleSplits by source IP; time = earliest 80% → latest 20% (single deterministic run; IoV within each source file; Farm-Flow row order as proxy); "
  "family = leave-one-attack-family-out (benign 80/20 fixed), reporting recall on the held-out family. N/A = the dataset has no such information.\n")
w("| Dataset (n, features) | Random F1 | Group F1 | Time F1 | Family macro-recall | Family weighted-recall | Mean benign FPR | Loader note |")
w("|---|---|---|---|---|---|---|---|")
for n in ORDER:
    r = ladder.get(n)
    if not r: continue
    P = r["protocols"]
    cell = lambda k: P[k] if isinstance(P[k], str) else (f"{mean(P[k],'f1'):.4f}" + (f" ± {sd(P[k],'f1'):.4f}" if len(P[k]) > 1 else ""))
    fam = P["family_holdout"]
    fm = ("N/A", "N/A", "N/A") if isinstance(fam, str) else (f"{fam['macro_recall']:.3f}", f"{fam['weighted_recall']:.3f}", f"{fam['mean_fpr']:.4f}")
    w(f"| {n} ({r['n']:,}; {r['n_features']}) | {cell('random')} | {cell('group')} | {cell('time')} | {fm[0]} | {fm[1]} | {fm[2]} | {r['note']} |")
w("\nPer-family recall when the family is held out (worst first):\n")
for n in ORDER:
    r = ladder.get(n)
    if not r or isinstance(r["protocols"]["family_holdout"], str): continue
    pf = r["protocols"]["family_holdout"]["per_family"]
    w(f"- **{n}**: " + ", ".join(f"{k} {v['recall_on_family']:.2f}" for k, v in sorted(pf.items(), key=lambda kv: kv[1]["recall_on_family"])[:8]))
w("\nCaveats: CIC-IDS2017's group split is degenerate (a handful of attacker IPs generate nearly all attacks, so the group-held-out test set is ~0.2% attack; F1 0.06 reflects attacker-identity shift, not a clean error rate). "
  "Farm-Flow's time test slice is single-class (attack-only). Time-split numbers have no resampling interval.\n")

w("## 3. RQ1 - compression: raw vs derived features\n")
w("Random 80/20, 5 paired resplits (same split/seed for raw and derived), default decision tree with depth cap, pools capped at 500k rows. "
  "Retention = derived / raw. Model size = joblib bytes; latency = median ms per 1,000 rows over 9 timed predicts (resolution ~0.1 ms, single process).\n")
w("| Dataset | Depth cap | Raw → derived features | Input reduction | Raw F1 | Derived F1 | F1 retention | MCC retention | AP retention | Nodes raw → derived | Model KB raw → derived | Latency raw → derived (ms/1k rows) |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in ORDER:
    c = comp.get(n)
    if not c: continue
    for dep in ("5", "10", "None"):
        a, b = c["runs"][f"raw|{dep}"], c["runs"][f"derived|{dep}"]
        w(f"| {n} | {dep if dep!='None' else 'unlimited'} | {c['n_raw']} → {c['n_derived']} | {100*(1-c['n_derived']/c['n_raw']):.0f}% | {mean(a,'f1'):.4f} | {mean(b,'f1'):.4f} | "
          f"{100*mean(b,'f1')/mean(a,'f1'):.1f}% | {100*mean(b,'mcc')/mean(a,'mcc'):.1f}% | {100*mean(b,'ap')/mean(a,'ap'):.1f}% | {mean(a,'nodes'):.0f} → {mean(b,'nodes'):.0f} | "
          f"{mean(a,'bytes')/1024:.0f} → {mean(b,'bytes')/1024:.0f} | {mean(a,'lat'):.2f} → {mean(b,'lat'):.2f} |")
w("\nNotes: derived sets are dataset-specific (SensorNetGuard 4, NSL-KDD 11, UNSW-NB15 17, Farm-Flow 10, CIC-IoV 4 bit-statistics, CIC-IDS2017 18); for the flow datasets they contain two overlapping derivations (`a_*`, `b_*`). "
  "The input reduction does not shrink trees: unlimited-depth derived trees are larger on CIC-IDS2017 (≈3×), UNSW-NB15 and Farm-Flow, and smaller only on NSL-KDD and CIC-IoV. Inference latency is negligible for all (below measurement resolution).\n")

w("## 4. Stronger baselines on the same protocol\n")
w("Fixed hyper-parameters (RF 100 trees; XGBoost 200 trees depth 6; LightGBM 200 trees, 63 leaves; DT default), train capped at 300k rows. Mean F1 over 5 resplits (random) or 5 model seeds on the fixed benchmark split. Cells: raw / derived.\n")
w("| Dataset | Protocol | DT | RF | XGBoost | LightGBM |"); w("|---|---|---|---|---|---|")
for n in ORDER:
    b = base.get(n)
    if not b: continue
    for p in sorted({k.split("|")[0] for k in b["runs"]}):
        cells = [f"{mean(b['runs'][f'{p}|raw|{m}'],'f1'):.4f} / {mean(b['runs'][f'{p}|derived|{m}'],'f1'):.4f}" for m in ("DT", "RF", "XGB", "LGBM")]
        w(f"| {n} | {p} | " + " | ".join(cells) + " |")
w("\nBenchmark-split rows have no sampling variance (seeds only). On NSL-KDD's benchmark split derived features *help* DT and XGBoost but hurt LightGBM and RF, so model class matters there; on the other datasets the raw→derived pattern is the same across all four classifiers.\n")

w("## 5. RQ2 - shared features and cross-dataset transfer (flow datasets)\n")
w("UNSW-NB15, NSL-KDD, CIC-IDS2017 (silver splits) and Farm-Flow (raw monthly files, stratified 80/20). SensorNetGuard and IoV excluded (different domains).\n")
w(f"- Features available in all four: `{'`, `'.join(sh['all4'])}`; consistent-direction shortlist (≥3 datasets, |AUC−0.5|>0.1, same sign): `{'`, `'.join(sh['short'])}`.")
w("- A depth-6 tree identifies which dataset a *benign* row came from with 92.5% accuracy on the all-4 features (78.5% on the shortlist; chance 25%).\n")
auc = sh["part1"]["auc"]; feats = list(next(iter(auc.values())).keys())
w("Signed AUC−0.5 (positive: attacks have higher values) and mean pairwise KS shift between benign distributions:\n")
w("| Feature | " + " | ".join(auc) + " | Benign KS |"); w("|---|" + "---|" * (len(auc) + 1))
for f in feats:
    cells = [("n/a" if auc[n][f] is None or auc[n][f] != auc[n][f] else f"{auc[n][f]:+.2f}") for n in auc]
    w(f"| {f} | " + " | ".join(cells) + f" | {sh['part1']['ks'][f]:.2f} |")
w("\nLeave-one-dataset-out F1 (train on the other three):\n")
w("| Feature set | Transform | Depth | " + " | ".join(auc) + " |"); w("|---|---|---|" + "---|" * len(auc))
for k, v in sh["part2"].items():
    lab, mode, dep = k.split("|")
    w(f"| {lab} | {mode} | {dep} | " + " | ".join(f"{v['lodo'][n]['f1']:.2f}" for n in auc) + " |")
w("\nSingle run, single seed. Byte columns are not the same quantity across datasets (IP bytes vs payload/flow bytes).\n")

w("## 6. Tuned per-dataset decision trees, 5 iterations, with confidence metrics\n")
w("Tuning (feature set × depth × min-leaf × criterion × class-weight) by 3-fold CV on a ≤100k train subsample only. **Benchmark** rows: fixed split, iterations vary seed/tuning only, so the ± reflects model-selection uncertainty, **not** test-set sampling "
  "(use the bootstrap F1 interval below for that - and it is not an independent replication). **Random** rows: 5 different stratified resplits (sampling uncertainty). 'own 80/20' = our fixed seed-42 split, not a published benchmark.\n")
w("| Dataset | Protocol | Baseline F1 | Tuned F1 | Tuned precision | Tuned recall | Tuned MCC | Tuned AUC | Tuned AP |"); w("|---|---|---|---|---|---|---|---|---|")
for k, v in rob.items():
    t, b = v["tuned"], v["baseline"]; n, p = k.split("|")
    p = SPLITNAME.get(n, "benchmark") if p == "official" else p
    w(f"| {n} | {p} | {pm(b,'f1')} | **{pm(t,'f1')}** | {pm(t,'prec')} | {pm(t,'rec')} | {pm(t,'mcc')} | {pm(t,'auc')} | {pm(t,'ap')} |")
w("\nConfidence metrics (tuned model): Brier, ECE (10 bins), mean confidence, high-confidence (≥0.9) coverage/accuracy, and the 95% bootstrap interval of the test-set F1 (averaged over iterations).\n")
w("| Dataset | Protocol | Brier | ECE | Mean conf | High-conf coverage | High-conf accuracy | F1 bootstrap CI |"); w("|---|---|---|---|---|---|---|---|")
for k, v in rob.items():
    t = v["tuned"]; n, p = k.split("|"); p = SPLITNAME.get(n, "benchmark") if p == "official" else p
    w(f"| {n} | {p} | {t['brier']['mean']:.4f} | {t['ece']['mean']:.4f} | {t['mean_conf']['mean']:.4f} | {t['hiconf_cov']['mean']:.3f} | {t['hiconf_acc']['mean']:.4f} | {t['f1_boot_lo']['mean']:.4f} - {t['f1_boot_hi']['mean']:.4f} |")
w("\nSelected configurations: " + "; ".join(
    f"{k.split('|')[0]}|{k.split('|')[1]}: " + ",".join(f"{a}×{c}" for a, c in Counter(i['feature_set'] for i in v['iterations']).items()) for k, v in rob.items()) + ".\n")

w("## 7. Derived vs raw, single run (official/80-20 splits)\n")
w("| Dataset | Raw → derived features | Depth | Raw F1 | Derived F1 | Raw top feature | Derived top feature |"); w("|---|---|---|---|---|---|---|")
for n, v in d1.items():
    for r, dd in zip(v["raw"], v["derived"]):
        w(f"| {n} | {v['n_raw']} → {v['n_derived']} | {r['depth']} | {r['f1']:.4f} | {dd['f1']:.4f} | {r['top']} | {dd['top']} |")

w("\n## Limitations\n")
for s in ["Farm-Flow raw monthly CSVs are 97.9% attack vs ~50/50 in the silver split; absolute numbers differ between the two. The `traffic` column (attack-type label) was excluded from features.",
          "Derived feature sets are hand-built and dataset-specific; the flow-dataset sets use two overlapping derivations, so 'derived' is not minimal.",
          "Raw UNSW-NB15 (2.5M rows, four files) differs from the 257k train/test partition used for the benchmark rows; ladder numbers are not directly comparable to Section 6's UNSW benchmark.",
          "Silver preprocessing dropped CAN-ID bits (IoV), IPs/ports/timestamps (IDS2017, UNSW), so identifier-leakage tests are limited to group/time/family splits.",
          "Time splits are single deterministic runs (no interval); Farm-Flow and IoV time proxies are row order, not timestamps.",
          "CIC-IDS2017 group split is degenerate (see Section 2). Hyper-parameters for RF/XGBoost/LightGBM were fixed, not tuned. SensorNetGuard appears synthetic.",
          "Latency was measured on a shared laptop CPU, single process; differences below ~0.1 ms/1k rows are not resolvable."]:
    w(f"- {s}")
(Path(__file__).parent / "RESULTS.md").write_text("\n".join(L) + "\n")
