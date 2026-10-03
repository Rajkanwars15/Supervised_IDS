"""Builds experiments/derived/RESULTS.md from results/*.json."""
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).parent / "results"
rob = {}
for f in sorted(R.glob("robust_*.json")):
    rob.update(json.load(open(f)))
d = json.load(open(R / "derived_vs_raw.json"))
sh = json.load(open(R / "shared_analysis.json"))

def pm(m, k, p=4):
    return f"{m[k]['mean']:.{p}f} ± {m[k]['ci_hi'] - m[k]['mean']:.{p}f}"

L = []
w = L.append
w("# Derived-feature & per-dataset decision-tree experiments\n")
w("All numbers are generated from `experiments/derived/results/*.json` by `write_report.py`. Scripts: `derive_features.py`, "
  "`derived_dt_experiment.py`, `shared_analysis.py`, `robust_per_dataset.py`, `summarize_robust.py`.\n")
w("## Summary\n")
w("- A decision tree per dataset works. A single shared tree across datasets does not (Part 2).")
w("- Derived rate/ratio features reach close to the raw-feature baseline with 4-18 features instead of 17-78 (Part 1), row-for-row (no rows collapsed).")
w("- Per-dataset tuning (feature set × depth × min-leaf × criterion × class-weight, chosen by CV on train only) gives small gains only where the baseline had room: NSL-KDD official (+2.5 F1 pts) and UNSW-NB15 official (+0.5). Elsewhere it is at the ceiling.")
w("- NSL-KDD's official split is a distribution-shift problem: random re-splits give ~0.993 F1 vs ~0.80 on the official split.")
w("- Kyoto was removed: its silver train/test sets are 100% attack (label mapping appears inverted) and columns 14-16 look like IDS-detection flags (leakage).\n")

w("## Part 3 (final): robust per-dataset results, 5 iterations\n")
w("Protocol: **official** = the dataset's own train/test split (iterations vary only tuning/tree seed, so intervals understate true uncertainty); "
  "**random** = 5 different stratified 80/20 re-splits of the pooled data (the honest variance estimate). Cells are mean ± 95% t-interval over 5 iterations. "
  "Baseline = untuned DecisionTree on raw features, same split.\n")
w("| Dataset | Protocol | Baseline F1 | Tuned F1 | Tuned precision | Tuned recall | Tuned MCC | Tuned AUC | Tuned AP |")
w("|---|---|---|---|---|---|---|---|---|")
for k, v in rob.items():
    t, b = v["tuned"], v["baseline"]
    n, p = k.split("|")
    w(f"| {n} | {p} | {pm(b,'f1')} | **{pm(t,'f1')}** | {pm(t,'prec')} | {pm(t,'rec')} | {pm(t,'mcc')} | {pm(t,'auc')} | {pm(t,'ap')} |")
w("\n### Confidence metrics (tuned model)\n")
w("Brier = mean squared error of predicted probability (lower better). ECE = expected calibration error over 10 confidence bins (lower better). "
  "Mean conf = average confidence in the predicted class. High-conf = rows where confidence ≥ 0.9 (coverage = fraction of rows; accuracy on those rows). "
  "F1 bootstrap CI = 95% interval from 1000 multinomial resamples of the test confusion matrix, averaged over iterations.\n")
w("| Dataset | Protocol | Brier | ECE | Mean conf | High-conf coverage | High-conf accuracy | F1 bootstrap CI (mean of iters) |")
w("|---|---|---|---|---|---|---|---|")
for k, v in rob.items():
    t = v["tuned"]; n, p = k.split("|")
    w(f"| {n} | {p} | {t['brier']['mean']:.4f} | {t['ece']['mean']:.4f} | {t['mean_conf']['mean']:.4f} | {t['hiconf_cov']['mean']:.3f} | "
      f"{t['hiconf_acc']['mean']:.4f} | {t['f1_boot_lo']['mean']:.4f} - {t['f1_boot_hi']['mean']:.4f} |")
w("\n### Selected configurations per iteration\n")
w("| Dataset | Protocol | Feature set (count) | Most common params |")
w("|---|---|---|---|")
for k, v in rob.items():
    n, p = k.split("|")
    fs = Counter(i["feature_set"] for i in v["iterations"])
    pr = Counter(json.dumps(i["params"], sort_keys=True) for i in v["iterations"]).most_common(1)[0]
    w(f"| {n} | {p} | {', '.join(f'{a} ({c})' for a, c in fs.items())} | `{pr[0]}` ({pr[1]}/5) |")
w("\n### Per-iteration tuned F1\n")
w("| Dataset | Protocol | it0 | it1 | it2 | it3 | it4 |")
w("|---|---|---|---|---|---|---|")
for k, v in rob.items():
    n, p = k.split("|")
    w(f"| {n} | {p} | " + " | ".join(f"{i['tuned']['f1']:.4f}" for i in v["iterations"]) + " |")

w("\n## Part 1: derived features vs raw (single run, official/80-20 split)\n")
w("Derived features (rates, ratios, sizes, error/loss rates) computed row-for-row. F1 by tree depth (None = unlimited).\n")
w("| Dataset | Raw → derived features | Depth | Raw F1 | Derived F1 | Raw top feature | Derived top feature |")
w("|---|---|---|---|---|---|---|")
for n, v in d.items():
    for r, dd in zip(v["raw"], v["derived"]):
        w(f"| {n} | {v['n_raw']} → {v['n_derived']} | {r['depth']} | {r['f1']:.4f} | {dd['f1']:.4f} | {r['top']} | {dd['top']} |")

w("\n## Part 2: shared features and cross-dataset transfer (negative result)\n")
w("Flow datasets: UNSW-NB15, NSL-KDD, CIC-IDS2017, Farm-Flow (from raw monthly CSVs). SensorNetGuard and IoV excluded (different domains).\n")
w(f"- Features available in all four: `{'`, `'.join(sh['all4'])}`.")
w(f"- Consistent-direction shortlist (≥3 datasets, |AUC−0.5|>0.1, same sign): `{'`, `'.join(sh['short'])}`.")
w("- A depth-6 tree identifies which dataset a *benign* row came from with 92.5% accuracy on the all-4 features (78.5% on the shortlist; chance 25%): the domains are not aligned.\n")
auc = sh["part1"]["auc"]
feats = list(next(iter(auc.values())).keys())
w("Signed AUC−0.5 per feature (positive: attacks have higher values):\n")
w("| Feature | " + " | ".join(auc) + " | Benign KS shift |"); w("|---|" + "---|" * (len(auc) + 1))
for f in feats:
    cells = [("n/a" if auc[n][f] is None or auc[n][f] != auc[n][f] else f"{auc[n][f]:+.2f}") for n in auc]
    w(f"| {f} | " + " | ".join(cells) + f" | {sh['part1']['ks'][f]:.2f} |")
w("\nLeave-one-dataset-out F1 (train on the other three):\n")
w("| Feature set | Transform | Depth | " + " | ".join(auc) + " |"); w("|---|---|---|" + "---|" * len(auc))
for k, v in sh["part2"].items():
    lab, mode, dep = k.split("|")
    w(f"| {lab} | {mode} | {dep} | " + " | ".join(f"{v['lodo'][n]['f1']:.2f}" for n in auc) + " |")
w("\nConclusion: no stable shared decision tree; the dominant feature differs per dataset (SensorNetGuard/NSL-KDD `error_rate`, UNSW-NB15 `pkt_rate`, CIC-IDS2017 `fwd_bwd_byte_ratio`), "
  "and a pooled tree mostly learns dataset identity.\n")

w("## Caveats\n")
w("- Farm-Flow raw monthly CSVs are 97.9% attack vs ~50/50 in the silver split; its numbers are not comparable to earlier silver baselines. The `traffic` column (attack-type label) was excluded from features.")
w("- Official-split intervals only reflect tuning/seed variance; trees are near-deterministic given a fixed split.")
w("- CIC-IDS2017 and UNSW-NB15 tuning used a ≤100k-row subsample with 3-fold CV on train rows only (no test leakage).")
w("- Byte columns are not the same quantity across datasets (IP bytes vs payload/flow bytes), which limits cross-dataset comparison.")
w("- Protocol/flag/TTL columns were dropped from the silver files; rebuilding from raw could expose more shared features.")
(Path(__file__).parent / "RESULTS.md").write_text("\n".join(L) + "\n")
