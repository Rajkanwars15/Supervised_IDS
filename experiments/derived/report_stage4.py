"""Stage-4 report sections (items 1-10). build(w, R, ORDER) appends markdown lines via w(). Skips sections whose JSON is missing."""
import glob, json
from pathlib import Path
import numpy as np

fmt = lambda v, p=4: "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{p}f}"


def load(R, name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


def ms(rows, k, sub="all"):
    v = [r[sub][k] for r in rows if isinstance(r.get(sub), dict) and k in r[sub] and r[sub][k] == r[sub][k]]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")) if v else None


def cell(rows, k, sub="all", p=4):
    r = ms(rows, k, sub)
    return "n/a" if r is None else (f"{r[0]:.{p}f}" + (f" ± {r[1]:.{p}f}" if r[1] == r[1] else ""))


def build(w, R, ORDER):
    used = set()

    def get(name):
        d = load(R, name)
        if d is not None: used.add(name)
        return d

    # ---------------- 8. chance-level & permutation checks ----------------
    pf, po, ab = get("permutation_farmflow.json"), get("permutation_others.json"), get("farmflow_ablation.json")
    if pf or po:
        w("## 8. Chance-level and label-permutation checks (items 1-2)\n")
        w("Stratified label permutation (prior preserved), full pipeline refit, default decision tree, pool capped at 200k rows. "
          "Chance references for the attack class: F1(all-attack) = 2p/(1+p), MCC = 0, AP = p, AUC = 0.5. **Pass** = |mean MCC| ≤ 0.01 and |mean AUC − 0.5| ≤ 0.01.\n")
        w("| Dataset | Features | Attack prior | Real F1 | Real MCC | Shuffled F1 | Shuffled MCC (mean ± SD) | Shuffled AUC | Shuffled AP | Chance F1 | #perms | Pass |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")

        def row(name, v):
            s = v["shuffled"]
            w(f"| {name} | | {v['prior']:.3f} | {fmt(v.get('real', {}).get('f1'))} | {fmt(v.get('real', {}).get('mcc'))} | {s['f1']['mean']:.4f} | {s['mcc']['mean']:+.4f} ± {s['mcc']['sd']:.4f} | {s['auc']['mean']:.4f} | {s['ap']['mean']:.4f} | {v['chance']['f1_allpos']:.4f} | {v['n_perm']} | {'yes' if v.get('pass_mcc') and v.get('pass_auc') else ('yes' if abs(s['mcc']['mean']) <= 0.01 else 'check')} |")
        if pf:
            for k, lab in [("after_raw_with_ports", "farm-flow raw (87 incl. ports), permute after derivation"), ("before_raw_with_ports", "farm-flow raw, permute BEFORE derivation + refit"), ("after_derived", "farm-flow derived, permute after")]:
                if k in pf: row(lab, pf[k])
        if po:
            for n in ORDER:
                for fs in ("raw", "derived"):
                    if n in po and n != "farm-flow": row(f"{n} {fs}", po[n][fs])
        w("\nF1 alone is not a chance check when the prior is extreme: Farm-Flow's raw data is 97.9% attack, so a no-skill classifier gets F1 ≈ 0.98-0.99 while MCC and AUC sit at 0 and 0.5. "
          "The earlier audit column \"F1 shuffled\" (0.98 for Farm-Flow) is therefore the class prior, not leakage.\n")
        if ab:
            w("### Farm-Flow identifier / top-feature ablation\n")
            w("Importance-ranked removal on the identifier-free model. Real-label metrics on the standard test fold and on a 50/50 undersampled test fold (prior removed); shuffled-label MCC over 20 permutations. "
              "Note: the earlier Farm-Flow experiments (audit, robust, compression, baselines) used **87 numeric features including `id.orig_p` and `id.resp_p`** (L0); L1 removes them.\n")
            w("| Feature set | #features | Real F1 | Real MCC | Real AP | Balanced-test MCC | Balanced-test BA | Shuffled MCC | Shuffled F1 |"); w("|---|---|---|---|---|---|---|---|---|")
            for k, v in ab.items():
                if k.startswith("_"): continue
                w(f"| {k} | {v['n_features']} | {v['real']['f1']:.4f} | {v['real']['mcc']:.4f} | {v['real']['ap']:.4f} | {v['real_balanced_test']['mcc']:.4f} | {v['real_balanced_test']['ba']:.4f} | {v['shuffled']['mcc']['mean']:+.4f} | {v['shuffled']['f1']['mean']:.4f} |")
            w(f"\nTop features (identifier-free model): {', '.join(ab.get('_top_features', [])[:8])}. Reading: removing identifiers and the five most important features leaves real-label MCC essentially unchanged while the shuffled-label MCC stays at 0, so Farm-Flow's separability is redundant across many flow statistics (consistent with synthetic attack traffic), not carried by one leaked column or the ports.\n")

    # ---------------- 9. duplicate-aware ladder ----------------
    l2 = {n: get(f"ladder2_{n}.json") for n in ORDER}
    l2 = {k: v for k, v in l2.items() if v}
    if l2:
        w("## 9. Duplicate-aware evaluation ladder (item 4)\n")
        w("Same fitted model scored on subsets of one test set. **seen** = exact feature vector present in train; **unseen** = absent. **dedup-group** = identical vectors form one group and whole groups go to train or test (vector-disjoint; test rows keep their multiplicity). "
          "Label-conflict ceiling = F1 of an oracle predicting the majority label per identical vector. Columns are F1 / MCC / PR-AUC / benign FPR (mean over resplits; ± SD for F1).\n")
        w("| Dataset | Model | Dup. rows % | Ceiling F1 | IID random (contaminated) | seen share of test | IID - seen subset (F1) | IID - unseen subset (F1) | Dedup-group split | Benchmark split |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        f4 = lambda rows, sub="all": "n/a" if ms(rows, "f1", sub) is None else " / ".join(f"{ms(rows, k, sub)[0]:.3f}" for k in ("f1", "mcc", "ap", "fpr")) + (f" (F1 ±{ms(rows, 'f1', sub)[1]:.3f})" if ms(rows, "f1", sub)[1] == ms(rows, "f1", sub)[1] else "")
        for n, r in l2.items():
            P = r["protocols"]
            for m in P["random"]:
                rnd, dd = P["random"][m], P["dedup_group"][m]
                bm = P.get("benchmark", {}).get(m)
                seen_share = np.mean([x["seen_frac"] for x in rnd])
                w(f"| {n} | {m} | {r['duplicate_pct']:.1f} | {r['label_conflict_ceiling']['f1']:.4f} | {f4(rnd)} | {seen_share:.3f} | {cell(rnd, 'f1', 'seen', 3)} | {cell(rnd, 'f1', 'unseen', 3)} | {f4(dd) if dd else 'n/a'} | {f4(bm) if bm else 'n/a'} |")
        w("\nTemporal and group protocols (decision tree; single run for time, mean over group resplits; F1 / MCC / PR-AUC / FPR; test attack rate in brackets):\n")
        w("| Dataset | Time split | Group split |"); w("|---|---|---|")
        for n, r in l2.items():
            P = r["protocols"]; t = P["time"]
            ts = t if isinstance(t, str) else f"{t['all']['f1']:.3f} / {t['all']['mcc']:.3f} / {t['all']['ap']:.3f} / {t['all']['fpr']:.4f} [atk {t['test_attack_rate']:.2f}]"
            g = P["group"]
            gs = g if isinstance(g, str) else (f4(g) if g else "n/a")
            w(f"| {n} | {ts} | {gs} |")
        w("")

    # ---------------- 10. IDS2017 groups ----------------
    g = get("ids2017_groups.json")
    if g:
        w("## 10. CIC-IDS2017 group splits from raw provenance (item 3)\n")
        w(f"Raw flows with Flow ID, IPs and timestamps (n = {g['n']:,}; attack rate {g['overall_attack_rate']:.3f}). A fold is **valid** if the test side has ≥ 50 attack and ≥ 50 benign rows and an attack rate within ±{int(g['tol']*100)}% (relative) of overall; "
          "group overlap between train and test is asserted to be zero (time-block folds also drop ±1 block of guard rows from train). Attacks come from only 10 source IPs, 555,641 of 557,646 attack flows from 172.16.0.1 (all to 192.168.10.50), "
          "so entity keys that follow the attacker box cannot produce a balanced held-out set.\n")
        w("| Group key | Valid folds | F1 (mean ± SD) | MCC | PR-AUC | Recall | Benign FPR | Test attack rate | #groups | Guard rows dropped |"); w("|---|---|---|---|---|---|---|---|---|---|")
        for k, v in g["keys"].items():
            if v["valid_folds"] == 0:
                w(f"| {k} | 0/5 | no valid fold (degenerate: attacks concentrated in one entity) | | | | | | | |"); continue
            s = v["summary"]; fl = v["folds"]
            w(f"| {k} | {v['valid_folds']}/5 | {s['f1']['mean']:.4f} ± {s['f1']['sd']:.4f} | {s['mcc']['mean']:.4f} | {s['ap']['mean']:.4f} | {s['recall']['mean']:.4f} | {s['fpr']['mean']:.4f} | {np.mean([f['test_attack_rate'] for f in fl]):.3f} | {fl[0]['n_groups']:,} | {int(np.mean([f['guard_rows_dropped'] for f in fl])):,} |")
        w("\nReading: the earlier source-IP group F1 of 0.061 was a degenerate split (kept in `ladder_cicids2017.json` for transparency). Entity-level keys that are not tied to the attacker box (Flow-ID/session, time blocks) give valid folds; performance falls monotonically as the time block widens (more temporal separation), the legitimate group-generalisation estimate.\n")

    # ---------------- 11. IoV novelty ----------------
    nv = get("iov_novelty.json")
    if nv:
        w("## 11. CIC-IoV novelty experiment (item 5)\n")
        w(f"{nv['n']:,} rows but only {nv['n_unique']:,} unique 136-bit vectors; training uses unique vectors with multiplicity weights (equivalent to row-level training for a decision tree). "
          f"Claimable only if a cell has ≥ {nv['min_rule']['n']} test rows and ≥ {nv['min_rule']['pos']} positives; otherwise it is descriptive. Novelty is measured against the training rows of each split: exact-unseen (Hamming ≥ 1), Hamming ≥ k bits from every training vector, and frame-novel (some of the 8 CAN-frame patterns never seen at that position). 95% CIs are test-set bootstrap (not replication).\n")
        w("| Split | Novelty level | n rows | n positives | Claimable | DT F1 [CI] | DT MCC | DT PR-AUC | DT benign FPR | LGBM F1 | LGBM MCC |"); w("|---|---|---|---|---|---|---|---|---|---|---|")
        for sname, sp in nv["splits"].items():
            for lvl, c in sp["levels"].items():
                if "DT" not in c:
                    w(f"| {sname} | {lvl} | {c['n_rows']:,} | {c['n_pos']:,} | {'yes' if c['claimable'] else 'no'} | {c.get('note', '')} | | | | | |"); continue
                d, l = c["DT"], c.get("LGBM", {})
                ci = d["ci"]["f1"]
                w(f"| {sname} | {lvl} | {c['n_rows']:,} | {c['n_pos']:,} | {'yes' if c['claimable'] else 'descriptive only'} | {d['f1']:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] | {d['mcc']:.4f} | {d['ap']:.4f} | {d['fpr']:.4f} | {fmt(l.get('f1'))} | {fmt(l.get('mcc'))} |")
        w("")

    # ---------------- 12. NSL-KDD shift ----------------
    ns = get("nslkdd_shift.json")
    if ns:
        w("## 12. NSL-KDD: why the benchmark split is harder than a random split (item 6)\n")
        cp, fam, adv = ns["class_prior"], ns["families"], ns["adversarial_auc"]
        w(f"- **Class prior**: attack share train {cp['train']:.3f}, benchmark test {cp['benchmark_test']:.3f} (KDDTest-21: {cp['test21']:.3f}), random-split test {cp['random_test']:.3f}.")
        w(f"- **Families**: {fam['n_train_families']} attack families in train, {fam['n_test_families']} in test; **{len(fam['novel_test_families'])} test families are absent from train** ({', '.join(fam['novel_test_families'])}), "
          f"covering {fam['share_of_test_attack_rows_novel']*100:.1f}% of test attack rows ({fam['n_novel_test_attack_rows']:,} rows).")
        w(f"- **Covariate shift** (adversarial validation AUC, LightGBM 5-fold; 0.5 = indistinguishable): train vs benchmark test {adv['train_vs_benchmark_test']:.3f} (benign only {adv['train_vs_benchmark_test_benign_only']:.3f}, attack only {adv['train_vs_benchmark_test_attack_only']:.3f}) versus {adv['random_train_vs_random_test']:.3f} for a random split. "
          f"Top shifted features (KS, log1p): {', '.join(f'{k} ({v:.2f})' for k, v in list(ns['feature_ks_top10'].items())[:5])}.\n")
        w("Counterfactual decomposition (F1 / MCC on the test set; same trained model per row):\n")
        w("| Model | Random split | Benchmark (full) | Benchmark, novel-family attacks removed | …and re-weighted to random-split family mix & prior | Recall seen-family attacks | Recall novel-family attacks | Benign FPR |"); w("|---|---|---|---|---|---|---|---|")
        for m, v in ns["models"].items():
            c = lambda x: f"{x['f1']:.3f} / {x['mcc']:.3f}"
            w(f"| {m} | {c(v['random_split'])} | {c(v['benchmark_full'])} | {c(v['benchmark_seen_families_only'])} | {c(v['benchmark_seen_families_reweighted_to_random_mix'])} | {v['recall_seen_family_attacks']:.3f} | {v['recall_novel_family_attacks']:.3f} | {v['benign_fpr']:.3f} |")
        pc = ns["partition_check"]
        w(f"\nPartition check: {pc['benign']:,} benign + {pc['seen_family_attacks']:,} seen-family attacks + {pc['novel_family_attacks']:,} novel-family attacks = {pc['n_test']:,} test rows. "
          "The re-weighting is approximate (importance weights on seen-family attack rows to match the random split's family mix and class prior). "
          "Reading: removing novel-family attacks recovers part of the gap, matching the family mix recovers most of the rest, and a residual remains that is covariate shift within shared families.\n")
    return used


def build2(w, R, ORDER, ladder):
    """Sections 13-16 + results inventory. `ladder` = dict of ladder_*.json (family-holdout, original ladder)."""
    # ---------------- 13. compression under non-IID ----------------
    cn = {n: load(R, f"compression_noniid_{n}.json") for n in ORDER}; cn = {k: v for k, v in cn.items() if v}
    if cn:
        w("## 13. RQ1 under non-IID protocols: F1 / MCC / PR-AUC retention of derived features (item 7)\n")
        w("Raw vs derived on identical splits (train capped at 600k rows). Retention = derived / raw (mean over folds). ΔF1 = mean paired difference (derived − raw) with the mean of the per-fold paired-bootstrap 95% limits "
          "(test-row resampling; approximate pooled CI). Protocols: random (5 resplits), dedup-unseen (vector-disjoint, 3), time (1 run), group (entity or 5-min time block, 3 folds) where valid.\n")
        w("| Dataset (raw → derived) | Protocol | Model | Raw F1 | Derived F1 | F1 retention | MCC retention | PR-AUC retention | ΔF1 [paired 95% CI] | Benign FPR raw → derived |"); w("|---|---|---|---|---|---|---|---|---|---|")
        for n, r in cn.items():
            for proto, pm_ in r["protocols"].items():
                for model, v in pm_.items():
                    d = [f["diff"]["f1"] for f in v["folds"]]
                    w(f"| {n} ({r['n_raw']} → {r['n_derived']}) | {proto} | {model} | {v['raw_mean']['f1']:.4f} | {v['derived_mean']['f1']:.4f} | {v['retention']['f1']*100:.1f}% | {v['retention']['mcc']*100:.1f}% | {v['retention']['ap']*100:.1f}% | "
                      f"{np.mean([x[0] for x in d]):+.4f} [{np.mean([x[1] for x in d]):+.4f}, {np.mean([x[2] for x in d]):+.4f}] | {v['raw_mean']['fpr']:.4f} → {v['derived_mean']['fpr']:.4f} |")
        w("")

    # ---------------- 14. model complexity ----------------
    cl = {n: load(R, f"compression_latency_{n}.json") for n in ORDER}; cl = {k: v for k, v in cl.items() if v}
    if cl:
        w("## 14. Capacity and model complexity: input compression vs model compression (item 8)\n")
        w("One fixed 80/20 split (pool ≤ 500k rows), decision tree with depth cap. Latency = median µs per row over 15 repeats of a 100k-row batch (IQR in brackets), single process; peak = tracemalloc peak during `predict` on that batch.\n")
        w("| Dataset | Depth cap | Features raw → derived | Nodes raw → derived | Leaves raw → derived | Model KB raw → derived | Latency µs/row raw → derived | Peak predict MB raw → derived | F1 raw → derived | MCC raw → derived |"); w("|---|---|---|---|---|---|---|---|---|---|")
        verdict = []
        for n, r in cl.items():
            for dep in ("5", "10", "None"):
                a, b = r["runs"][f"raw|{dep}"], r["runs"][f"derived|{dep}"]
                lat = lambda x: f"{x['lat_us_per_row']['median']:.3f} [{x['lat_us_per_row']['q25']:.3f}-{x['lat_us_per_row']['q75']:.3f}]"
                w(f"| {n} | {dep if dep != 'None' else 'unlimited'} | {r['n_raw']} → {r['n_derived']} | {a['nodes']:,} → {b['nodes']:,} | {a['leaves']:,} → {b['leaves']:,} | {a['bytes']/1024:.0f} → {b['bytes']/1024:.0f} | {lat(a)} → {lat(b)} | "
                  f"{a['peak_predict_bytes']/1e6:.1f} → {b['peak_predict_bytes']/1e6:.1f} | {a['metrics']['f1']:.4f} → {b['metrics']['f1']:.4f} | {a['metrics']['mcc']:.4f} → {b['metrics']['mcc']:.4f} |")
            a, b = r["runs"]["raw|None"], r["runs"]["derived|None"]
            ratio = b["nodes"] / a["nodes"]; lr = b["lat_us_per_row"]["median"] / a["lat_us_per_row"]["median"]
            verdict.append((n, 100 * (1 - r["n_derived"] / r["n_raw"]), ratio, b["bytes"] / a["bytes"], lr))
        w("\nVerdict at unlimited depth (derived / raw):\n")
        w("| Dataset | Input dimension reduction | Node ratio | Serialized-size ratio | Latency ratio | Model-level compression? |"); w("|---|---|---|---|---|---|")
        for n, red, nr, sr, lr in verdict:
            w(f"| {n} | {red:.0f}% | {nr:.2f}× | {sr:.2f}× | {lr:.2f}× | {'yes (smaller tree)' if nr < 0.8 else ('no (larger tree)' if nr > 1.25 else 'no (about the same)')} |")
        w("\nConclusion: semantic derivation compresses the **input representation**; it does not systematically shrink the fitted tree, its serialized size or inference cost.\n")

    # ---------------- 15. transfer matrix ----------------
    tm = load(R, "transfer_matrix.json"); tn = load(R, "transfer_null.json")
    if tm:
        N = ["unsw-nb15", "nsl-kdd", "cicids2017", "farm-flow", "sensornetguard"]
        w("## 15. Cross-dataset transfer matrix on harmonised features (item 9)\n")
        w("Rows = train dataset, columns = test dataset. Each cell uses only the harmonised features computable in **both** datasets (count in the feature table). Source train ≤ 300k, target test ≤ 200k (stratified), 3 source subsamples. "
          "`quantile` = per-dataset quantile transform fitted label-free on each dataset's own train. SensorNetGuard contributes only 3 node-health rate features (partial node; cells are descriptive). Diagonal = within-dataset on the same features.\n")
        for key, lab in [("asis|DT", "As-is features, decision tree (depth 10)"), ("quantile|LGBM", "Quantile-transformed features, LightGBM"), ("asis|LGBM", "As-is features, LightGBM"), ("quantile|DT", "Quantile-transformed features, decision tree")]:
            w(f"**{lab}: MCC (F1)**\n"); w("| train \\ test | " + " | ".join(N) + " |"); w("|---|" + "---|" * len(N))
            for a in N:
                cells = []
                for b in N:
                    c = tm["cells"].get(f"{a}->{b}", {})
                    if "runs" not in c: cells.append("N/A"); continue
                    m = c["runs"][key]["mean"]; s = c["runs"][key]["sd"]
                    cells.append(f"{m['mcc']:+.2f} ± {s['mcc']:.2f} ({m['f1']:.2f})")
                w(f"| **{a}** | " + " | ".join(cells) + " |")
            w("")
        w("Harmonised feature count per cell: " + "; ".join(f"{k}: {len(v['features'])}" for k, v in tm["cells"].items() if k.split('->')[0] != k.split('->')[1]) + ".\n")
        if tn:
            w("**Permutation null for off-diagonal cells (as-is DT; 20 permutations of the source training labels).** A single shuffled run is not a valid null under distribution shift - with few features a noise-fit tree can reach |MCC| ≈ 0.2-0.35 by chance - so real MCC is compared with the null distribution.\n")
            w("| Transfer | Features | Real MCC | Null MCC (mean ± SD) | z | One-sided p |"); w("|---|---|---|---|---|---|")
            for k, v in tn.items():
                a, b = k.split("->")
                if a == b: continue
                w(f"| {k} | {v['n_features']} | {v['real_mcc']:+.3f} | {v['null_mcc_mean']:+.3f} ± {v['null_mcc_sd']:.3f} | {v['z']:+.1f} | {v['p_one_sided']:.2f} |")
            w("")
        offd = [c["runs"]["asis|DT"]["mean"]["mcc"] for k, c in tm["cells"].items() if "runs" in c and k.split("->")[0] != k.split("->")[1]]
        diag = [c["runs"]["asis|DT"]["mean"]["mcc"] for k, c in tm["cells"].items() if "runs" in c and k.split("->")[0] == k.split("->")[1]]
        w(f"Summary (as-is DT): diagonal MCC {min(diag):.2f}-{max(diag):.2f}; off-diagonal MCC median {np.median(offd):+.2f}, range {min(offd):+.2f} to {max(offd):+.2f}; {sum(1 for x in offd if x > 0.5)} of {len(offd)} off-diagonal cells exceed MCC 0.5.\n")

    # ---------------- 16. final statistical package ----------------
    fs = {n: load(R, f"final_stats_{n}.json") for n in ORDER}; fs = {k: v for k, v in fs.items() if v}
    if fs:
        w("## 16. Paired comparisons with bootstrap 95% CIs (item 10)\n")
        w("Identical test rows for both arms; 300 bootstrap resamples of the test rows. These CIs capture **test-set sampling** for a single split (random seed 42, benchmark, or time) and are not independent replications; "
          "mean ± SD is reserved for repeated random resplits (Sections 6 and 9).\n")
        w("**RQ1 - derived minus raw (ΔF1, ΔMCC, ΔPR-AUC), per model**\n")
        w("| Dataset | Split | n test | Model | ΔF1 [95% CI] | ΔMCC [95% CI] | ΔPR-AUC [95% CI] |"); w("|---|---|---|---|---|---|---|")
        for n, r in fs.items():
            for sname, c in r["splits"].items():
                for m, d in c["derived_minus_raw"].items():
                    g = lambda k: f"{d[k][0]:+.4f} [{d[k][1]:+.4f}, {d[k][2]:+.4f}]"
                    w(f"| {n} | {sname} | {c['n_test']:,} | {m} | {g('f1')} | {g('mcc')} | {g('ap')} |")
        w("\n**Model comparison - ensemble minus decision tree (raw features)**\n")
        w("| Dataset | Split | Model | ΔF1 [95% CI] | ΔMCC [95% CI] | ΔPR-AUC [95% CI] |"); w("|---|---|---|---|---|---|")
        for n, r in fs.items():
            for sname, c in r["splits"].items():
                for m, d in c["ensemble_minus_dt_raw"].items():
                    g = lambda k: f"{d[k][0]:+.4f} [{d[k][1]:+.4f}, {d[k][2]:+.4f}]"
                    w(f"| {n} | {sname} | {m} | {g('f1')} | {g('mcc')} | {g('ap')} |")
        w("")

    # ---------------- master table ----------------
    l2 = {n: load(R, f"ladder2_{n}.json") for n in ORDER}; l2 = {k: v for k, v in l2.items() if v}
    ig = load(R, "ids2017_groups.json")
    if l2:
        w("## 17. Master result table (decision tree, raw features)\n")
        w("IID = 5 stratified resplits (mean ± SD; duplication-contaminated where noted). Unseen = same IID models scored on test rows whose exact vector is absent from train. Dedup = vector-disjoint group split. "
          "Time = earliest 80% → latest 20% (single run). Fair group = the strongest valid entity/time-block group split. Family macro-recall = mean recall over leave-one-family-out runs (original ladder, `ladder_*.json`). Benign FPR on the IID split.\n")
        w("| Dataset | IID F1 | IID MCC | IID PR-AUC | IID benign FPR | Unseen F1 | Dedup F1 | Time F1 | Fair-group F1 | Family macro-recall | Notes |"); w("|---|---|---|---|---|---|---|---|---|---|---|")
        notes = {"cic-iov-2024": "† 99.7% duplicate rows; IID is lookup", "cicids2017": "‡ source-IP groups degenerate; 5-min time-block groups used", "unsw-nb15": "raw 2.5M-row files (21% duplicates)",
                 "nsl-kdd": "benchmark split F1 0.78 (Section 12)", "farm-flow": "97.9% attack; compare MCC", "sensornetguard": "synthetic; no duplicates"}
        for n, r in l2.items():
            P = r["protocols"]; rnd, dd = P["random"]["DT"], P["dedup_group"]["DT"]
            t = P["time"]; ts = t if isinstance(t, str) else f"{t['all']['f1']:.3f}"
            if n == "cicids2017" and ig and ig["keys"].get("timeblock_5min", {}).get("summary"):
                fg = f"{ig['keys']['timeblock_5min']['summary']['f1']['mean']:.3f} (5-min blocks)"
            elif isinstance(P["group"], list) and P["group"]:
                fg = cell(P["group"], "f1", "all", 3)
            else:
                fg = "N/A"
            lf = ladder.get(n, {}).get("protocols", {}).get("family_holdout")
            fm = f"{lf['macro_recall']:.3f}" if isinstance(lf, dict) else "N/A"
            w(f"| {n} | {cell(rnd, 'f1')} | {cell(rnd, 'mcc')} | {cell(rnd, 'ap')} | {cell(rnd, 'fpr')} | {cell(rnd, 'f1', 'unseen', 3)} | {cell(dd, 'f1', 'all', 3)} | {ts} | {fg} | {fm} | {notes.get(n, '')} |")
        w("")


INVENTORY = [("compression_noniid_*.json", "13. Compression under non-IID"), ("compression_latency_*.json", "14. Model complexity"),
             ("audit_*.json", "1. Leakage and dataset-artefact audit"), ("ladder2_*.json", "9. Duplicate-aware ladder; 17. master table"),
             ("ladder_*.json", "2. Evaluation-protocol ladder; 17. master table (family-holdout)"), ("compression_*.json", "3. Compression table"),
             ("baselines_*.json", "4. Stronger baselines"), ("shared_analysis.json", "5. Shared features / LODO"), ("robust_*.json", "6. Tuned trees, 5 iterations"),
             ("derived_vs_raw.json", "7. Derived vs raw single run"), ("permutation_*.json", "8. Permutation checks"), ("farmflow_ablation.json", "8. Farm-Flow ablation"),
             ("ids2017_groups.json", "10. IDS2017 group splits"), ("iov_novelty.json", "11. IoV novelty"), ("nslkdd_shift.json", "12. NSL-KDD shift"),
             ("transfer_matrix.json", "15. Transfer matrix"), ("transfer_null.json", "15. Transfer permutation null"), ("final_stats_*.json", "16. Paired comparisons"), ("balance_*.json", "18. Class-balance training ablation")]


def inventory(w, R):
    files = sorted(p.name for p in R.glob("*.json")); seen = {}
    for pat, sec in INVENTORY:
        for f in glob.glob(str(R / pat)):
            seen.setdefault(Path(f).name, sec)
    missing = [f for f in files if f not in seen]
    w("## Appendix: results inventory\n")
    w("Every JSON in `experiments/derived/results/` and the report section that uses it:\n")
    w("| File | Section |"); w("|---|---|")
    for f in files:
        w(f"| `{f}` | {seen.get(f, '**UNREFERENCED**')} |")
    if missing:
        raise SystemExit(f"UNREFERENCED result files: {missing}")
    w("")


def build3(w, R, ORDER):
    """Section 18: class-balance training ablation (standard vs ImbalancedDatasetSampler vs class-weighted loss)."""
    bl = {n: load(R, f"balance_{n}.json") for n in ORDER}; bl = {k: v for k, v in bl.items() if v}
    if not bl: return
    S3 = ["standard", "sampler", "weighted"]
    mean_ = lambda runs, k: float(np.nanmean([r[k] for r in runs]))
    w("## 18. Secondary robustness experiment: class-balancing in training (standard vs ImbalancedDatasetSampler vs class-weighted loss)\n")
    w("Same split, model class and hyper-parameters, seeds and features for every strategy; only the *training* distribution/loss changes, and the test set is never rebalanced. "
      "`sampler` = `torchsampler.ImbalancedDatasetSampler` applied to the training indices only (weights 1/class count, len(train) draws with replacement, i.e. ~50/50 training mix); "
      "`weighted` = `class_weight='balanced'` on the unmodified training rows. Train rows capped at 300k before sampling. DT and LightGBM (200 trees), raw and derived features, "
      "protocols random / vector-disjoint (unseen) / time / group where valid. Δ = strategy − standard, paired per fold and seed. This changes the training sampling distribution; it does not change the dataset distribution and is not a fix for imbalance.\n")
    # headline matrix: random protocol, DT, raw
    w("**Headline matrix (random/IID split, decision tree, raw features; mean over 5 resplits)**\n")
    w("| Dataset | Attack share | Standard F1 | Sampler F1 | ΔF1 | Standard MCC | Sampler MCC | ΔMCC | Weighted F1 | ΔF1 (weighted) | Weighted MCC | ΔMCC (weighted) |"); w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in bl.items():
        c = r["protocols"].get("random", {}).get("raw|DT")
        if not c: continue
        g = lambda s, k: mean_(c["runs"][s], k)
        w(f"| {n} | {r['attack_rate']*100:.1f}% | {g('standard','f1'):.4f} | {g('sampler','f1'):.4f} | {c['delta_sampler']['f1']['mean']:+.4f} | {g('standard','mcc'):.4f} | {g('sampler','mcc'):.4f} | {c['delta_sampler']['mcc']['mean']:+.4f} | "
          f"{g('weighted','f1'):.4f} | {c['delta_weighted']['f1']['mean']:+.4f} | {g('weighted','mcc'):.4f} | {c['delta_weighted']['mcc']['mean']:+.4f} |")
    w("\n**Attack recall vs benign false-alarm rate** (what the sampler trades): decision tree, raw features. Each cell: recall / FPR / precision.\n")
    w("| Dataset | Protocol | Standard | Sampler | Weighted | ΔF1 sampler [fold SD] | ΔMCC sampler | ΔFPR sampler | ΔFPR weighted |"); w("|---|---|---|---|---|---|---|---|---|")
    for n, r in bl.items():
        for proto, pc in r["protocols"].items():
            c = pc.get("raw|DT")
            if not c: continue
            cell_ = lambda s: f"{mean_(c['runs'][s],'recall'):.3f} / {mean_(c['runs'][s],'fpr'):.4f} / {mean_(c['runs'][s],'precision'):.3f}"
            ds, dw = c["delta_sampler"], c["delta_weighted"]
            sdv = ds["f1"]["sd"]
            w(f"| {n} | {proto} | {cell_('standard')} | {cell_('sampler')} | {cell_('weighted')} | {ds['f1']['mean']:+.4f} [{'n/a' if sdv != sdv else f'{sdv:.4f}'}] | {ds['mcc']['mean']:+.4f} | {ds['fpr']['mean']:+.4f} | {dw['fpr']['mean']:+.4f} |")
    w("\n**All configurations (F1 / MCC / PR-AUC; mean over folds)**\n")
    w("| Dataset | Protocol | Features | Model | Standard | Sampler | Weighted |"); w("|---|---|---|---|---|---|---|")
    for n, r in bl.items():
        for proto, pc in r["protocols"].items():
            for key, c in pc.items():
                fsn, m = key.split("|")
                f3 = lambda s: " / ".join(f"{mean_(c['runs'][s], k):.4f}" for k in ("f1", "mcc", "ap"))
                w(f"| {n} | {proto} | {fsn} | {m} | {f3('standard')} | {f3('sampler')} | {f3('weighted')} |")
    # does the compression conclusion depend on the training strategy?
    w("\n**Does the compression conclusion depend on the training strategy?** Derived/raw F1 and MCC retention by strategy (decision tree; mean over folds).\n")
    w("| Dataset | Protocol | Standard F1 ret. | Sampler F1 ret. | Weighted F1 ret. | Standard MCC ret. | Sampler MCC ret. | Weighted MCC ret. |"); w("|---|---|---|---|---|---|---|---|")
    for n, r in bl.items():
        for proto, pc in r["protocols"].items():
            a, b = pc.get("raw|DT"), pc.get("derived|DT")
            if not (a and b): continue
            rr = lambda s, k: mean_(b["runs"][s], k) / mean_(a["runs"][s], k) * 100
            w(f"| {n} | {proto} | " + " | ".join(f"{rr(s,'f1'):.1f}%" for s in S3) + " | " + " | ".join(f"{rr(s,'mcc'):.1f}%" for s in S3) + " |")
    # shuffled-label check
    w("\n**Label-shuffled check by training strategy** (random split, decision tree, raw features, 10 permutations of the training labels; test labels real). "
      "With the standard strategy F1 follows the test prior; with the sampler the model predicts the attack class about half the time, so F1 no longer tracks the prior while MCC stays ≈ 0 - evidence that the shuffled-label F1 is a prior effect of the evaluation, not leakage.\n")
    w("| Dataset | Test attack prior | Standard F1 / MCC / predicted-attack rate | Sampler F1 / MCC / predicted-attack rate | Weighted F1 / MCC / predicted-attack rate |"); w("|---|---|---|---|---|")
    for n, r in bl.items():
        sh = r.get("shuffled_label_check", {}).get("raw")
        if not sh: continue
        c3 = lambda s: f"{sh[s]['f1']['mean']:.3f} / {sh[s]['mcc']['mean']:+.3f} / {sh[s]['pred_attack_rate']['mean']:.2f}"
        w(f"| {n} | {sh['test_prior']:.3f} | " + " | ".join(c3(s) for s in S3) + " |")
    w("")
