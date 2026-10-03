import json, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob(str(Path(__file__).parent / "results/robust_*.json"))):
    for k, v in json.load(open(f)).items():
        rows.append((k, v))
def c(m, k): return f"{m[k]['mean']:.4f} ±{m[k]['ci_hi']-m[k]['mean']:.4f}"
print(f"{'dataset|protocol':28} {'baseline F1 (mean ±95%CI)':26} {'tuned F1':20} {'tuned AUC':20} {'Brier':8} {'ECE':8} {'hi-conf cov/acc':16} feature sets")
for k, v in rows:
    t, b = v["tuned"], v["baseline"]
    fs = ",".join(sorted({i["feature_set"] for i in v["iterations"]}))
    print(f"{k:28} {c(b,'f1'):26} {c(t,'f1'):20} {c(t,'auc'):20} {t['brier']['mean']:.4f}  {t['ece']['mean']:.4f}  {t['hiconf_cov']['mean']:.3f}/{t['hiconf_acc']['mean']:.4f}  {fs}")
