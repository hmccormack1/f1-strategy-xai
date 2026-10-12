"""Raw laps -> features -> models -> explanations -> evaluation -> figures.

    python run_pipeline.py [--data data/f1_laps_2023_2026.csv]

Writes data/features.csv, data/pit_stops.csv, models/, results/metrics.json,
results/summary.md and figures/.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_features, pit_laps, FEATURE_COLS, SHORT_NAMES
from src.models import train_all, split, make_xgb, DEFAULT_SPLIT
from src import explain as ex
from src import figures as fg

ROOT = Path(__file__).parent
for d in ("data", "models", "results", "figures"):
    (ROOT / d).mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample(df, n, seed=0):
    return df.sample(min(n, len(df)), random_state=seed) if len(df) else df


def main(data_path, cfg=DEFAULT_SPLIT, quick=False):
    R = {"split": cfg}
    TEST, SHIFT = cfg["test"], cfg.get("shift")

    # Features
    log("Building features ...")
    raw = pd.read_csv(data_path, low_memory=False)
    feat = build_features(raw)
    stops = pit_laps(raw)
    feat.to_csv(ROOT / "data/features.csv", index=False)
    stops.to_csv(ROOT / "data/pit_stops.csv", index=False)
    R["data"] = {
        "races_per_season": feat.groupby("Year")["race_id"].nunique().to_dict(),
        "rows_per_season": feat.groupby("Year").size().to_dict(),
        "pit_label_rate_per_season": feat.groupby("Year")["label"].mean().round(4).to_dict(),
        "features": FEATURE_COLS,
    }
    log(f"  {len(feat):,} driver-laps from {feat['race_id'].nunique()} races")

    # Models
    log("Training models ...")
    models, thr, metrics = train_all(feat, cfg, log=log)
    R["metrics"] = metrics
    for name, m in models.items():
        joblib.dump(m, ROOT / f"models/{name}.joblib")
    json.dump(thr, open(ROOT / "models/thresholds.json", "w"), indent=2)
    xgbm, ebm = models["xgboost"], models["ebm"]
    tr, te, sh = split(feat, cfg)

    # Explanations + agreement
    log(f"Explaining the {TEST} test season ...")
    te_s = sample(te, 8000)
    shap_c, ebm_c = ex.shap_values(xgbm, te_s), ex.ebm_contribs(ebm, te_s)
    R["agreement"] = ex.agreement(te_s, shap_c, ebm_c)
    top = pd.Series(R["agreement"]["importance_shap"]).sort_values(ascending=False).index[:4]
    curves = {f: (ex.shape_curves(te_s, shap_c, f), ex.shape_curves(te_s, ebm_c, f)) for f in top}
    fg.importance_comparison(R["agreement"]["importance_shap"], R["agreement"]["importance_ebm"],
                             "XGBoost + SHAP", "EBM (glass box)",
                             "Do the two explanations agree on what matters?",
                             ROOT / "figures/1_importance_shap_vs_ebm.png")
    fg.shape_comparison(curves, ROOT / "figures/2_effect_shapes.png")

    # Stability
    log("Stability across retraining seeds ...")
    seeds = (0, 1, 2) if quick else (0, 1, 2, 3, 4)
    R["stability"] = ex.stability(make_xgb, tr, tr["label"], sample(te, 4000, 1), seeds=seeds, log=log)
    fg.stability_bars(R["stability"]["importance_mean"], R["stability"]["importance_std"],
                      ROOT / "figures/3_stability.png",
                      title=f"SHAP importance across {len(seeds)} retrainings")

    # Faithfulness
    log("Faithfulness (deletion test) ...")
    p = xgbm.predict_proba(te_s[FEATURE_COLS])[:, 1]
    sel = np.where(p >= thr["xgboost"])[0][:1500]
    R["faithfulness"] = ex.faithfulness(xgbm, te_s.iloc[sel], shap_c[sel], ebm_c[sel], sample(tr, 5000, 2))
    R["faithfulness"]["n_laps"] = int(len(sel))
    fg.deletion_plot(R["faithfulness"], ROOT / "figures/4_faithfulness.png")

    # Distribution shift
    fg.performance_bars(metrics, [str(TEST)] + ([str(SHIFT)] if len(sh) else []), ROOT / "figures/5_performance.png")
    if len(sh):
        log(f"{SHIFT} distribution-shift test ...")
        R["shift"] = ex.shift_report(xgbm, te_s, sample(sh, 8000))
        fg.importance_comparison(R["shift"]["importance_test"], R["shift"]["importance_shift"],
                                 f"{TEST} races", f"{SHIFT} races (new regs)",
                                 "Does the model reason the same way under the 2026 rules?" if SHIFT == 2026 else "Does the model reason the same way next season?",
                                 ROOT / "figures/6_shift_importance.png", color_b=fg.AQUA)
    else:
        log(f"  no {SHIFT} races in the data; skipping shift test")

    # Design iterations
    log("Design iterations (feature-set versions) ...")
    seeds_it = (0, 1, 2) if quick else (0, 1, 2, 3, 4)
    R["iterations"] = ex.design_iterations(make_xgb, tr, te, seeds=seeds_it)
    fg.iterations_plot(R["iterations"], ROOT / "figures/8_design_iterations.png")

    # Example race timeline
    log("Example race timeline ...")
    stops_te = stops[stops["race_id"].str.startswith(str(TEST))]
    multi = stops_te.groupby(["race_id", "Driver"]).size()
    sc_races = set(te.loc[te["safety_car"] == 1, "race_id"])
    cands = [k for k in multi[multi >= 2].index if k[0] in sc_races] or list(multi.index)
    race, drv = cands[0]
    d = te[(te["race_id"] == race) & (te["Driver"] == drv)].sort_values("LapNumber")
    fg.race_timeline(d, xgbm.predict_proba(d[FEATURE_COLS])[:, 1],
                     stops_te[(stops_te["race_id"] == race) & (stops_te["Driver"] == drv)]["LapNumber"].tolist(),
                     thr["xgboost"], ROOT / "figures/7_example_race.png",
                     f"{d['Event'].iloc[0]} {TEST} — {drv}")

    json.dump(R, open(ROOT / "results/metrics.json", "w"), indent=2, default=str)
    write_summary(R, TEST, SHIFT)
    log("Done. See results/summary.md and figures/.")


def write_summary(R, TEST, SHIFT):
    m, a, s, f = R["metrics"], R["agreement"], R["stability"], R["faithfulness"]
    L = ["# Results summary", "", "## Data",
         f"- Races per season: {R['data']['races_per_season']}",
         f"- Share of laps labeled 'pits within 3 laps': {R['data']['pit_label_rate_per_season']}", "",
         "## Prediction quality", "", "| Model | Season | PR-AUC | ROC-AUC | Precision | Recall |", "|---|---|---|---|---|---|"]
    for name in ("baseline", "xgboost", "ebm"):
        for season, v in m[name].items():
            if isinstance(v, dict):
                L.append(f"| {name} | {season} | {v['pr_auc']} | {v['roc_auc']} | {v['precision']} | {v['recall']} |")
    L += ["", f"Random guessing PR-AUC = the pit rate ({m['xgboost'][str(TEST)]['pit_rate']} in {TEST}).", "",
          f"## 1. Agreement (SHAP vs EBM, {TEST})",
          f"- Rank correlation of global importance: **{a['global_rank_spearman']}**",
          f"- Average overlap of each lap's top-3 reasons (Jaccard): **{a['local_top3_jaccard']}**",
          "- Shape correlation per feature: " + ", ".join(f"{SHORT_NAMES[k]} {v}" for k, v in a["shape_correlation"].items()), "",
          "## 2. Stability (SHAP across retraining seeds)",
          f"- Mean pairwise rank correlation: **{s['mean_pairwise_rank_spearman']}**",
          f"- Mean pairwise top-3 overlap per lap: **{s['mean_pairwise_local_top3_jaccard']}**", "",
          f"## 3. Faithfulness (deletion test on {f['n_laps']} predicted-pit laps)",
          f"- Features removed: {f['k']}"]
    L += [f"- {k}: {v}" for k, v in f["curves"].items()]
    if "shift" in R:
        L += ["", f"## 4. Distribution shift ({SHIFT})",
              f"- Importance rank correlation {TEST} vs {SHIFT}: **{R['shift']['importance_rank_spearman']}**",
              "- Largest feature shifts (in SDs): " + ", ".join(
                  f"{SHORT_NAMES[k]} {v}" for k, v in sorted(R["shift"]["feature_mean_shift_sd"].items(),
                                                            key=lambda kv: -abs(kv[1]))[:4])]
    if "iterations" in R:
        L += ["", "## 5. Design iterations (XGBoost PR-AUC on the test season)"]
        L += [f"- {k}: {v['pr_auc_mean']} ± {v['pr_auc_sd']}" for k, v in R["iterations"].items()]
    (ROOT / "results/summary.md").write_text("\n".join(L) + "\n")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/f1_laps_2023_2026.csv")
    ap.add_argument("--train", default="2023,2024", help="training seasons, comma-separated")
    ap.add_argument("--test", type=int, default=2025)
    ap.add_argument("--shift", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="fewer seeds, for testing")
    args = ap.parse_args()
    cfg = {"train": [int(y) for y in args.train.split(",")], "test": args.test, "shift": args.shift}
    main(args.data, cfg, args.quick)
