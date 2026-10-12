"""Case-study figures: replay a race and explain one key lap.

    python case_studies.py      (after run_pipeline.py)

Writes figures/case_*.png and results/case_studies.md. Races not in the data are skipped.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, SHORT_NAMES
from src import explain as ex
from src import figures as fg

# focus = lap to explain; otherwise the highest-probability lap in the 3 laps before `stop`
CASES = [
    dict(key="textbook", year=2025, event="São Paulo Grand Prix", driver="NOR", stop=30,
         role="Textbook stop", subtitle="Green-flag stop on worn tyres: can the model see it coming?"),
    dict(key="safety_car", year=2026, event="Australian Grand Prix", driver="RUS", stop=12,
         role="Safety car call", subtitle="Early stop under a VSC on fairly fresh tyres: does the safety car fix work?"),
    dict(key="singapore", year=2026, event="Singapore Grand Prix", driver="VER", stop=None,
         role="Blind-spot gamble", subtitle="Pace management and track position: what the model can't see"),
    dict(key="stayed_out", year=2025, event="Italian Grand Prix", driver="NOR", focus=33,
         role="Stayed out (fallback gamble)", subtitle="The model expected a stop; the team extended the stint"),
]
COMP = {1: "Soft", 2: "Medium", 3: "Hard", 4: "Inter", 5: "Wet"}

feat = pd.read_csv("data/features.csv")
stops = pd.read_csv("data/pit_stops.csv")
xgbm, ebm = joblib.load("models/xgboost.joblib"), joblib.load("models/ebm.joblib")
thr = json.load(open("models/thresholds.json"))

out = ["# Case studies", "", "Numbers behind figures/case_*.png. Probabilities are P(pit within 3 laps); "
       f"XGBoost pit-likely threshold = {thr['xgboost']:.2f}, EBM = {thr['ebm']:.2f}.", ""]
for c in CASES:
    d = feat[(feat["Year"] == c["year"]) & (feat["Event"] == c["event"]) & (feat["Driver"] == c["driver"])]
    if d.empty:
        print(f"skip: {c['year']} {c['event']} {c['driver']} not in data")
        out += [f"## {c['role']}: {c['year']} {c['event']} ({c['driver']})", "_Not in the data yet._", ""]
        continue
    d = d.sort_values("LapNumber").reset_index(drop=True)
    rid = d["race_id"].iloc[0]
    s_laps = [int(x) for x in stops[(stops["race_id"] == rid) & (stops["Driver"] == c["driver"])]["LapNumber"]]
    p = xgbm.predict_proba(d[FEATURE_COLS])[:, 1]
    pe = ebm.predict_proba(d[FEATURE_COLS])[:, 1]
    if c.get("focus"):
        focus = c["focus"]
    else:
        stop = c["stop"] or (s_laps[0] if s_laps else int(d["LapNumber"].iloc[int(np.argmax(p))]))
        w = d[(d["LapNumber"] >= stop - 3) & (d["LapNumber"] < stop)]
        focus = int(w["LapNumber"].iloc[int(np.argmax(p[w.index]))])
    i = int(np.where(d["LapNumber"] == focus)[0][0])
    row = d.iloc[[i]]
    sv, ev = ex.shap_values(xgbm, row)[0], ex.ebm_contribs(ebm, row)[0]
    order = np.argsort(-np.maximum(np.abs(sv), np.abs(ev)))[:6]
    names = [f"{SHORT_NAMES[FEATURE_COLS[j]]} = {row[FEATURE_COLS[j]].iloc[0]:.3g}" for j in order]
    title = f"{c['role']}: {c['event']} {c['year']}, {c['driver']}"
    fg.case_study_panel(d, p, s_laps, thr["xgboost"], focus, names, sv[order], ev[order],
                        title, c["subtitle"], f"figures/case_{c['key']}.png")
    top_s = [SHORT_NAMES[FEATURE_COLS[j]] for j in np.argsort(-np.abs(sv))[:3]]
    top_e = [SHORT_NAMES[FEATURE_COLS[j]] for j in np.argsort(-np.abs(ev))[:3]]
    r = row.iloc[0]
    out += [f"## {c['role']}: {c['year']} {c['event']} ({c['driver']})",
            f"- Actual stops: laps {s_laps}",
            f"- Explained lap {focus}: {COMP.get(int(r['compound']))} tyres, {int(r['tyre_life'])} laps old; "
            f"pace drop-off {r['pace_delta']:.2f}s; position {int(r['position'])}; "
            f"SC/VSC on upcoming lap: {'yes' if r['sc_upcoming'] else 'no'}",
            f"- XGBoost P(pit) = {p[i]:.0%}; EBM P(pit) = {pe[i]:.0%}",
            f"- SHAP top 3: {', '.join(top_s)}",
            f"- EBM top 3: {', '.join(top_e)}",
            f"- Same top-3 set: {'yes' if set(top_s) == set(top_e) else 'no'}", ""]
    print(f"done: {title}")
open("results/case_studies.md", "w").write("\n".join(out) + "\n")
