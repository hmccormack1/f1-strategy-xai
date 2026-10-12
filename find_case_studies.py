"""Shortlist interesting moments in the test seasons for case studies:
textbook stops, safety-car races, surprise stops, and confident misses
where the driver stayed out.

    python find_case_studies.py      (after run_pipeline.py)
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, SHORT_NAMES
from src import explain as ex

YEARS = (2025, 2026)
N = 8

feat = pd.read_csv("data/features.csv")
stops = pd.read_csv("data/pit_stops.csv")
xgbm = joblib.load("models/xgboost.joblib")
thr = json.load(open("models/thresholds.json"))["xgboost"]

d = feat[feat["Year"].isin(YEARS)].copy()
d["p"] = xgbm.predict_proba(d[FEATURE_COLS])[:, 1]
d = d.sort_values(["race_id", "Driver", "LapNumber"])
stops = stops[stops["race_id"].str[:4].astype(int).isin(YEARS)]


def top_reasons(row, k=3):
    sv = ex.shap_values(xgbm, row.to_frame().T.astype({c: float for c in FEATURE_COLS}))[0]
    idx = np.argsort(-sv)[:k]
    return ", ".join(f"{SHORT_NAMES[FEATURE_COLS[i]]} ({sv[i]:+.2f})" for i in idx)


# For every real stop: the model's highest probability in the 3 laps before it
rows = []
for s in stops.itertuples():
    w = d[(d["race_id"] == s.race_id) & (d["Driver"] == s.Driver) &
          (d["LapNumber"] >= s.LapNumber - 3) & (d["LapNumber"] < s.LapNumber)]
    if len(w):
        best = w.loc[w["p"].idxmax()]
        rows.append(dict(race_id=s.race_id, Event=best["Event"], Driver=s.Driver, stop_lap=s.LapNumber,
                         max_p=best["p"], tyre_life=best["tyre_life"], compound=best["compound"],
                         sc=int(w["sc_upcoming"].max()), featrow=best))
st = pd.DataFrame(rows)

comp = {1: "Soft", 2: "Medium", 3: "Hard", 4: "Inter", 5: "Wet"}
L = ["# Case-study candidates (2025–2026, model never trained on these)", "",
     f"Probabilities are XGBoost's P(pit within 3 laps); the decision threshold is {thr:.2f}.", ""]


def section(title, df, note):
    L.extend([f"## {title}", note, "", "| Race | Driver | Stop lap | Model prob. | Tyres | Top SHAP reasons |",
              "|---|---|---|---|---|---|"])
    for r in df.itertuples():
        L.append(f"| {r.race_id[:4]} {r.Event} | {r.Driver} | {int(r.stop_lap)} | {r.max_p:.0%} | "
                 f"{comp.get(int(r.compound), '?')}, {int(r.tyre_life)} laps | {top_reasons(r.featrow)} |")
    L.append("")


section("1. Textbook stops", st[st["sc"] == 0].nlargest(N, "max_p"),
        "Green-flag stops the model was most confident about.")
section("3. Surprise stops", st[st["sc"] == 0].nsmallest(N, "max_p"),
        "Green-flag stops the model did not see coming. Bold call, or information the model can't see?")

# Safety-car waves: laps where an SC/VSC appears and several cars pit
waves = (st[st["sc"] == 1].groupby(["race_id", "Event"])
         .agg(cars_pitted=("Driver", "size"), lap=("stop_lap", "median"), mean_p=("max_p", "mean"))
         .sort_values("cars_pitted", ascending=False).head(N).reset_index())
L.extend(["## 2. Safety-car races", "Races with the most pit stops made under SC/VSC (all caution periods in the race combined).", "",
          "| Race | Stops under SC/VSC | Median lap | Avg. model prob. before stop |", "|---|---|---|---|"])
L += [f"| {r.race_id[:4]} {r.Event} | {r.cars_pitted} | {int(r.lap)} | {r.mean_p:.0%} |" for r in waves.itertuples()]
L.append("")

# Stayed out: confident predictions with no stop in the next 5 laps
stop_set = {(s.race_id, s.Driver, l) for s in stops.itertuples() for l in range(int(s.LapNumber) - 5, int(s.LapNumber) + 1)}
d["stopped_soon"] = [(r, dr, l) in stop_set for r, dr, l in zip(d["race_id"], d["Driver"], d["LapNumber"])]
# exclude drivers whose race ended within 5 laps (retirement), which isn't "staying out"
last_lap = d.groupby(["race_id", "Driver"])["LapNumber"].transform("max")
d["retired_soon"] = (last_lap < d["TotalLaps"] - 1) & (d["LapNumber"] >= last_lap - 5)
out = (d[~d["stopped_soon"] & ~d["retired_soon"]].sort_values("p", ascending=False)
       .drop_duplicates(["race_id", "Driver"]).head(N))
L.extend(["## 4. Stayed out", "The model was sure a stop was coming, and the driver stayed out.", "",
          "| Race | Driver | Lap | Model prob. | Tyres | Top SHAP reasons |", "|---|---|---|---|---|---|"])
for _, r in out.iterrows():
    L.append(f"| {r['race_id'][:4]} {r['Event']} | {r['Driver']} | {int(r['LapNumber'])} | {r['p']:.0%} | "
             f"{comp.get(int(r['compound']), '?')}, {int(r['tyre_life'])} laps | {top_reasons(r[FEATURE_COLS])} |")

open("results/case_study_candidates.md", "w").write("\n".join(L) + "\n")
print("Wrote results/case_study_candidates.md")
