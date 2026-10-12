"""Pit Wall Explainer: replay a race and see why the model expects a stop.

    streamlit run app.py      (run run_pipeline.py first)
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLS, SHORT_NAMES
from src import explain as ex
from src import figures as fg

st.set_page_config(page_title="Pit Wall Explainer", layout="wide")


@st.cache_resource
def load():
    feat = pd.read_csv("data/features.csv")
    stops = pd.read_csv("data/pit_stops.csv")
    models = {n: joblib.load(f"models/{n}.joblib") for n in ("xgboost", "ebm")}
    thr = json.load(open("models/thresholds.json"))
    return feat, stops, models, thr


feat, stops, models, thr = load()
xgbm, ebm = models["xgboost"], models["ebm"]

st.title("🏁 Pit Wall Explainer")
st.caption("Replay a race and see when an AI expects each driver to pit, and *why*, according to two "
           "different explanation methods: SHAP on a black-box XGBoost model, and a glass-box EBM.")

# race / driver picker
with st.sidebar:
    year = st.selectbox("Season", sorted(feat["Year"].unique(), reverse=True))
    races = feat[feat["Year"] == year][["race_id", "Round", "Event"]].drop_duplicates().sort_values("Round")
    event = st.selectbox("Race", races["Event"].tolist())
    race_id = races.loc[races["Event"] == event, "race_id"].iloc[0]
    drivers = sorted(feat.loc[feat["race_id"] == race_id, "Driver"].unique())
    driver = st.selectbox("Driver", drivers)
    st.markdown("---")
    st.caption(f"Training data: 2023–2024 seasons. 2025 = unseen test season; "
               f"2026 = new regulations. Earlier seasons were seen in training, so predictions "
               f"there look better than they really are.")

d = feat[(feat["race_id"] == race_id) & (feat["Driver"] == driver)].sort_values("LapNumber").reset_index(drop=True)
actual = stops[(stops["race_id"] == race_id) & (stops["Driver"] == driver)]["LapNumber"].tolist()
p_xgb = xgbm.predict_proba(d[FEATURE_COLS])[:, 1]
p_ebm = ebm.predict_proba(d[FEATURE_COLS])[:, 1]

lap = st.slider("Lap", int(d["LapNumber"].min()), int(d["LapNumber"].max()), int(d["LapNumber"].min()))
st.pyplot(fg.race_timeline(d, p_xgb, actual, thr["xgboost"], None,
                           f"{event} {year} — {driver}", highlight_lap=lap))

# selected lap
i = int(np.argmin(np.abs(d["LapNumber"].values - lap)))
row = d.iloc[[i]]
sv = ex.shap_values(xgbm, row)[0]
ev = ex.ebm_contribs(ebm, row)[0]

c1, c2, c3 = st.columns(3)
c1.metric("XGBoost: P(pit in next 3 laps)", f"{p_xgb[i]:.0%}",
          "PIT LIKELY" if p_xgb[i] >= thr["xgboost"] else "stay out", delta_color="off")
c2.metric("EBM: P(pit in next 3 laps)", f"{p_ebm[i]:.0%}",
          "PIT LIKELY" if p_ebm[i] >= thr["ebm"] else "stay out", delta_color="off")
nxt = [s for s in actual if s > row["LapNumber"].iloc[0]]
c3.metric("What actually happened", f"Pitted on lap {int(nxt[0])}" if nxt else "No more stops")

order = np.argsort(-np.maximum(np.abs(sv), np.abs(ev)))[:7]
names = [f"{SHORT_NAMES[FEATURE_COLS[j]]} = {row[FEATURE_COLS[j]].iloc[0]:.3g}" for j in order]
st.pyplot(fg.lap_explanation(names, sv[order], ev[order]))

agree = set(np.argsort(-np.abs(sv))[:3]) == set(np.argsort(-np.abs(ev))[:3])
st.info("✅ Both methods agree on the top 3 reasons for this lap." if agree else
        "⚠️ The two methods disagree on the top 3 reasons for this lap. A reminder that an "
        "explanation is an estimate, not the ground truth.")

with st.expander("How to read this"):
    st.markdown(
        "- **Bars** show how much each factor pushed the prediction toward (right) or away from (left) a pit stop, "
        "in log-odds. Larger bars matter more.\n"
        "- **SHAP** explains a black-box model after the fact. **EBM** is a glass-box model whose prediction is "
        "literally the sum of these bars.\n"
        "- The model learned from what teams **did**, not what was **optimal**, so it copies real teams' habits, "
        "good and bad.")
