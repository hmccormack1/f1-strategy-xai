# Pit Wall Explainer: Can you trust an AI race strategist's reasons?

An F1 model predicts whether a driver will pit in the next 3 laps, then this project
tests whether the model's **explanations** can be trusted: do two different explanation
methods agree, do they stay stable, are they faithful to the model, and do they survive
the 2026 regulation change?

*AIPI 590: Emerging Trends in Explainable AI, Duke University, Fall 2026, Project 1.*

---

## Why this matters

Real F1 teams lean on strategy models under seconds of time pressure. An explanation like
*"pit now: tyres are 24 laps old and a safety car just came out"* only helps a strategist if
it reflects what the model is actually doing. If it does not, it builds the wrong kind of
trust: confident, persuasive and wrong (Unit 0, *calibrated trust*). The same questions apply
anywhere people act on explained AI decisions, so F1 is a fast, public, well-measured test bed
for them.

## Research question

**Can we trust the explanations of an AI race-strategy model?**

1. **Agreement:** Do a post-hoc explanation (SHAP on XGBoost) and a glass-box model (EBM) tell the same story?
2. **Stability:** Do the explanations stay the same when the model is retrained with a different random seed?
3. **Faithfulness:** When SHAP says a feature matters, does removing it actually change the prediction?
4. **Shift:** Do accuracy and explanations hold up on 2026 races, after the regulations changed the cars?

## Results

Data: 86 races (2023: 22, 2024: 24, 2025: 24, 2026: 16), about 92,000 driver-laps.
Trained on 2023–24, tested on 2025, distribution-shift test on 2026. Full numbers are in
`results/summary.md` and `results/metrics.json`.

**Prediction.** Pit laps are rare (about 8% of laps), so the metric is PR-AUC, where random guessing scores about 0.08.

| Model | 2025 (test) | 2026 (new regulations) |
|---|---|---|
| Tyre-age rule of thumb | 0.15 | 0.17 |
| XGBoost | **0.31** | **0.33** |
| EBM (glass box) | 0.29 | 0.31 |

**Findings**

1. **The explanations agree on the big picture but not on individual laps.** SHAP and the EBM rank
   features almost identically overall (rank correlation 0.95), yet share only about half of each lap's
   top-3 reasons (Jaccard 0.50). A strategist asking "why this lap?" can get different answers
   depending on the method.
2. **SHAP is stable and faithful to its model.** Its rankings barely change across five retrainings
   (0.995). Removing SHAP's top features cuts the predicted pit probability far faster than removing
   random features (0.39 → 0.18 after one feature, vs. 0.36 for random).
3. **Faithful is not the same as meaningful.** In the first version SHAP ranked absolute track
   temperature #3. It turned out to work as a race fingerprint (it varies about 9°C between races
   but only about 1.3°C within one), not as tyre-degradation physics. SHAP faithfully reported a shortcut.
4. **Explanations exposed a blind spot, and the fix held up.** SHAP showed the model barely used the
   safety car, against basic F1 knowledge. The cause: 27% of stops happen the moment an SC/VSC appears,
   which the original lap-end snapshot couldn't see. Adding "SC/VSC out on the driver's upcoming lap"
   improved 2025 PR-AUC from 0.319 to 0.330 (5 seeds, SD ≈ 0.004).
5. **Not every explanation-driven fix replicates.** Replacing absolute temperature with temperature
   *change* during the race was the best version on 2024 (used for development), but not on 2025,
   where dropping temperature entirely did best (0.340 vs. 0.316). The final model keeps the version
   chosen before seeing 2025, so the test season was never used for tuning.
6. **Additivity has limits.** In the EBM, race progress and laps remaining get large, opposite-signed
   contributions (they carry almost the same information), and temperature hurts it because its effect
   depends on rain and compound, an interaction a purely additive model can't represent.
7. **The 2026 rules didn't break the model.** Performance held (0.33) and its reasoning barely moved
   (importance rank correlation 2025 vs. 2026: 0.99). Pit-strategy logic carried over the regulation change.

**Case studies** (`results/case_studies.md`, `figures/case_*.png`)

| Case | What happened | What the model said |
|---|---|---|
| Textbook: 2025 São Paulo, Norris, lap 30 | Green-flag stop on 29-lap-old mediums | 79% one lap before. Top reasons: tyre age, pace drop-off, others pitting |
| Safety car: 2026 Australia, Russell, lap 12 | Stop under a VSC on 11-lap-old tyres | 40% (above threshold). The SC/VSC feature was a top reason in both methods |
| Gamble: 2025 Italy, Norris | Stayed out until lap 46 | 85% on lap 33, driven by tyre age and laps remaining. His pace drop-off was only 0.23 s: the team read the tyres; the model followed habit |
| Gamble: 2026 Singapore, Verstappen | *Pending: added once the race data is published* | |

| Figure | What it shows |
|---|---|
| `figures/1_importance_shap_vs_ebm.png` | What each method says matters most |
| `figures/2_effect_shapes.png` | How each top feature pushes the prediction, per method |
| `figures/3_stability.png` | Importance across retrainings (error bars) |
| `figures/4_faithfulness.png` | Deletion test vs. random |
| `figures/5_performance.png` | Prediction quality vs. a tyre-age rule of thumb |
| `figures/6_shift_importance.png` | 2025 vs. 2026 reasoning |
| `figures/7_example_race.png` | One race replayed lap by lap |
| `figures/8_design_iterations.png` | How each explanation-driven change affected accuracy |
| `figures/case_*.png` | The case studies |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python get_f1_data.py          # optional: data/f1_laps_2023_2026.csv is included; this re-downloads/updates it
python run_pipeline.py         # features, models, explanations, evaluation, figures (~10–15 min)
python case_studies.py         # case-study figures (edit CASES to change races)
python case_studies.py --poster  # poster-sized versions of the case-study figures
python poster_figures.py       # poster-sized importance and design-iteration figures
python find_case_studies.py    # optional: shortlist of interesting moments in 2025–26
streamlit run app.py           # the interactive replay app
```

On a Mac, XGBoost also needs OpenMP: `brew install libomp`. No API keys are needed. Data comes from F1's public timing feed via the open-source
[FastF1](https://docs.fastf1.dev/) library. To test the code without downloading,
`python tests/make_synthetic_data.py` creates fake data in the same format
(`python run_pipeline.py --data data/synthetic_laps.csv --quick`).

## How it works

**Data.** One row per driver per lap, 2023–2026, with only information a strategist would
have at the decision point: tyre compound and age, stint, race progress, position, gaps to
the cars ahead and behind, safety car / VSC / yellow flags, whether an SC/VSC is out on the
driver's upcoming lap (known before pit entry), recent pace drop-off, how many rivals just
pitted, whether the car ahead just pitted (undercut cue), whether the two-compound rule is
satisfied, change in track temperature since the start (capped at −6 to +4°C, where data is
plentiful) and rain.

**Label.** `1` if the driver makes a pit stop in the next 3 laps.

**Avoiding leakage.**
- The split is by season, so no race is in both training and testing: train 2023–24, test 2025, shift-test 2026.
- In-laps are dropped, because their slow lap time already reveals the stop.
- The safety-car feature uses each driver's *own* next lap. An earlier version matched laps by number
  across the field, which leaked the future (a backmarker's lap 28 happens minutes after the leader's).
  A case-study check caught it: a "stayed out under the SC" moment turned out to be a retirement plus
  a caution the leader couldn't have known about yet. All reported results use the fixed version.
- Retirements into the garage and red-flag tyre changes are not counted as strategy stops.

**Models.**
- *Baseline:* historical pit rate for the current compound and tyre age.
- *XGBoost:* gradient-boosted trees, a black box, explained with SHAP (TreeExplainer).
- *EBM:* Explainable Boosting Machine, a GAM with interactions turned off, so the model is its own explanation.

Both explanations are in the same unit (log-odds of pitting), so they can be compared directly.

**Metrics.** Pit laps are rare (about 8–10% of rows, depending on the season), so accuracy is misleading. The main metric is
**PR-AUC**, where random guessing scores the pit rate. Precision and recall are reported at an
F1-optimal threshold chosen on 2024 data.

**Evaluation of explanations** (Explainable AI workbook, Unit 2, Ch. 9):
- *Agreement:* rank correlation of global importance, overlap of each lap's top-3 reasons, correlation of effect curves.
- *Stability:* five retrainings with different seeds, then pairwise rank correlation and top-3 overlap.
- *Faithfulness:* deletion test on predicted-pit laps. Remove the top-k features by SHAP, by EBM and at random (replaced with values from random training laps), then track the drop in predicted probability.
- *Shift:* metrics and SHAP importance on 2026 vs. 2025, plus feature drift.

## Limitations

- **It learns what teams did, not what was optimal.** The model copies real habits, mistakes included (historical bias).
- **It can't see** tyre wear, team radio, weather forecasts or each team's plan, so some stops are unpredictable from public data.
- **Tyre compound names are relative.** "Soft" at one race may be a harder rubber than "soft" at another.
- **Correlated features split credit.** Race progress and laps remaining carry similar information, so SHAP and the EBM can divide importance between them differently even when they agree on the overall story.
- **Modest accuracy.** About a quarter of predicted-pit laps are right (precision ≈ 0.27, PR-AUC ≈ 0.31). Pit timing depends on things public data can't see, so the explanations describe a model that is useful but often wrong.
- **One test season.** The temperature result flipped between 2024 and 2025, a reminder that single-season findings can be noise. Results are averaged over 5 seeds but not over multiple seasons.
- **The deletion test creates unrealistic laps** (out-of-distribution inputs). It is compared against a random baseline for that reason, but it is still an approximation.
- **Replay only, not live.** FastF1 data is loaded after a session ends.

## Future work

- Live mode during races (FastF1 can record the live timing feed).
- Predict *which* compound comes next, and whether an undercut will work.
- A user study with F1 fans or analysts: do the explanations improve their predictions of team decisions?

## Project structure

```
get_f1_data.py        download raw laps (run locally)
run_pipeline.py       everything else, end to end
case_studies.py       case-study figures
poster_figures.py     poster versions of the importance and iteration figures
find_case_studies.py  shortlist of interesting moments in the test seasons
app.py                Streamlit replay app
src/features.py       feature engineering and label
src/models.py         baseline, XGBoost, EBM, split, metrics
src/explain.py        SHAP/EBM explanations and the four evaluation experiments
src/figures.py        figures
tests/                synthetic data generator for testing
results/              metrics.json, summary.md
figures/              PNGs for the README (figures/poster/ for the poster)
```

## References

- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Nori et al. (2019). *InterpretML: A Unified Framework for Machine Learning Interpretability* (EBM).
- Lou, Caruana & Gehrke (2012). *Intelligible Models for Classification and Regression* (GAMs).
- Bent, B. *Explainable AI Workbook*, Units 0–2 (calibrated trust, GAMs, SHAP, evaluating explanations).
- FastF1: https://docs.fastf1.dev/

## AI use

Built with Claude (Anthropic) as a coding assistant, in line with the course AI policy.
Research question, design decisions, case studies and interpretation of results are the author's own.
