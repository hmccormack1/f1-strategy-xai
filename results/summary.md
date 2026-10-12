# Results summary

## Data
- Races per season: {2023: 22, 2024: 24, 2025: 24, 2026: 16}
- Share of laps labeled 'pits within 3 laps': {2023: 0.0927, 2024: 0.0846, 2025: 0.0839, 2026: 0.0955}

## Prediction quality

| Model | Season | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| baseline | 2025 | 0.1497 | 0.6846 | 0.1407 | 0.5816 |
| baseline | 2026 | 0.1718 | 0.6934 | 0.1604 | 0.5303 |
| xgboost | 2025 | 0.3131 | 0.8122 | 0.2723 | 0.5333 |
| xgboost | 2026 | 0.3313 | 0.7871 | 0.2971 | 0.4985 |
| ebm | 2025 | 0.2917 | 0.7901 | 0.2778 | 0.4817 |
| ebm | 2026 | 0.3088 | 0.7778 | 0.3029 | 0.4578 |

Random guessing PR-AUC = the pit rate (0.0839 in 2025).

## 1. Agreement (SHAP vs EBM, 2025)
- Rank correlation of global importance: **0.946**
- Average overlap of each lap's top-3 reasons (Jaccard): **0.495**
- Shape correlation per feature: Tyre age 0.998, Compound 0.843, Race progress 0.976, Laps remaining 0.736, Stint -0.006, Position 0.742, Gap ahead 0.927, Gap behind 0.945, Pace drop-off 0.996, Others pitting 0.999, Track temp change 0.877

## 2. Stability (SHAP across retraining seeds)
- Mean pairwise rank correlation: **0.995**
- Mean pairwise top-3 overlap per lap: **0.855**

## 3. Faithfulness (deletion test on 1285 predicted-pit laps)
- Features removed: [0, 1, 2, 3, 4, 5]
- SHAP ranking: [0.3891, 0.1827, 0.1141, 0.0782, 0.059, 0.0485]
- EBM ranking (applied to XGBoost): [0.3891, 0.2066, 0.1327, 0.0905, 0.0693, 0.0551]
- Random ranking: [0.3891, 0.3559, 0.3283, 0.3014, 0.2757, 0.2582]

## 4. Distribution shift (2026)
- Importance rank correlation 2025 vs 2026: **0.992**
- Largest feature shifts (in SDs): Track temp change 0.205, Compound -0.184, Gap ahead 0.162, Gap behind 0.161

## 5. Design iterations (XGBoost PR-AUC on the test season)
- v1 original (absolute temp, no SC-upcoming): 0.3189 ± 0.0048
- v2 + SC/VSC deployed feature: 0.3297 ± 0.0041
- No temperature (control): 0.34 ± 0.0011
- v3 final (temp change instead of absolute): 0.3164 ± 0.0018
