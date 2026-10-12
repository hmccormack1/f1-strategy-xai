# Case studies

Numbers behind figures/case_*.png. Probabilities are P(pit within 3 laps); XGBoost pit-likely threshold = 0.21, EBM = 0.23.

## Textbook stop: 2025 São Paulo Grand Prix (NOR)
- Actual stops: laps [30, 50]
- Explained lap 29: Medium tyres, 29 laps old; pace drop-off 1.30s; position 1; SC/VSC on upcoming lap: no
- XGBoost P(pit) = 79%; EBM P(pit) = 60%
- SHAP top 3: Tyre age, Pace drop-off, Others pitting
- EBM top 3: Tyre age, Laps remaining, Pace drop-off
- Same top-3 set: no

## Safety car call: 2026 Australian Grand Prix (RUS)
- Actual stops: laps [12]
- Explained lap 11: Medium tyres, 11 laps old; pace drop-off 0.72s; position 2; SC/VSC on upcoming lap: yes
- XGBoost P(pit) = 40%; EBM P(pit) = 77%
- SHAP top 3: Race progress, SC/VSC deployed, Others pitting
- EBM top 3: Race progress, SC/VSC deployed, Laps remaining
- Same top-3 set: no

## Blind-spot gamble: 2026 Singapore Grand Prix (VER)
_Not in the data yet._

## Stayed out (fallback gamble): 2025 Italian Grand Prix (NOR)
- Actual stops: laps [46]
- Explained lap 33: Medium tyres, 33 laps old; pace drop-off 0.23s; position 2; SC/VSC on upcoming lap: no
- XGBoost P(pit) = 85%; EBM P(pit) = 66%
- SHAP top 3: Laps remaining, Tyre age, Others pitting
- EBM top 3: Tyre age, Laps remaining, Others pitting
- Same top-3 set: yes

