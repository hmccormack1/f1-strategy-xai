# Case-study candidates (2025–2026, model never trained on these)

Probabilities are XGBoost's P(pit within 3 laps); the decision threshold is 0.21.

## 1. Textbook stops
Green-flag stops the model was most confident about.

| Race | Driver | Stop lap | Model prob. | Tyres | Top SHAP reasons |
|---|---|---|---|---|---|
| 2025 Bahrain Grand Prix | SAI | 14 | 92% | Soft, 16 laps | Compound (+0.89), Race progress (+0.64), Others pitting (+0.63) |
| 2026 Bahrain Grand Prix | VER | 33 | 91% | Soft, 23 laps | Tyre age (+1.32), Laps remaining (+0.96), Others pitting (+0.96) |
| 2026 Bahrain Grand Prix | HAD | 31 | 91% | Soft, 24 laps | Tyre age (+1.41), Laps remaining (+0.82), Pace drop-off (+0.72) |
| 2025 Bahrain Grand Prix | ALB | 16 | 90% | Soft, 14 laps | Others pitting (+0.94), Compound (+0.88), Pace drop-off (+0.76) |
| 2025 Las Vegas Grand Prix | LEC | 24 | 89% | Medium, 23 laps | Tyre age (+1.06), Laps remaining (+0.84), Gap ahead (+0.55) |
| 2026 Bahrain Grand Prix | ANT | 33 | 89% | Medium, 23 laps | Tyre age (+1.26), Others pitting (+0.96), Laps remaining (+0.95) |
| 2026 Bahrain Grand Prix | HUL | 32 | 89% | Soft, 23 laps | Tyre age (+1.26), Laps remaining (+0.91), Others pitting (+0.78) |
| 2025 United States Grand Prix | HAM | 31 | 87% | Medium, 30 laps | Tyre age (+1.28), Others pitting (+0.82), Laps remaining (+0.74) |

## 3. Surprise stops
Green-flag stops the model did not see coming. Bold call, or information the model can't see?

| Race | Driver | Stop lap | Model prob. | Tyres | Top SHAP reasons |
|---|---|---|---|---|---|
| 2026 Canadian Grand Prix | NOR | 2 | 0% | Inter, 1 laps | Two compounds used (+0.26), Stint (+0.14), Rain (+0.05) |
| 2026 Canadian Grand Prix | BOR | 2 | 0% | Inter, 1 laps | Two compounds used (+0.26), Stint (+0.08), Others pitting (+0.08) |
| 2025 Monaco Grand Prix | VER | 77 | 0% | Medium, 46 laps | Tyre age (+0.42), Laps remaining (+0.24), Pace drop-off (+0.23) |
| 2026 Canadian Grand Prix | HUL | 2 | 0% | Inter, 1 laps | Two compounds used (+0.25), Stint (+0.08), Rain (+0.06) |
| 2026 British Grand Prix | ANT | 43 | 0% | Medium, 7 laps | Laps remaining (+0.40), Gap ahead (+0.30), Compound (+0.10) |
| 2026 Dutch Grand Prix | LIN | 5 | 0% | Medium, 1 laps | Position (+0.32), Yellow flag (+0.23), Gap behind (+0.14) |
| 2026 Canadian Grand Prix | PER | 2 | 0% | Inter, 1 laps | Position (+0.30), Two compounds used (+0.26), Gap behind (+0.25) |
| 2025 Monaco Grand Prix | SAI | 53 | 1% | Medium, 3 laps | Laps remaining (+0.48), Gap ahead (+0.26), Compound (+0.13) |

## 2. Safety-car races
Races with the most pit stops made under SC/VSC (all caution periods in the race combined).

| Race | Stops under SC/VSC | Median lap | Avg. model prob. before stop |
|---|---|---|---|
| 2025 Australian Grand Prix | 71 | 4 | 12% |
| 2025 Canadian Grand Prix | 51 | 67 | 7% |
| 2026 Monaco Grand Prix | 49 | 65 | 24% |
| 2026 Azerbaijan Grand Prix | 34 | 31 | 40% |
| 2026 Bahrain Grand Prix | 31 | 9 | 20% |
| 2026 British Grand Prix | 27 | 46 | 30% |
| 2025 Dutch Grand Prix | 26 | 52 | 39% |
| 2025 Emilia Romagna Grand Prix | 25 | 29 | 37% |

## 4. Stayed out
The model was sure a stop was coming, and the driver stayed out.

| Race | Driver | Lap | Model prob. | Tyres | Top SHAP reasons |
|---|---|---|---|---|---|
| 2026 Canadian Grand Prix | STR | 42 | 93% | Soft, 28 laps | SC/VSC deployed (+1.23), Tyre age (+1.18), Compound (+0.72) |
| 2025 Singapore Grand Prix | SAI | 38 | 92% | Medium, 38 laps | Tyre age (+1.27), Laps remaining (+1.01), Pace drop-off (+0.68) |
| 2026 British Grand Prix | PER | 36 | 88% | Hard, 15 laps | Laps remaining (+1.24), SC/VSC deployed (+1.02), Others pitting (+0.54) |
| 2026 British Grand Prix | ANT | 21 | 86% | Medium, 21 laps | SC/VSC deployed (+1.48), Others pitting (+0.77), Tyre age (+0.63) |
| 2025 Italian Grand Prix | PIA | 34 | 85% | Medium, 34 laps | Laps remaining (+1.25), Tyre age (+1.11), Others pitting (+0.62) |
| 2025 Italian Grand Prix | NOR | 33 | 85% | Medium, 33 laps | Laps remaining (+1.15), Tyre age (+1.09), Others pitting (+0.65) |
| 2025 Canadian Grand Prix | OCO | 48 | 85% | Hard, 48 laps | Tyre age (+1.78), Laps remaining (+0.92), Two compounds used (+0.64) |
| 2026 British Grand Prix | BOT | 20 | 84% | Medium, 20 laps | SC/VSC deployed (+1.25), Others pitting (+0.58), Tyre age (+0.55) |
