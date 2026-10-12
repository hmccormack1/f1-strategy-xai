"""Fake race data in the same format as get_f1_data.py, for testing the
pipeline without downloading. Not used for any reported result.

    python tests/make_synthetic_data.py   # -> data/synthetic_laps.csv
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
DRIVERS = [f"D{i:02d}" for i in range(20)]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
DEG = {"SOFT": 0.09, "MEDIUM": 0.055, "HARD": 0.035}       # s/lap of tyre wear
TARGET = {"SOFT": 17, "MEDIUM": 27, "HARD": 37}            # typical stint length


def simulate_race(year, rnd, shift=False):
    total = int(rng.integers(50, 71))
    scale = total / 57
    sc_laps, vsc_laps = set(), set()
    for _ in range(rng.poisson(0.8)):
        s = int(rng.integers(3, total - 5)); sc_laps |= set(range(s, s + int(rng.integers(3, 6))))
    if rng.random() < 0.3:
        s = int(rng.integers(3, total - 5)); vsc_laps |= set(range(s, s + 2))
    track_temp = rng.uniform(25, 50)
    rows = []
    for grid, drv in enumerate(DRIVERS):
        base = 90 + grid * 0.08 + rng.normal(0, 0.2)
        comp = rng.choice(COMPOUNDS, p=[0.25, 0.6, 0.15])
        used, stint, life = {comp}, 1, 0
        t = grid * 0.3
        dnf_lap = int(rng.integers(5, total)) if rng.random() < 0.06 else None
        pit_next = False
        for lap in range(1, total + 1):
            life += 1
            out_lap = pit_next
            if pit_next:
                left = total - lap
                choices = [c for c in COMPOUNDS if c not in used] or COMPOUNDS
                comp = "HARD" if left > 30 and "HARD" in choices else rng.choice(choices)
                used.add(comp); stint += 1; life = 1; pit_next = False
            status = "4" if lap in sc_laps else ("6" if lap in vsc_laps else "1")
            target = TARGET[comp] * scale * (1.3 if shift else 1.0)
            z = 0.35 * (life - target) + (1.8 if status in "46" and life > 8 else 0)
            p = 1 / (1 + np.exp(-z))
            must = len(used) < 2 and total - lap <= 3
            pit = (rng.random() < p * 0.35 or must) and total - lap >= 2 and life > 3
            lt = base + DEG[comp] * life * (0.7 if shift else 1) + rng.normal(0, 0.3)
            lt += 30 if status == "4" else (15 if status == "6" else 0)
            lt += 3 if pit else 0
            lt += 18 if out_lap else 0
            t += lt
            retire = dnf_lap == lap
            rows.append(dict(
                Year=year, Round=rnd, Event=f"Synthetic GP {rnd}", Driver=drv,
                DriverNumber=grid + 1, Team=f"Team{grid // 2}", LapNumber=lap,
                Position=np.nan, LapTime=lt, Time=t, LapStartTime=t - lt,
                Stint=stint, Compound=comp, TyreLife=float(life), FreshTyre=True,
                PitInTime=t - 1 if (pit or retire) else np.nan,
                PitOutTime=(t - lt + 20) if out_lap else np.nan,
                TrackStatus=status, Deleted=False, IsAccurate=status == "1",
                AirTemp=track_temp - 12, TrackTemp=track_temp, Humidity=50.0,
                Rainfall=False, WindSpeed=2.0, TotalLaps=total))
            if retire:
                break
            pit_next = pit
    df = pd.DataFrame(rows)
    df["Position"] = df.groupby("LapNumber")["Time"].rank(method="first")
    return df


if __name__ == "__main__":
    races = []
    for year, n in [(2023, 22), (2024, 24), (2025, 24), (2026, 18)]:
        for rnd in range(1, n + 1):
            races.append(simulate_race(year, rnd, shift=(year == 2026)))
    out = pd.concat(races, ignore_index=True)
    out.to_csv("data/synthetic_laps.csv", index=False)
    print(f"{len(out):,} synthetic laps -> data/synthetic_laps.csv")
