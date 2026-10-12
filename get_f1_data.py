"""Download lap-by-lap race data for 2023-2026 with FastF1 into one CSV.

    python get_f1_data.py                 # -> data/f1_laps_2023_2026.csv
    python get_f1_data.py --combine-only  # rebuild the CSV from data/raw only

FastF1 allows 500 requests/hour (~25-30 races); the script pauses and resumes
on its own. Races already in data/raw are skipped, so it's safe to rerun.
"""

import os
import sys
import time
from datetime import datetime

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

SEASONS = [2023, 2024, 2025, 2026]   # 2026 = new regulations
RAW_DIR = "data/raw"
CACHE_DIR = "data/cache"
OUT_FILE = "data/f1_laps_2023_2026.csv"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Columns we keep from FastF1's lap table
LAP_COLS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "Position",
    "LapTime", "Time", "LapStartTime",
    "Stint", "Compound", "TyreLife", "FreshTyre",
    "PitInTime", "PitOutTime", "TrackStatus",
    "Deleted", "IsAccurate",
]
WEATHER_COLS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
TIME_COLS = ["LapTime", "Time", "LapStartTime", "PitInTime", "PitOutTime"]


def load_race(year, rnd, event_name):
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps
    df = laps[[c for c in LAP_COLS if c in laps.columns]].copy()

    # Weather at the time of each lap (matched by FastF1)
    try:
        wx = laps.get_weather_data().reset_index(drop=True)
        for c in WEATHER_COLS:
            if c in wx.columns:
                df[c] = wx[c].values
    except Exception as e:
        print(f"    (no weather data: {e})")

    # Timedeltas -> seconds so the CSV is easy to work with
    for c in TIME_COLS:
        if c in df.columns:
            df[c] = df[c].dt.total_seconds()

    df.insert(0, "Year", year)
    df.insert(1, "Round", rnd)
    df.insert(2, "Event", event_name)
    df["TotalLaps"] = int(df["LapNumber"].max())
    return df


RATE_LIMIT_WAIT = 10 * 60   # FastF1 allows 500 requests/hour; pause 10 min when we hit it


def wait_out_rate_limit(fn, *args):
    """Call fn; if FastF1's hourly request limit is hit, pause and try again."""
    while True:
        try:
            return fn(*args)
        except RateLimitExceededError:
            resume = datetime.now() + pd.Timedelta(seconds=RATE_LIMIT_WAIT)
            print(f"    hourly request limit reached; pausing until {resume:%H:%M} "
                  f"(this is normal, leave it running)")
            time.sleep(RATE_LIMIT_WAIT)


def combine():
    """Merge every per-race file in data/raw into one CSV."""
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    if not files:
        print("No races downloaded yet.")
        return
    combined = pd.concat([pd.read_csv(f"{RAW_DIR}/{f}") for f in files], ignore_index=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(combined):,} laps from {len(files)} races -> {OUT_FILE}")
    print(combined.groupby("Year")["Event"].nunique().rename("races per season").to_string())


def main():
    today = pd.Timestamp(datetime.now().date())
    failed = []

    for year in SEASONS:
        schedule = wait_out_rate_limit(fastf1.get_event_schedule, year)
        schedule = schedule[schedule["RoundNumber"] > 0]   # drop pre-season testing
        for _, ev in schedule.iterrows():
            rnd, name = int(ev["RoundNumber"]), ev["EventName"]
            if pd.Timestamp(ev["EventDate"]).tz_localize(None) >= today:
                continue  # race hasn't happened yet

            out = f"{RAW_DIR}/{year}_{rnd:02d}.csv"
            if os.path.exists(out):
                continue  # already downloaded (resume support)

            print(f"{year} R{rnd:02d} {name} ...")
            for attempt in range(3):
                try:
                    wait_out_rate_limit(load_race, year, rnd, name).to_csv(out, index=False)
                    break
                except Exception as e:
                    print(f"    attempt {attempt + 1} failed: {str(e)[:120]}")
                    time.sleep(10)
            else:
                failed.append(f"{year} R{rnd} {name}")

    if failed:
        print("\nFailed (run the script again to retry):", *failed, sep="\n  ")


if __name__ == "__main__":
    if "--combine-only" in sys.argv:
        combine()            # just build the CSV from what's already downloaded
    else:
        try:
            main()
        finally:
            combine()        # always build the CSV, even if the run stops early
