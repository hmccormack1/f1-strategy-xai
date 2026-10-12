"""Turn raw FastF1 laps into a modeling table.

One row per driver per lap, using only what the pit wall knows at that point.
Label: does the driver pit within the next HORIZON laps?

In-laps are dropped (their timing gives the stop away). Retirements and
red-flag tyre changes don't count as pit stops.
"""
import pandas as pd

HORIZON = 3
GAP_CAP = 60.0                   # s; anything bigger is strategically the same
TEMP_CHANGE_RANGE = (-6.0, 4.0)  # ~5th-95th pct; larger swings are a few rain races

COMPOUND_CODE = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
DRY = ["SOFT", "MEDIUM", "HARD"]

FEATURES = {
    "tyre_life":          "Laps on current tyres",
    "compound":           "Tyre compound (1=Soft, 2=Medium, 3=Hard, 4=Inter, 5=Wet)",
    "race_progress":      "Fraction of race completed",
    "laps_remaining":     "Laps remaining",
    "stint":              "Current stint number",
    "has_two_dry":        "Already used two dry compounds (rule satisfied)",
    "position":           "Race position",
    "gap_ahead":          "Gap to car ahead (s)",
    "gap_behind":         "Gap to car behind (s)",
    "safety_car":         "Safety car this lap",
    "vsc":                "Virtual safety car this lap",
    "sc_upcoming":        "Safety car / VSC out on the driver's next lap",
    "yellow":             "Yellow flag this lap",
    "pace_delta":         "Recent pace vs. best of stint (s)",
    "field_pitted_last2": "Other cars pitting in last 2 laps",
    "ahead_pitted":       "Car that was ahead pitted in last 2 laps",
    "temp_change":        "Track temperature change since race start (°C)",
    "rainfall":           "Rain falling",
}
FEATURE_COLS = list(FEATURES)
SHORT_NAMES = {
    "tyre_life": "Tyre age", "compound": "Compound", "race_progress": "Race progress",
    "laps_remaining": "Laps remaining", "stint": "Stint", "has_two_dry": "Two compounds used",
    "position": "Position", "gap_ahead": "Gap ahead", "gap_behind": "Gap behind",
    "safety_car": "Safety car", "vsc": "VSC", "sc_upcoming": "SC/VSC deployed", "yellow": "Yellow flag",
    "pace_delta": "Pace drop-off", "field_pitted_last2": "Others pitting",
    "ahead_pitted": "Car ahead pitted", "temp_change": "Track temp change",
    "track_temp": "Track temp (absolute)", "rainfall": "Rain",
}


def _status_has(series, codes):
    s = series.fillna("").astype(str)
    return s.apply(lambda v: any(c in v for c in codes)).astype(int)


def build_features(raw: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["LapNumber"].notna() & df["Driver"].notna()]
    df["race_id"] = df["Year"].astype(int).astype(str) + "_" + df["Round"].astype(int).map("{:02d}".format)
    df = df.sort_values(["race_id", "Driver", "LapNumber"]).reset_index(drop=True)
    key = ["race_id", "Driver"]

    # FastF1 track status codes: 2 yellow, 4 SC, 5 red, 6/7 VSC
    df["safety_car"] = _status_has(df["TrackStatus"], ["4"])
    df["vsc"] = _status_has(df["TrackStatus"], ["6", "7"])
    df["yellow"] = _status_has(df["TrackStatus"], ["2"])
    df["red_flag"] = _status_has(df["TrackStatus"], ["5"])

    # a pit stop = an in-lap followed by an out-lap
    df["is_inlap"] = df["PitInTime"].notna()
    g = df.groupby(key, sort=False)
    continued = g["PitOutTime"].shift(-1).notna()
    df["pit_stop"] = (df["is_inlap"] & continued & (df["red_flag"] == 0)).astype(int)

    g = df.groupby(key, sort=False)
    future = sum(g["pit_stop"].shift(-i).fillna(0) for i in range(1, horizon + 1))
    df["label"] = (future > 0).astype(int)

    df["tyre_life"] = df["TyreLife"]
    df["compound"] = df["Compound"].map(COMPOUND_CODE)
    df["stint"] = df["Stint"]
    df["race_progress"] = df["LapNumber"] / df["TotalLaps"]
    df["laps_remaining"] = df["TotalLaps"] - df["LapNumber"]
    for c in DRY:
        df[f"_used_{c}"] = (df["Compound"] == c).astype(int).groupby([df[k] for k in key]).cummax()
    df["has_two_dry"] = (df[[f"_used_{c}" for c in DRY]].sum(axis=1) >= 2).astype(int)

    # gaps: order cars by session time at the end of each lap
    lap_key = ["race_id", "LapNumber"]
    df = df.sort_values(lap_key + ["Time"])
    lg = df.groupby(lap_key, sort=False)["Time"]
    df["gap_ahead"] = (df["Time"] - lg.shift(1)).fillna(GAP_CAP).clip(0, GAP_CAP)
    df["gap_behind"] = (lg.shift(-1) - df["Time"]).fillna(GAP_CAP).clip(0, GAP_CAP)
    df["position"] = df["Position"].fillna(lg.rank(method="first"))
    df["_ahead_driver"] = df.groupby(lap_key, sort=False)["Driver"].shift(1)

    # pace drop-off: last 3 clean laps vs. best clean lap of the stint
    df = df.sort_values(key + ["LapNumber"]).reset_index(drop=True)
    clean = (df["TrackStatus"].astype(str) == "1") & ~df["is_inlap"] & df["PitOutTime"].isna()
    df["_clean_lt"] = df["LapTime"].where(clean)
    sk = key + ["Stint"]
    best = df.groupby(sk)["_clean_lt"].cummin()
    recent = df.groupby(sk)["_clean_lt"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["pace_delta"] = recent - best
    df["pace_delta"] = df.groupby(sk)["pace_delta"].ffill().fillna(0).clip(-2, 10)

    stops = df.groupby(lap_key)["pit_stop"].sum()
    field = stops.groupby(level=0).transform(lambda s: s.rolling(2, min_periods=1).sum())
    df["field_pitted_last2"] = df.set_index(lap_key).index.map(field).values
    df["field_pitted_last2"] = df["field_pitted_last2"] - df["pit_stop"]  # exclude self

    # undercut cue: did the car ahead (as of 2 laps ago) just pit?
    df["_ahead_2ago"] = df.groupby(key)["_ahead_driver"].shift(2)
    pit_lookup = df.set_index(["race_id", "Driver", "LapNumber"])["pit_stop"]
    def pitted(row_ahead, race, lap):
        return sum(pit_lookup.get((race, row_ahead, lap - i), 0) for i in (0, 1))
    mask = df["_ahead_2ago"].notna()
    df["ahead_pitted"] = 0
    df.loc[mask, "ahead_pitted"] = [
        int(pitted(a, r, l) > 0) for a, r, l in
        zip(df.loc[mask, "_ahead_2ago"], df.loc[mask, "race_id"], df.loc[mask, "LapNumber"])]

    # SC/VSC on the driver's next lap: the pit wall knows before pit entry.
    # Must be the driver's own next lap. Matching lap numbers across the field
    # leaks the future (a backmarker's lap 28 happens after the leader's).
    caution = (df["safety_car"] | df["vsc"]).astype(int)
    df["sc_upcoming"] = caution.groupby([df[k] for k in key]).shift(-1).fillna(0).astype(int)

    df["track_temp"] = df["TrackTemp"]
    df["rainfall"] = df["Rainfall"].fillna(False).astype(str).str.lower().isin(["true", "1"]).astype(int)

    out = df[~df["is_inlap"]].copy()
    out = out.dropna(subset=["tyre_life", "compound"])
    # absolute temp mostly identifies the race, so the model gets the change since
    # the start; absolute is kept only for the design-iteration experiment
    out["track_temp"] = out.groupby("race_id")["track_temp"].transform(lambda s: s.ffill().bfill())
    out["track_temp"] = out["track_temp"].fillna(out["track_temp"].median())
    first = out.sort_values("LapNumber").groupby("race_id")["track_temp"].first()
    out["temp_change"] = (out["track_temp"] - out["race_id"].map(first)).clip(*TEMP_CHANGE_RANGE)

    keep = ["race_id", "Year", "Round", "Event", "Driver", "Team", "LapNumber",
            "TotalLaps", "pit_stop", "label", "track_temp"] + [c for c in FEATURE_COLS if c != "track_temp"]
    return out[keep].reset_index(drop=True)


def pit_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Actual pit stops: race_id, Driver, in-lap number."""
    df = raw.sort_values(["Year", "Round", "Driver", "LapNumber"]).copy()
    df["race_id"] = df["Year"].astype(int).astype(str) + "_" + df["Round"].astype(int).map("{:02d}".format)
    nxt = df.groupby(["race_id", "Driver"])["PitOutTime"].shift(-1).notna()
    red = _status_has(df["TrackStatus"], ["5"]).astype(bool)
    stops = df[df["PitInTime"].notna() & nxt & ~red]
    return stops[["race_id", "Driver", "LapNumber"]].reset_index(drop=True)
