#!/usr/bin/env python3
"""
simulate_match_plus.py  –  Bo1/Bo3 Monte-Carlo simulator for Counter-Strike 2.

EXAMPLES
--------

# Bo1 on Mirage
python simulate_match_plus.py "M80" "Team Spirit" Mirage

# Bo3 (map order matters)
python simulate_match_plus.py "M80" "Team Spirit" Mirage Nuke Ancient --bo3

# 50 000-draw Bo3
python simulate_match_plus.py FURIA "Team Liquid" Inferno Vertigo Overpass --bo3 -n 50000
"""

from __future__ import annotations
import sys, difflib, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ───────── CONFIG ──────────────────────────────────────────────────
CSV_PATH        = Path("cs2_matches.csv")
RNG_SEED        = 42
DEFAULT_DRAWS   = 5_000
STR_KEEP        = {"team1", "team2", "map"}
PLAYER_METRICS  = {
    "rating":      "Rating 2.0",
    "kdr":         "K/D Ratio",
    "dmg":         "Damage / Round",
    "kills_pr":    "Kills / round",
    "assists_pr":  "Assists / round",
    "deaths_pr":   "Deaths / round",
}

# ───────── ARGPARSE ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Simulate a Best-of-1 or Best-of-3 CS-2 series using LightGBM.",
    epilog="""
Tips:
  • For Bo3 you must provide at least three maps and add --bo3.
  • Use -n/--draws to change the Monte-Carlo sample size (default 5 000).
""",
    formatter_class=argparse.RawTextHelpFormatter,
)

# positional arguments (optional so we can fall back to prompt)
parser.add_argument("team_a", nargs="?", help="Team A name")
parser.add_argument("team_b", nargs="?", help="Team B name")
parser.add_argument("maps",   nargs="*", help="One map for Bo1, ≥3 maps for Bo3")

# options
parser.add_argument("--bo3", action="store_true",
                    help="Interpret map list as a Best-of-3.")
parser.add_argument("-n", "--draws", type=int, default=DEFAULT_DRAWS,
                    help=f"Monte-Carlo repetitions (default {DEFAULT_DRAWS})")

args = parser.parse_args()

# interactive fallback only if nothing was provided
if args.team_a is None or args.team_b is None or len(args.maps) == 0:
    args.team_a = input("Team A name: ").strip()
    args.team_b = input("Team B name: ").strip()
    maps_input  = input("Map list (comma-separated for Bo1; 3+ for Bo3): ").strip()
    args.maps   = [m.strip() for m in maps_input.split(",") if m.strip()]
    if len(args.maps) >= 3:
        use_bo3 = input("Treat as Bo3? [y/N]: ").lower().startswith("y")
        args.bo3 = use_bo3

if args.bo3 and len(args.maps) < 3:
    sys.exit("‣  --bo3 flag set but fewer than three maps supplied.")

team_a, team_b = args.team_a.strip(), args.team_b.strip()
map_list       = args.maps
n_draws        = args.draws
series_label   = "Best-of-3" if args.bo3 else "Best-of-1"

print(f"\n⟐  {series_label} simulation: {team_a} vs {team_b} on {', '.join(map_list)}\n")

# ───────── 1. LOAD & CLEAN DATA ────────────────────────────────────
df = (pd.read_csv(CSV_PATH, sep=";", skipinitialspace=True, na_values=[""])
        .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")])
df = df.loc[:, ~df.columns.duplicated()]

# %-strings → floats
pct_cols = [c for c in df.columns if c.endswith("mapwinrate") or "Headshot %" in c]
for c in pct_cols:
    df[c] = (df[c].astype(str).str.rstrip("%").str.strip().replace("", np.nan)
                   .astype("float32") / 100.0)

# numeric coercion (leave labels intact)
for col in df.columns:
    if col in STR_KEEP:
        df[col] = df[col].astype(str).str.strip()
    elif df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

df = df[df["team1_win"].isin([0, 1])].copy()

# ───────── 2. PER-ROW METRICS & DIFFS ─────────────────────────────
def row_metric(prefix: str, key: str) -> pd.Series:
    col_name = PLAYER_METRICS[key]
    cols = [f"{prefix}_player{i}_{col_name}" for i in range(5)
            if f"{prefix}_player{i}_{col_name}" in df.columns]
    return df[cols].mean(axis=1, skipna=True)

for key in PLAYER_METRICS:
    df[f"T1_{key}"] = row_metric("T1", key)
    df[f"T2_{key}"] = row_metric("T2", key)
    df[f"{key}_diff"] = df[f"T1_{key}"] - df[f"T2_{key}"]

df["skill_diff"] = df["rating_diff"]  # back-compat alias

# ───────── 3. TEAM-LEVEL AGGREGATES ───────────────────────────────
team_aggs = {}
for key in PLAYER_METRICS:
    s1 = df[["team1", f"T1_{key}"]].rename(columns={"team1": "team", f"T1_{key}": key})
    s2 = df[["team2", f"T2_{key}"]].rename(columns={"team2": "team", f"T2_{key}": key})
    team_aggs[key] = pd.concat([s1, s2]).groupby("team")[key].mean()

map_rates = (
    pd.concat([
        df[["team1", "map", "team1_win"]].rename(
            columns={"team1": "team", "team1_win": "win"}),
        df[["team2", "map", "team1_win"]].rename(
            columns={"team2": "team", "team1_win": "win"})
             .assign(win=lambda d: 1 - d["win"])
    ]).groupby(["team", "map"]).win.mean()
)
overall_wr = (
    pd.concat([
        df[["team1", "team1_win"]].rename(columns={"team1": "team", "team1_win": "win"}),
        df[["team2", "team1_win"]].rename(columns={"team2": "team", "team1_win": "win"})
             .assign(win=lambda d: 1 - d["win"])
    ]).groupby("team").win.mean()
)

# ───────── 4. TRAIN MATCH-LEVEL MODEL ─────────────────────────────
feature_cols = (["T1_mapwinrate", "T2_mapwinrate"]
              + [f"{k}_diff" for k in PLAYER_METRICS]
              + ["map"])
num_cols   = [c for c in feature_cols if c != "map"]
cat_cols   = ["map"]

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])),
    ("clf",  LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        random_state=RNG_SEED,
        n_jobs=-1,
    )),
]).fit(df[feature_cols], df["team1_win"])

# ───────── 5. TEAM NAME CANONICALISATION ──────────────────────────
canon = {t.lower(): t for t in overall_wr.index}
def canon_name(name: str) -> str|None:
    if name.lower() in canon:
        return canon[name.lower()]
    close = difflib.get_close_matches(name.lower(), canon.keys(), n=1, cutoff=0.7)
    return canon.get(close[0]) if close else None

for original in (team_a, team_b):
    if canon_name(original) is None:
        sys.exit(f"✖  Team not found in dataset: {original}")

team_a = canon_name(team_a)
team_b = canon_name(team_b)

# ───────── 6. HELPER FNS ──────────────────────────────────────────
def map_wr(team: str, map_: str) -> float:
    try:
        return map_rates.loc[(team, map_)]
    except KeyError:
        return overall_wr[team]

def feature_row(map_: str) -> pd.DataFrame:
    row = {
        "T1_mapwinrate": map_wr(team_a, map_),
        "T2_mapwinrate": map_wr(team_b, map_),
        "map": map_,
    }
    for k in PLAYER_METRICS:
        row[f"{k}_diff"] = team_aggs[k].get(team_a, np.nan) \
                         - team_aggs[k].get(team_b, np.nan)
    return pd.DataFrame([row])

def map_prob(map_: str) -> float:
    return pipe.predict_proba(feature_row(map_))[0, 1]

rng = np.random.default_rng(RNG_SEED)

# ───────── 7. SIMULATION ──────────────────────────────────────────
if not args.bo3:                                 # Bo1
    p = map_prob(map_list[0])
    wins = rng.random(n_draws) < p
    print(f"Model P({team_a} wins map): {p:.2%}")
    print(f"Monte-Carlo tally (n={n_draws:,d} maps):")
    print(f"  {team_a:>20} wins {wins.sum():5d}×  ({wins.mean():.2%})")
    print(f"  {team_b:>20} wins {n_draws - wins.sum():5d}×  ({1 - wins.mean():.2%})")
    print(f"  sampling SE ±{wins.std(ddof=1):.2%}")

else:                                            # Bo3
    probs = [map_prob(m) for m in map_list[:3]]
    a_series_wins = 0
    for _ in range(n_draws):
        a, b = 0, 0
        for p in probs:          # play maps in order
            a += rng.random() < p
            b += 1 - (rng.random() < p)
            if a == 2 or b == 2:
                break
        if a > b:
            a_series_wins += 1
    series_p = a_series_wins / n_draws
    se = np.sqrt(series_p * (1 - series_p) / n_draws)

    print("Per-map P(Team A wins):",
          ", ".join(f"{m}:{p:.2%}" for m, p in zip(map_list[:3], probs)))
    print(f"\nSeries win-probability for {team_a}: {series_p:.2%}")
    print(f"Monte-Carlo tally (n={n_draws:,d} series):")
    print(f"  {team_a:>20} wins {a_series_wins:5d}×  ({series_p:.2%})")
    print(f"  {team_b:>20} wins {n_draws - a_series_wins:5d}×  ({1 - series_p:.2%})")
    print(f"  sampling SE ±{se:.2%}")
