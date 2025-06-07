#!/usr/bin/env python3
"""
simulate_match_plus.py  –  Map-aware Monte-Carlo simulator with richer features
Usage:
    python simulate_match_plus.py "Team A" "Team B" "Mirage"
If you omit arguments, the script asks interactively.

Needs:
    pip install numpy pandas scikit-learn lightgbm
"""

from __future__ import annotations
import sys, difflib
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ───────── CONFIG ──────────────────────────────────────────────────
CSV_PATH        = Path("cs2_matches.csv")
RNG_SEED        = 42
N_DRAWS         = 5_000
STR_KEEP        = {"team1", "team2", "map"}
PLAYER_METRICS  = {
    "rating":      "Rating 2.0",
    "kdr":         "K/D Ratio",
    "dmg":         "Damage / Round",
    "kills_pr":    "Kills / round",
    "assists_pr":  "Assists / round",
    "deaths_pr":   "Deaths / round",
}

# ───────── CLI ARGS / PROMPT ───────────────────────────────────────
if len(sys.argv) == 4:
    team_a, team_b, map_name = sys.argv[1:]
else:
    team_a   = input("Team A name: ").strip()
    team_b   = input("Team B name: ").strip()
    map_name = input("Map name: ").strip()

print(f"\n⟐  Simulating '{team_a}' vs '{team_b}' on '{map_name}'\n")

# ───────── 1. LOAD & BASIC CLEANING ────────────────────────────────
df = (
    pd.read_csv(CSV_PATH, sep=";", skipinitialspace=True, na_values=[""])
      .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
)
df = df.loc[:, ~df.columns.duplicated()]

# %-strings → floats
pct_cols = [c for c in df.columns
            if c.endswith("mapwinrate") or "Headshot %" in c]
for c in pct_cols:
    df[c] = (
        df[c].astype(str)
             .str.rstrip("%").str.strip().replace("", np.nan)
             .astype("float32") / 100.0
    )

# numeric-coerce everything except team/map labels
for col in df.columns:
    if col in STR_KEEP:
        df[col] = df[col].astype(str).str.strip()
    elif df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

df = df[df["team1_win"].isin([0, 1])].copy()

# ───────── 2. PER-ROW METRIC DIFFS FOR MODEL TRAINING ──────────────
def row_metric(prefix: str, metric_name: str) -> pd.Series:
    pat = PLAYER_METRICS[metric_name]
    cols = [f"{prefix}_player{i}_{pat}" for i in range(5)
            if f"{prefix}_player{i}_{pat}" in df.columns]
    return df[cols].mean(axis=1, skipna=True)

for metric in PLAYER_METRICS:
    df[f"T1_{metric}"] = row_metric("T1", metric)
    df[f"T2_{metric}"] = row_metric("T2", metric)
    df[f"{metric}_diff"] = df[f"T1_{metric}"] - df[f"T2_{metric}"]

df["skill_diff"] = df["rating_diff"]  # alias for backward compat

# ───────── 3. TEAM-LEVEL AGGREGATES (for future sims) ──────────────
team_aggs = {}
for metric in PLAYER_METRICS:
    # gather both sides, rename to common cols
    side1 = df[["team1", f"T1_{metric}"]].rename(
        columns={"team1": "team", f"T1_{metric}": metric})
    side2 = df[["team2", f"T2_{metric}"]].rename(
        columns={"team2": "team", f"T2_{metric}": metric})
    team_aggs[metric] = (
        pd.concat([side1, side2])
          .groupby("team", dropna=True)[metric].mean())

# map-specific win rate table
map_rates = (
    pd.concat([
        df[["team1", "map", "team1_win"]].rename(
            columns={"team1": "team", "team1_win": "win"}),
        df[["team2", "map", "team1_win"]].rename(
            columns={"team2": "team", "team1_win": "win"})
            .assign(win=lambda d: 1 - d["win"])
    ])
    .groupby(["team", "map"]).win.mean()
)

# overall win rate (fallback)
overall_wr = (
    pd.concat([
        df[["team1", "team1_win"]].rename(columns={"team1": "team", "team1_win": "win"}),
        df[["team2", "team1_win"]].rename(columns={"team2": "team", "team1_win": "win"})
             .assign(win=lambda d: 1 - d["win"])
    ])
    .groupby("team").win.mean()
)

# ───────── 4. MODEL TRAINING DATA ─────────────────────────────────
feature_cols = ["T1_mapwinrate", "T2_mapwinrate"] \
             + [f"{m}_diff" for m in PLAYER_METRICS] \
             + ["map"]
numeric_cols = [c for c in feature_cols if c != "map"]
categorical_cols = ["map"]

X = df[feature_cols]
y = df["team1_win"]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

model = LGBMClassifier(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    random_state=RNG_SEED,
    n_jobs=-1,
)
pipe = Pipeline([("prep", preprocess), ("clf", model)]).fit(X, y)

# ───────── 5. CANONICAL TEAM NAME LOOKUP ──────────────────────────
canon = {t.lower(): t for t in overall_wr.index}

def canon_name(name: str) -> str | None:
    if name.lower() in canon:
        return canon[name.lower()]
    close = difflib.get_close_matches(name.lower(), canon.keys(), n=1, cutoff=0.7)
    return canon.get(close[0]) if close else None

team_a_c = canon_name(team_a)
team_b_c = canon_name(team_b)
missing = [n for n, c in [(team_a, team_a_c), (team_b, team_b_c)] if c is None]
if missing:
    print("Team(s) not in dataset:", ", ".join(missing))
    sys.exit(1)
team_a, team_b = team_a_c, team_b_c

# ───────── 6. BUILD SYNTHETIC MATCH FEATURES ──────────────────────
def get_mapwr(team: str, map_: str) -> float:
    try:
        return map_rates.loc[(team, map_)]
    except KeyError:
        return overall_wr[team]

feat = {
    "T1_mapwinrate": get_mapwr(team_a, map_name),
    "T2_mapwinrate": get_mapwr(team_b, map_name),
    "map": map_name,
}
for metric in PLAYER_METRICS:
    diff = team_aggs[metric].get(team_a, np.nan) - team_aggs[metric].get(team_b, np.nan)
    feat[f"{metric}_diff"] = diff

match_df = pd.DataFrame([feat])

# ───────── 7. PREDICT & MONTE-CARLO ───────────────────────────────
p = pipe.predict_proba(match_df)[0, 1]
rng = np.random.default_rng(RNG_SEED)
wins = rng.random(N_DRAWS) < p

print(f"Model win-probability for {team_a}: {p:.2%}")
print(f"Monte-Carlo (n={N_DRAWS:,}):")
print(f"  {team_a:>20} wins {wins.sum():4d}×  ({wins.mean():.2%})")
print(f"  {team_b:>20} wins {N_DRAWS - wins.sum():4d}×  ({1 - wins.mean():.2%})")
print(f"  sampling SE ±{wins.std(ddof=1):.2%}")
