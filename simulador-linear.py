#!/usr/bin/env python3
"""
simulate_match.py  –  One-off Monte-Carlo game simulator
Usage:
    python simulate_match.py "TeamA" "TeamB" "MapName"
If you omit the arguments you’ll get interactive prompts.

Assumes the match dataset is in 'cs2_matches.csv' (semicolon-delimited).
"""

from __future__ import annotations
import sys
from pathlib import Path
import difflib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_results(counts, t1_rates, t2_rates, team1, team2, map_played, outfile="mc_winrate.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(counts, t1_rates, marker="o", label=f"{team1} win‑rate")
    plt.plot(counts, t2_rates, marker="s", label=f"{team2} win‑rate")
    
    for x, y in zip(counts, t1_rates):
        plt.text(x, y + 0.02, f"{y * 100:.2f}%", ha="center", va="bottom", fontsize=8)
    for x, y in zip(counts, t2_rates):
        plt.text(x, y + 0.02, f"{y * 100:.2f}%", ha="center", va="bottom", fontsize=8)
    
    plt.xlabel("Número de simulações")
    plt.ylabel("Taxa de vitória")
    plt.xscale("log")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.title(f"{team1} X {team2} - {map_played}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"[INFO] plot saved to {outfile}", file=sys.stderr)

# ───── configuration ──────────────────────────────────────────────
CSV_PATH   = Path("cs2_matches.csv")   # change if needed
RNG_SEED   = 42
N_DRAWS    = 1000000
PLAYER_METRICS = [
    "Total kills", "K/D Ratio", "Damage / Round",
    "Kills / round", "Assists / round", "Deaths / round",
    "Rating 2.0",
]
STR_COLUMNS_KEEP = {"team1", "team2", "map"}

# ───── get CLI args or prompt ─────────────────────────────────────
if len(sys.argv) == 4:
    team_a, team_b, map_name = sys.argv[1:]
else:
    team_a   = input("Team A name: ").strip()
    team_b   = input("Team B name: ").strip()
    map_name = input("Map (e.g., Mirage): ").strip()

print(f"\nSimulating {team_a!r} vs {team_b!r} on {map_name!r} …\n")

# ───── 1. load & clean dataset ────────────────────────────────────
df = pd.read_csv(
    CSV_PATH,
    sep=";",
    skipinitialspace=True,
    na_values=[""],
).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
df = df.loc[:, ~df.columns.duplicated()]

# convert %-strings → floats (map win-rates + HS%)
pct_cols = [c for c in df.columns
            if c.endswith("mapwinrate") or "Headshot %" in c]
for c in pct_cols:
    df[c] = (
        df[c].astype(str)
             .str.rstrip("%")
             .str.strip()
             .replace("", np.nan)
             .astype("float32") / 100.0
    )

# numeric-coerce *only* true stat columns
for col in df.columns:
    if col in STR_COLUMNS_KEEP:
        # just trim whitespace
        df[col] = df[col].astype(str).str.strip()
    elif df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

df = df[df["team1_win"].isin([0, 1])].copy()

# ───── 2. per-row player averages ─────────────────────────────────
def skill_avg(prefix: str) -> pd.Series:
    cols = [
        f"{prefix}_player{i}_{met}"
        for i in range(5)
        for met in PLAYER_METRICS
        if f"{prefix}_player{i}_{met}" in df.columns
    ]
    if not cols:
        return np.nan
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

df["T1_skill_avg"] = skill_avg("T1")
df["T2_skill_avg"] = skill_avg("T2")
df["skill_diff"]   = df["T1_skill_avg"] - df["T2_skill_avg"]

# ───── 3. build per-team summary table ────────────────────────────
team_rows = []

for side, prefix in [("team1", "T1"), ("team2", "T2")]:
    if f"{prefix}_mapwinrate" not in df.columns:
        continue
    part = df[[side, f"{prefix}_mapwinrate", f"{prefix}_skill_avg"]].rename(
        columns={
            side: "team",
            f"{prefix}_mapwinrate": "mapwinrate",
            f"{prefix}_skill_avg":  "skill_avg",
        }
    )
    team_rows.append(part)

team_means = (
    pd.concat(team_rows, ignore_index=True)
      .groupby("team", dropna=True)
      .agg(mapwinrate=("mapwinrate", "mean"),
           skill_avg =("skill_avg",  "mean"))
)

# helper: canonical spelling or fuzzy suggestion
canon = {t.lower(): t for t in team_means.index}

def match_name(name: str) -> str | None:
    key = name.lower()
    if key in canon:
        return canon[key]
    # fuzzy fallback ≥70 % similarity
    close = difflib.get_close_matches(key, canon.keys(), n=1, cutoff=0.7)
    return canon[close[0]] if close else None

team_a_exact = match_name(team_a)
team_b_exact = match_name(team_b)

missing = [n for n, exact in [(team_a, team_a_exact), (team_b, team_b_exact)]
           if exact is None]
if missing:
    print("Team(s) not found:", ", ".join(missing))
    for m in missing:
        sug = difflib.get_close_matches(m.lower(), canon.keys(), n=3)
        if sug:
            print("  Suggestions:", ", ".join(canon[s] for s in sug))
    sys.exit(1)

team_a, team_b = team_a_exact, team_b_exact

# ───── 4. train match-level model ─────────────────────────────────
feature_cols   = ["T1_mapwinrate", "T2_mapwinrate", "skill_diff"]
numeric_cols   = ["T1_mapwinrate", "T2_mapwinrate", "skill_diff"]
categorical_cols = []

if "map" in df.columns and df["map"].notna().any():
    df["map"] = df["map"].astype(str).fillna("Unknown")
    feature_cols.append("map")
    categorical_cols.append("map")

X = df[feature_cols]
y = df["team1_win"]

transformers = [
    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric_cols)
]
if categorical_cols:
    transformers.append(
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols)
    )

pipe = Pipeline([
    ("prep", ColumnTransformer(transformers)),
    ("clf",  LogisticRegression(max_iter=300, solver="lbfgs")),
]).fit(X, y)

# ───── 5. craft synthetic match row ───────────────────────────────
feat = {
    "T1_mapwinrate": team_means.loc[team_a, "mapwinrate"],
    "T2_mapwinrate": team_means.loc[team_b, "mapwinrate"],
    "skill_diff":    team_means.loc[team_a, "skill_avg"]
                   - team_means.loc[team_b, "skill_avg"],
}
if categorical_cols:
    feat["map"] = map_name

match_df = pd.DataFrame([feat])

print("\nFeature vector used for this match:")
print(match_df.T)
print("\nRaw model coefficients:")
for name, coef in zip(pipe.named_steps["prep"].get_feature_names_out(), 
                      pipe.named_steps["clf"].coef_[0]):
    print(f"{name:20} {coef:+.3f}")

# ───── 6. probability + Monte-Carlo tally ────────────────────────
p = pipe.predict_proba(match_df)[0, 1]          # P(team A wins)
rng = np.random.default_rng(RNG_SEED)
wins = rng.random(N_DRAWS) < p

print(f"Model win-probability for {team_a}: {p:.2%}")

counts = np.logspace(2, np.log10(N_DRAWS), num=10, dtype=int)  # e.g. from 100 to 1,000,000
t1_rates = [wins[:n].mean() for n in counts]
t2_rates = [1 - r for r in t1_rates]

# Print MC result summary
print(f"Monte-Carlo tally over {N_DRAWS:,d} replays:")
print(f"  {team_a:>20} wins: {wins.sum():5d}  ({wins.mean():.2%})")
print(f"  {team_b:>20} wins: {N_DRAWS - wins.sum():5d}  ({1 - wins.mean():.2%})")
print(f"  ± sampling SE  : {wins.std(ddof=1):.2%}")

# Generate plot
plot_results(counts, t1_rates, t2_rates, team_a, team_b, map_name)
