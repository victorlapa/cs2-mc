#!/usr/bin/env python3
"""
simulate_match_plus.py  –  Bo1 / Bo3 Monte-Carlo simulator **with graphs**.

# Bo1
python simulate_match_plus.py "M80" "Team Spirit" Mirage
# Bo3
python simulate_match_plus.py "M80" "Team Spirit" Mirage Nuke Ancient --bo3
"""

from __future__ import annotations
import sys, difflib, argparse, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    description="Simulate a Best-of-1 or Best-of-3 CS-2 series and plot results.")
parser.add_argument("team_a", nargs="?", help="Team A")
parser.add_argument("team_b", nargs="?", help="Team B")
parser.add_argument("maps",   nargs="*", help="Maps (1 for Bo1 | 3 for Bo3)")
parser.add_argument("--bo3", action="store_true", help="Treat as Best-of-3")
parser.add_argument("-n", "--draws", type=int,
                    default=DEFAULT_DRAWS, help="Monte-Carlo draws")
args = parser.parse_args()

# interactive fallback
if args.team_a is None or args.team_b is None or len(args.maps) == 0:
    args.team_a = input("Team A: ").strip()
    args.team_b = input("Team B: ").strip()
    args.maps   = [m.strip() for m in input("Map list (comma-sep): ").split(",") if m.strip()]
    if len(args.maps) >= 3:
        args.bo3 = input("Treat as Bo3? [y/N]: ").lower().startswith("y")

if args.bo3 and len(args.maps) < 3:
    sys.exit("‣ --bo3 flag set but fewer than 3 maps supplied.")

team_a, team_b = args.team_a.strip(), args.team_b.strip()
map_list       = args.maps
n_draws        = args.draws
series_label   = "Best-of-3" if args.bo3 else "Best-of-1"

print(f"\n⟐  {series_label}: {team_a} vs {team_b} on {', '.join(map_list)}\n")

# ───────── 1. LOAD & CLEAN ────────────────────────────────────────
df = (pd.read_csv(CSV_PATH, sep=";", skipinitialspace=True, na_values=[""])
        .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")])
df = df.loc[:, ~df.columns.duplicated()]

pct_cols = [c for c in df.columns if c.endswith("mapwinrate") or "Headshot %" in c]
for c in pct_cols:
    df[c] = (df[c].astype(str).str.rstrip("%").str.strip().replace("", np.nan)
                   .astype("float32") / 100.0)

for col in df.columns:
    if col in STR_KEEP:
        df[col] = df[col].astype(str).str.strip()
    elif df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

df = df[df["team1_win"].isin([0, 1])].copy()

# ───────── 2. PER-ROW METRICS & DIFFS ─────────────────────────────
def row_metric(prefix: str, key: str) -> pd.Series:
    pat = PLAYER_METRICS[key]
    cols = [f"{prefix}_player{i}_{pat}" for i in range(5)
            if f"{prefix}_player{i}_{pat}" in df.columns]
    return df[cols].mean(axis=1, skipna=True)

for k in PLAYER_METRICS:
    df[f"T1_{k}"] = row_metric("T1", k)
    df[f"T2_{k}"] = row_metric("T2", k)
    df[f"{k}_diff"] = df[f"T1_{k}"] - df[f"T2_{k}"]

df["skill_diff"] = df["rating_diff"]

# ───────── 3. TEAM-LEVEL AGGREGATES ───────────────────────────────
team_aggs = {}
for k in PLAYER_METRICS:
    s1 = df[["team1", f"T1_{k}"]].rename(columns={"team1":"team", f"T1_{k}":k})
    s2 = df[["team2", f"T2_{k}"]].rename(columns={"team2":"team", f"T2_{k}":k})
    team_aggs[k] = pd.concat([s1,s2]).groupby("team")[k].mean()

map_rates = (
    pd.concat([
        df[["team1","map","team1_win"]].rename(columns={"team1":"team","team1_win":"win"}),
        df[["team2","map","team1_win"]].rename(columns={"team2":"team","team1_win":"win"})
          .assign(win=lambda d:1-d.win)
    ]).groupby(["team","map"]).win.mean()
)
overall_wr = (
    pd.concat([
        df[["team1","team1_win"]].rename(columns={"team1":"team","team1_win":"win"}),
        df[["team2","team1_win"]].rename(columns={"team2":"team","team1_win":"win"})
          .assign(win=lambda d:1-d.win)
    ]).groupby("team").win.mean()
)

# ───────── 4. TRAIN MODEL ─────────────────────────────────────────
feature_cols = ["T1_mapwinrate","T2_mapwinrate","skill_diff"] \
             + [f"{k}_diff" for k in PLAYER_METRICS] + ["map"]
num_cols = [c for c in feature_cols if c != "map"]
cat_cols = ["map"]

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])
clf = LGBMClassifier(
        objective="binary", n_estimators=400,
        learning_rate=0.05, random_state=RNG_SEED, n_jobs=-1)
pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(df[feature_cols], df["team1_win"])

# ───── plot feature importance with readable names ─────
raw_names = pipe.named_steps["prep"].get_feature_names_out()
fi_vals   = pipe.named_steps["clf"].booster_.feature_importance()

# clean names: strip the transformer prefixes (“num__”, “cat__oh__”)
def clean(n:str)->str:
    n = re.sub(r"^num__", "", n)
    n = re.sub(r"^cat__[0-9]+_", "", n)
    n = n.replace("oh__", "")
    return n

fi_series = pd.Series(fi_vals, index=[clean(n) for n in raw_names])
fi_series = fi_series.sort_values().tail(15)  # top-15

plt.figure()
fi_series.plot.barh()
plt.title("Top-15 feature importances")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()

# ───────── 5. NAME CANONICALISATION ───────────────────────────────
canon = {t.lower():t for t in overall_wr.index}
def canon_name(name:str)->str|None:
    if name.lower() in canon:
        return canon[name.lower()]
    close = difflib.get_close_matches(name.lower(), canon.keys(), n=1, cutoff=0.7)
    return canon.get(close[0]) if close else None

for raw in (team_a, team_b):
    if canon_name(raw) is None:
        sys.exit(f"✖ Unknown team: {raw}")
team_a, team_b = canon_name(team_a), canon_name(team_b)

# ───────── 6. HELPERS ─────────────────────────────────────────────
def map_wr(team:str,map_:str)->float:
    try: return map_rates.loc[(team,map_)]
    except KeyError: return overall_wr[team]

def build_row(map_:str)->pd.DataFrame:
    row = {"T1_mapwinrate":map_wr(team_a,map_),
           "T2_mapwinrate":map_wr(team_b,map_),
           "map":map_}
    row.update({f"{k}_diff": team_aggs[k].get(team_a,np.nan) -
                              team_aggs[k].get(team_b,np.nan)
                for k in PLAYER_METRICS})
    return pd.DataFrame([row])

def map_prob(map_:str)->float:
    return pipe.predict_proba(build_row(map_))[0,1]

rng = np.random.default_rng(RNG_SEED)

# ───────── 7. PLOT HELPERS ────────────────────────────────────────
def plot_match_prob(p:float):
    plt.figure()
    plt.bar([team_a, team_b], [p,1-p])
    plt.ylim(0,1)
    plt.ylabel("Win probability")
    plt.title(f"P(win) on {map_list[0]}")
    for i,val in enumerate([p,1-p]):
        plt.text(i,val+0.02,f"{val:.1%}",ha="center")
    plt.tight_layout()
    plt.show()

def plot_bo3_hist(counter:Counter):
    labels, counts = zip(*sorted(counter.items()))
    plt.figure()
    plt.bar(labels, np.array(counts)/n_draws)
    plt.ylabel("Frequency")
    plt.title(f"{team_a} vs {team_b} — Bo3 score distribution")
    plt.tight_layout()
    plt.show()

# ───────── 8. SIMULATION + PLOTS ──────────────────────────────────
if not args.bo3:                                # Bo1
    p = map_prob(map_list[0])
    wins = rng.random(n_draws) < p
    print(f"P({team_a} wins map): {p:.2%}")
    plot_match_prob(p)

else:                                           # Bo3
    probs = [map_prob(m) for m in map_list[:3]]
    counter, a_series = Counter(), 0
    for _ in range(n_draws):
        a=b=0
        for p in probs:
            a += rng.random() < p
            b = 3 - a if (a==2 or b==2) else b+1  # quicker but valid
            if a==2 or b==2: break
        counter[f"{a}-{b}"] += 1
        if a>b: a_series += 1
    series_p = a_series / n_draws
    print("Per-map probs:",
          ", ".join(f"{m}:{p:.2%}" for m,p in zip(map_list[:3],probs)))
    print(f"P({team_a} wins series): {series_p:.2%}")
    plot_match_prob(probs[0])
    plot_bo3_hist(counter)
