#!/usr/bin/env python3
"""
simulate_match_v3.py    TEAM_A  TEAM_B  MAP1 [MAP2 …]

Example (Bo1 + MC graph):
    python simulate_match_v3.py MIBR Vitality Nuke --counts 5000 10000 500000 1000000

Changes vs v2:
• Added Monte‑Carlo simulation routine for a list of iteration counts.
• Generates a PNG line‑chart (team win‑rates vs number of simulations).
• Optional --counts argument lets you override the default list.
• All core model logic from v2 preserved.
"""

import argparse
import sys
import warnings
from pathlib import Path
import lightgbm as lgb

from matplotlib.ticker import ScalarFormatter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# 1.  DATA HELPERS
# ----------------------------------------------------------------------
KD_COL_TEMPLATE = "{}_player{}_K/D Ratio"  # e.g.  T1_player0_K/D Ratio
KD_COLS_T1 = [KD_COL_TEMPLATE.format("T1", i) for i in range(5)]
KD_COLS_T2 = [KD_COL_TEMPLATE.format("T2", i) for i in range(5)]
KD_COLS_ALL = KD_COLS_T1 + KD_COLS_T2

def load_data(csv_path: str = "cs2_matches.csv") -> pd.DataFrame:
    """Load historical matches, parse win‑rates & K/D ratios, add aggregates."""
    df = pd.read_csv(csv_path, sep=";")

    # map win‑rates -----------------------------------------------------
    for col in ["T1_mapwinrate", "T2_mapwinrate"]:
        df[f"{col}_num"] = (
            df[col]
            .astype(str)
            .str.rstrip("%")
            .str.strip()
            .replace("", np.nan)
            .astype(float)
        )

    # player K/D ratios -------------------------------------------------
    for col in KD_COLS_ALL:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # team‑level stats --------------------------------------------------
    df["T1_kd_avg"] = df[KD_COLS_T1].mean(axis=1)
    df["T2_kd_avg"] = df[KD_COLS_T2].mean(axis=1)
    df["kd_diff"] = df["T1_kd_avg"] - df["T2_kd_avg"]
    df["skill_diff"] = df["T1_mapwinrate_num"] - df["T2_mapwinrate_num"]

    keep_cols = (
        ["team1", "team2", "map", "team1_win"]
        + ["T1_mapwinrate_num", "T2_mapwinrate_num", "skill_diff"]
        + KD_COLS_ALL
        + ["T1_kd_avg", "T2_kd_avg", "kd_diff"]
    )
    return df[keep_cols].dropna(subset=["T1_mapwinrate_num", "T2_mapwinrate_num"])

def build_row(team1: str, team2: str, map_: str, hist: pd.DataFrame) -> pd.DataFrame:
    """Return a single‑row DataFrame with numeric + categorical features."""

    def team_map_wr(team: str) -> float:
        rows = pd.concat([
            hist.loc[(hist["team1"] == team) & (hist["map"] == map_), "T1_mapwinrate_num"],
            hist.loc[(hist["team2"] == team) & (hist["map"] == map_), "T2_mapwinrate_num"],
        ])
        if rows.empty:
            rows = pd.concat([
                hist.loc[hist["team1"] == team, "T1_mapwinrate_num"],
                hist.loc[hist["team2"] == team, "T2_mapwinrate_num"],
            ])
        return float(rows.mean()) if not rows.empty else 50.0

    def team_kd_stats(team: str):
        kd_vals = pd.concat([
            hist.loc[hist["team1"] == team, KD_COLS_T1],
            hist.loc[hist["team2"] == team, KD_COLS_T2],
        ]).stack()
        avg = float(kd_vals.mean()) if not kd_vals.empty else 1.00
        return avg, [avg] * 5

    # collect stats -----------------------------------------------------
    t1_wr, t2_wr = team_map_wr(team1), team_map_wr(team2)
    t1_kd_avg, t1_kds = team_kd_stats(team1)
    t2_kd_avg, t2_kds = team_kd_stats(team2)

    row_dict = {
        "team1": team1,
        "team2": team2,
        "map": map_,
        "T1_mapwinrate_num": t1_wr,
        "T2_mapwinrate_num": t2_wr,
        "skill_diff": t1_wr - t2_wr,
        "T1_kd_avg": t1_kd_avg,
        "T2_kd_avg": t2_kd_avg,
        "kd_diff": t1_kd_avg - t2_kd_avg,
    }
    for i in range(5):
        row_dict[KD_COL_TEMPLATE.format("T1", i)] = t1_kds[i]
        row_dict[KD_COL_TEMPLATE.format("T2", i)] = t2_kds[i]

    return pd.DataFrame([row_dict])

# ----------------------------------------------------------------------
# 2.  MODEL PIPELINE
# ----------------------------------------------------------------------

def build_pipeline(df: pd.DataFrame):
    label = "team1_win"
    cat_cols = ["team1", "team2", "map"]
    num_cols = [
        "T1_mapwinrate_num",
        "T2_mapwinrate_num",
        "skill_diff",
        *KD_COLS_ALL,
        "T1_kd_avg",
        "T2_kd_avg",
        "kd_diff",
    ]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    gbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    return Pipeline([("pre", pre), ("gbm", gbm)]), label


def train_model(df: pd.DataFrame, pipe: Pipeline, label: str):
    X, y = df.drop(columns=[label]), df[label]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_tr, y_tr)
    print(f"[INFO] validation accuracy: {pipe.score(X_val, y_val):.3f}", file=sys.stderr)
    return pipe

# ----------------------------------------------------------------------
# 3.  MONTE‑CARLO SIMULATIONS
# ----------------------------------------------------------------------

def run_mc(prob: float, n: int) -> float:
    """Return win‑rate for TEAM1 over *n* Bernoulli trials with p=prob."""
    wins = np.random.random(n) < prob
    return wins.mean()


def simulate_over_counts(prob: float, counts: list[int]):
    t1_rates = [run_mc(prob, n) for n in counts]
    t2_rates = [1 - r for r in t1_rates]
    return t1_rates, t2_rates


def plot_results(counts, t1_rates, t2_rates, team1, team2, map_played, outfile="mc_winrate.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(counts, t1_rates, marker="o", label=f"{team1} win‑rate")
    plt.plot(counts, t2_rates, marker="s", label=f"{team2} win‑rate")
    
    # Annotate data points with text
    for x, y in zip(counts, t1_rates):
        plt.text(x, y + 0.02, f"{y * 100:.2f}%", ha="center", va="bottom", fontsize=8)
    for x, y in zip(counts, t2_rates):
        plt.text(x, y + 0.02, f"{y * 100:.2f}%", ha="center", va="bottom", fontsize=8)
    
    plt.xlabel("Número de simulações")
    plt.ylabel("Taxa de vitória")
    
    # Use log scale but display ticks as absolute numbers
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

# ----------------------------------------------------------------------
# 4.  CLI ENTRYPOINT
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CS2 Bo1 win‑prob model with MC convergence plot")
    p.add_argument("team1")
    p.add_argument("team2")
    p.add_argument("map", nargs="+", help="Map list (first one used for Bo1)")
    p.add_argument("--data", default="cs2_matches.csv", help="CSV path with historical matches")
    p.add_argument("--model", help="Optional path to .joblib model")
    p.add_argument(
        "--counts",
        nargs="*",
        type=int,
        default=[5000, 10000, 500000, 1000000],
        help="List of MC iteration counts (space‑separated)",
    )
    args = p.parse_args()

    df = load_data(args.data)

    if args.model and Path(args.model).exists():
        pipe: Pipeline = joblib.load(args.model)
        print(f"[INFO] loaded model from {args.model}", file=sys.stderr)
    else:
        pipe, label = build_pipeline(df)
        pipe = train_model(df, pipe, label)
        if args.model:
            joblib.dump(pipe, args.model)
            print(f"[INFO] model saved to {args.model}", file=sys.stderr)

    map_played = args.map[0]
    X_one = build_row(args.team1, args.team2, map_played, df)
    prob = pipe.predict_proba(X_one)[0, 1]

    # Monte‑Carlo simulations ------------------------------------------
    counts = args.counts
    t1_rates, t2_rates = simulate_over_counts(prob, counts)

    # pretty print results ---------------------------------------------
    print(f"\n⟐  Best‑of‑1: {args.team1} vs {args.team2} on {args.map[0]}")
    print(f"   Model win‑prob: {args.team1} {prob:.2%}  |  {args.team2} {(1-prob):.2%}")
    print("\n   MC Convergence:")
    for n, r1 in zip(counts, t1_rates):
        print(f"   {n:>8,} sims →  {args.team1}: {r1:.2%}   {args.team2}: {1-r1:.2%}")

    # plot --------------------------------------------------------------
    plot_results(counts, t1_rates, t2_rates, args.team1, args.team2, args.map[0])


if __name__ == "__main__":
    main()
