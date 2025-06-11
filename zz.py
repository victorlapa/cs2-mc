#!/usr/bin/env python3
"""
bo3_mc_with_kd.py
---------------------------------------------------------------------------
Train a LightGBM model (with per-player K/D Ratio features) and run a Monte-
Carlo simulation of a best-of-3 Counter-Strike 2 series.

Edit ONLY the constants in the USER SETTINGS block.
---------------------------------------------------------------------------
"""

# ========= 1. USER SETTINGS ============================================

TEAM1       = "MIBR"                     # left-hand side team
TEAM2       = "Vitality"                 # right-hand side team
VETO_ORDER  = ["Mirage", "Nuke", "Inferno"]   # map order for the Bo3
CSV_PATH    = "cs2_matches.csv"          # semicolon-separated historical data
MODEL_PATH  = None                       # e.g. "lgbm_cs2.joblib" to reuse
MC_RUNS     = 1_000_000                     # Monte-Carlo iterations

# ======================================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import joblib
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# ----- 2. CONSTANTS ----------------------------------------------------
KD_COL_TMPL = "{}_player{}_K/D Ratio"    # T1_player0_K/D Ratio, …
KD_T1 = [KD_COL_TMPL.format("T1", i) for i in range(5)]
KD_T2 = [KD_COL_TMPL.format("T2", i) for i in range(5)]
KD_ALL = KD_T1 + KD_T2


# ----- 3. DATA UTILITIES ----------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Read the CSV, parse win-rates (%), K/D ratios, and add aggregate cols."""
    df = pd.read_csv(csv_path, sep=";")

    # win-rates -> float %
    for col in ["T1_mapwinrate", "T2_mapwinrate"]:
        df[f"{col}_num"] = (
            df[col].astype(str)
                   .str.rstrip("%")
                   .str.strip()
                   .replace("", np.nan)
                   .astype(float)
        )

    # K/D ratios -> float
    for col in KD_ALL:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # team aggregates
    df["T1_kd_avg"] = df[KD_T1].mean(axis=1)
    df["T2_kd_avg"] = df[KD_T2].mean(axis=1)
    df["kd_diff"]   = df["T1_kd_avg"] - df["T2_kd_avg"]
    df["skill_diff"] = df["T1_mapwinrate_num"] - df["T2_mapwinrate_num"]

    keep = (
        ["team1", "team2", "map", "team1_win"]
        + ["T1_mapwinrate_num", "T2_mapwinrate_num", "skill_diff"]
        + KD_ALL
        + ["T1_kd_avg", "T2_kd_avg", "kd_diff"]
    )
    return df[keep].dropna(subset=["T1_mapwinrate_num", "T2_mapwinrate_num"])


def historic_mean(series: pd.Series, default: float) -> float:
    return float(series.mean()) if not series.empty else default


def build_row(t1: str, t2: str, map_: str, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Create one inference row, using historical averages for any unknown KD.
    """
    # ---------- win-rates ----------------------------------------------
    t1_wr = historic_mean(
        pd.concat([
            hist.loc[(hist["team1"] == t1) & (hist["map"] == map_), "T1_mapwinrate_num"],
            hist.loc[(hist["team2"] == t1) & (hist["map"] == map_), "T2_mapwinrate_num"],
        ]),
        default=50.0,
    )
    t2_wr = historic_mean(
        pd.concat([
            hist.loc[(hist["team1"] == t2) & (hist["map"] == map_), "T1_mapwinrate_num"],
            hist.loc[(hist["team2"] == t2) & (hist["map"] == map_), "T2_mapwinrate_num"],
        ]),
        default=50.0,
    )

    # ---------- K/D ratios ---------------------------------------------
    t1_kds = pd.concat([
        hist.loc[hist["team1"] == t1, KD_T1],
        hist.loc[hist["team2"] == t1, KD_T2],
    ]).stack()
    t1_kd_avg = historic_mean(t1_kds, default=1.0)

    t2_kds = pd.concat([
        hist.loc[hist["team1"] == t2, KD_T1],
        hist.loc[hist["team2"] == t2, KD_T2],
    ]).stack()
    t2_kd_avg = historic_mean(t2_kds, default=1.0)

    row = {
        "team1": t1,
        "team2": t2,
        "map":   map_,
        "T1_mapwinrate_num": t1_wr,
        "T2_mapwinrate_num": t2_wr,
        "skill_diff": t1_wr - t2_wr,
        "T1_kd_avg":  t1_kd_avg,
        "T2_kd_avg":  t2_kd_avg,
        "kd_diff":    t1_kd_avg - t2_kd_avg,
    }

    # fill five per-player KD placeholders with the team average
    for i in range(5):
        row[KD_COL_TMPL.format("T1", i)] = t1_kd_avg
        row[KD_COL_TMPL.format("T2", i)] = t2_kd_avg

    return pd.DataFrame([row])


# ----- 4. MODEL --------------------------------------------------------
def build_pipeline(df: pd.DataFrame) -> tuple[Pipeline, str]:
    label = "team1_win"
    cat_cols = ["team1", "team2", "map"]
    num_cols = (
        ["T1_mapwinrate_num", "T2_mapwinrate_num", "skill_diff"]
        + KD_ALL
        + ["T1_kd_avg", "T2_kd_avg", "kd_diff"]
    )

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    gbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    return Pipeline([("pre", pre), ("gbm", gbm)]), label


def train_or_load(df: pd.DataFrame, path: str | None) -> Pipeline:
    if path and Path(path).exists():
        pipe = joblib.load(path)
        print(f"[INFO] loaded model from {path}")
        return pipe

    pipe, label = build_pipeline(df)
    X_tr, X_val, y_tr, y_val = train_test_split(
        df.drop(columns=[label]),
        df[label],
        test_size=0.2,
        stratify=df[label],
        random_state=42,
    )
    pipe.fit(X_tr, y_tr)
    print(f"[INFO] validation accuracy: {pipe.score(X_val, y_val):.3f}")

    if path:
        joblib.dump(pipe, path)
        print(f"[INFO] model saved to {path}")
    return pipe


# ----- 5. MONTE-CARLO --------------------------------------------------
def simulate_bo3(map_ps: list[float], runs: int) -> Counter:
    req = 2
    rng = np.random.default_rng(42)
    counts = Counter()

    for _ in range(runs):
        t1, t2 = 0, 0
        for p in map_ps:
            t1 += rng.random() < p
            t2 += rng.random() >= p
            if t1 == req or t2 == req:
                break
        counts[f"{t1}-{t2}"] += 1
    return counts


# ----- 6. MAIN FLOW ----------------------------------------------------
def main() -> None:
    df = load_data(CSV_PATH)
    pipe = train_or_load(df, MODEL_PATH)

    # per-map win probabilities for TEAM1
    map_ps = [
        pipe.predict_proba(build_row(TEAM1, TEAM2, m, df))[0, 1]
        for m in VETO_ORDER
    ]

    print("\nPer-map win probabilities for", TEAM1)
    for m, p in zip(VETO_ORDER, map_ps):
        print(f"  {m:<8}: {p:.3%}")

    # Monte-Carlo
    scores = simulate_bo3(map_ps, MC_RUNS)
    print(f"\nMonte-Carlo BO3  ({TEAM1} vs {TEAM2},  {MC_RUNS:,} runs)")
    for score, cnt in sorted(scores.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {score:<3}: {cnt} ({cnt / MC_RUNS:5.2%})")
    print()


if __name__ == "__main__":
    main()
