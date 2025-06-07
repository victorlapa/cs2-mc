#!/usr/bin/env python3
# win-probability.py • 2025-06-07
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

RNG_SEED  = 42
CSV_PATH  = Path("cs2_matches.csv")

# ───────── 1. LOAD ────────────────────────────────────────────────
df = pd.read_csv(
    CSV_PATH,
    sep=";",
    skipinitialspace=True,
    na_values=[""],
)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df = df.loc[:, ~df.columns.duplicated()]

# ───────── 2. PERCENT STRINGS → FLOATS ────────────────────────────
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

# ───────── 3. FORCE NUMERIC FOR TRUE-NUMERIC COLUMNS ──────────────
for col in df.columns:
    if col == "map":                # keep map as string
        continue
    if df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")

df = df[df["team1_win"].isin([0, 1])].copy()

# ───────── 4. FEATURE ENGINEERING ────────────────────────────────
player_metrics = [
    "Total kills", "K/D Ratio", "Damage / Round",
    "Kills / round", "Assists / round", "Deaths / round",
    "Rating 2.0",
]

def team_mean(prefix: str) -> pd.Series:
    cols = [
        f"{prefix}_player{i}_{m}"
        for i in range(5)
        for m in player_metrics
        if f"{prefix}_player{i}_{m}" in df.columns
    ]
    if not cols:
        return np.nan
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

df["T1_skill_avg"] = team_mean("T1")
df["T2_skill_avg"] = team_mean("T2")
df["skill_diff"]   = df["T1_skill_avg"] - df["T2_skill_avg"]

feature_cols   = ["T1_mapwinrate", "T2_mapwinrate", "skill_diff"]
numeric_cols   = ["T1_mapwinrate", "T2_mapwinrate", "skill_diff"]
categorical_cols = []

if "map" in df.columns and df["map"].notna().any():
    df["map"] = df["map"].astype(str).fillna("Unknown")
    feature_cols.append("map")
    categorical_cols.append("map")

X, y = df[feature_cols], df["team1_win"]

# ───────── 5. PREPROCESS + MODEL PIPELINE ────────────────────────
transformers = [
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric_cols)
]

if categorical_cols:            # only add if we truly have cat features
    transformers.append(
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols)
    )

preprocess = ColumnTransformer(transformers)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=200, solver="lbfgs")),
])

cv = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"5-fold ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")
pipe.fit(X, y)

# ───────── 6. MONTE-CARLO SIMULATION ─────────────────────────────
def simulate(row, n=5000, rng=RNG_SEED):
    p = pipe.predict_proba(row[feature_cols].to_frame().T)[0, 1]
    wins = np.random.default_rng(rng).random(n) < p
    return p, wins.mean(), wins.std(ddof=1)

out = []
for _, r in df.iterrows():
    p, mc_mean, mc_se = simulate(r)
    out.append({
        "matchID": r["matchID"],
        "map":     r.get("map", "N/A"),
        "p_model": p,
        "mc_mean": mc_mean,
        "mc_se":   mc_se,
    })

pd.DataFrame(out).to_csv("mc_match_results.csv", index=False)
print("✓ Monte-Carlo results saved to mc_match_results.csv")
