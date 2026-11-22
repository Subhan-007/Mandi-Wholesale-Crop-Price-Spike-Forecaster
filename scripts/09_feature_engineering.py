"""
Phase 2 — Step 2: Feature Engineering & Preprocessing

Reads ml_ready.csv (from Step 1), adds the lag and rolling features listed
in the project PDF, encodes categorical variables, fits a StandardScaler on
training data only, and saves everything needed for model training.

New features added
──────────────────
    price_lag_7d    wholesale price 7 calendar days ago
    price_lag_14d   wholesale price 14 calendar days ago
    roll14_mean     14-day rolling mean price
    season          Pakistani agricultural season (Winter / Spring / Summer / Monsoon)

Daily price spread (max − min)
    Not available — AMIS publishes only ONE price per day (no min/max/usual).
    roll7_std_pct (7-day rolling coefficient of variation) serves as a
    substitute measure of recent price volatility.

Outputs
───────
    data/processed/ml_featured.csv      dataset with all features added
    models/preprocessor.joblib          fitted ColumnTransformer (scaler + encoder)
    models/feature_config.joblib        feature lists, target names, metadata
    data/processed/feature_report.txt   summary log
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- paths (re-use Phase 1 config) -----------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import PROCESSED_DIR  # noqa: E402

ROOT = SCRIPT_DIR.parent
ML_READY_CSV = PROCESSED_DIR / "ml_ready.csv"
ML_FEATURED_CSV = PROCESSED_DIR / "ml_featured.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


# ---- feature definitions ---------------------------------------------------------

# Pakistan agricultural seasons
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Monsoon", 8: "Monsoon",
    9: "Monsoon", 10: "Autumn", 11: "Autumn",
}

NUMERICAL_FEATURES = [
    # price
    "price_pkr_per_100kg", "price_lag_7d", "price_lag_14d",
    # percentage changes
    "pct_chg_1d", "pct_chg_7d", "pct_chg_14d",
    # rolling stats / volatility (roll7_std_pct substitutes for daily price spread)
    "roll7_mean", "roll14_mean", "roll7_std_pct", "vol_14d",
    # cross-city
    "price_vs_city_median",
    # weather
    "rain_mm", "rain_7d_mm", "rain_14d_mm", "days_since_rain",
    "temp_max_c", "temp_min_c", "temp_mean_c", "temp_mean_7d_c", "heat_day",
    # calendar / seasonal
    "month", "day_of_week", "day_of_year",
    "is_ramadan", "days_to_next_eid", "days_since_last_eid", "is_pre_eid_10d",
    # quality flags (may help model down-weight unreliable rows)
    "is_filled", "is_stale",
]

CATEGORICAL_FEATURES = ["city", "crop", "variant", "season"]

CLF_TARGET = "spike_7d"
REG_TARGET = "future_price_7d"


# ---- feature engineering ---------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and seasonal features to the dataset."""
    df = df.sort_values(["city", "crop", "variant", "date"]).copy()

    # 1. Price lags — look up price at exactly (date − N days) in same series
    df["price_lag_7d"] = np.nan
    df["price_lag_14d"] = np.nan
    for _, g in df.groupby(["city", "crop", "variant"]):
        prices = g.set_index("date")["price_pkr_per_100kg"]
        idx = g.index
        dates = g["date"]
        df.loc[idx, "price_lag_7d"] = prices.reindex(
            (dates - pd.Timedelta(days=7)).values
        ).values
        df.loc[idx, "price_lag_14d"] = prices.reindex(
            (dates - pd.Timedelta(days=14)).values
        ).values

    # 2. 14-day rolling mean price
    df["roll14_mean"] = df.groupby(["city", "crop", "variant"])[
        "price_pkr_per_100kg"
    ].transform(lambda s: s.rolling(14, min_periods=7).mean())

    # 3. Season indicator (categorical)
    df["season"] = df["month"].map(SEASON_MAP)

    # 4. Impute price_vs_city_median NaN → 1.0 (neutral: "equals median")
    df["price_vs_city_median"] = df["price_vs_city_median"].fillna(1.0)

    # Round new numerical columns consistently with Phase 1
    for c in ("price_lag_7d", "price_lag_14d", "roll14_mean"):
        df[c] = df[c].round(4)

    return df


# ---- preprocessing ---------------------------------------------------------------

def build_preprocessor(df_train: pd.DataFrame):
    """Build and fit a ColumnTransformer on training data only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    preprocessor.fit(df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    return preprocessor


# ---- main ------------------------------------------------------------------------

def main() -> None:
    import joblib

    say("=" * 70)
    say("Phase 2 — Feature Engineering & Preprocessing")
    say("=" * 70)

    # ------------------------------------------------------------------ load
    df = pd.read_csv(ML_READY_CSV, parse_dates=["date"])
    say(f"\nLoaded {ML_READY_CSV.name}: {len(df):,} rows, {len(df.columns)} columns")

    # ------------------------------------------------------------------ add features
    df = add_features(df)
    say("\nNew features added: price_lag_7d, price_lag_14d, roll14_mean, season")

    for col in ("price_lag_7d", "price_lag_14d", "roll14_mean"):
        n = df[col].notna().sum()
        say(f"  {col:20s} populated: {n:,}/{len(df):,} ({n / len(df):.1%})")
    say(f"  {'season':20s} values:    {sorted(df['season'].unique())}")
    say(f"\n  NOTE: Daily price spread (max − min) is not available.")
    say(f"        AMIS publishes only ONE price per day.")
    say(f"        roll7_std_pct (7-day rolling CV) is used as a substitute.")

    # ------------------------------------------------------------------ model-ready rows
    feature_nans = df[NUMERICAL_FEATURES].isna().any(axis=1)
    clf_ready = ~feature_nans & df[CLF_TARGET].notna()
    reg_ready = ~feature_nans & df[REG_TARGET].notna()

    say(f"\n{'─' * 50}")
    say("Model-ready rows (no NaN in any feature + target present)")
    say(f"{'─' * 50}")
    say(f"  Classification (spike_7d):       {clf_ready.sum():,}")
    say(f"  Regression     (future_price_7d): {reg_ready.sum():,}")
    for split in ("train", "val", "test"):
        m = df["split"] == split
        say(f"    {split:5s}  clf: {(clf_ready & m).sum():>6,}  |  reg: {(reg_ready & m).sum():>6,}")

    # ------------------------------------------------------------------ fit preprocessor
    train_mask = (df["split"] == "train") & clf_ready
    df_train = df[train_mask]
    say(f"\nFitting preprocessor on {len(df_train):,} training rows (observed + filled, with targets)…")

    preprocessor = build_preprocessor(df_train)

    # output feature names
    num_names = list(NUMERICAL_FEATURES)
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    all_output_names = num_names + cat_names

    say(f"  Numerical features:         {len(num_names)}")
    say(f"  Encoded categorical features: {len(cat_names)}")
    say(f"  Total preprocessed features:  {len(all_output_names)}")
    say(f"  Encoded categories: {cat_names}")

    # quick sanity check
    X_test = preprocessor.transform(
        df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].head(5)
    )
    say(f"\n  Preprocessor test: (5, {len(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)}) "
        f"→ {X_test.shape} ✓")

    # ------------------------------------------------------------------ class balance
    say(f"\n{'─' * 50}")
    say("Class balance (classification target: spike_7d)")
    say(f"{'─' * 50}")
    for split in ("train", "val", "test"):
        sub = df[(df["split"] == split) & clf_ready]
        counts = sub[CLF_TARGET].value_counts()
        total = len(sub)
        n_spike = int(counts.get(1.0, 0))
        n_normal = int(counts.get(0.0, 0))
        ratio = n_normal / n_spike if n_spike > 0 else float("inf")
        say(f"  {split:5s}: Normal={n_normal:>5,} | Spike={n_spike:>4,} | "
            f"Ratio={ratio:.1f}:1 | Spike rate={n_spike / total * 100:.1f}%")
    say("  → Class imbalance detected. Use class_weight='balanced' in models.")

    # ------------------------------------------------------------------ scaling summary
    say(f"\n{'─' * 50}")
    say("Scaling summary (StandardScaler, fit on training data only)")
    say(f"{'─' * 50}")
    scaler = preprocessor.named_transformers_["num"]
    for feat, mu, sd in zip(NUMERICAL_FEATURES, scaler.mean_, scaler.scale_):
        say(f"  {feat:30s}  mean={mu:>12.4f}  std={sd:>12.4f}")

    # ------------------------------------------------------------------ save
    df.to_csv(ML_FEATURED_CSV, index=False)
    say(f"\n✓ Wrote {ML_FEATURED_CSV.name}: {len(df):,} rows, {len(df.columns)} columns")

    feature_config = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_output_names": all_output_names,
        "clf_target": CLF_TARGET,
        "reg_target": REG_TARGET,
    }
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(feature_config, MODELS_DIR / "feature_config.joblib")
    say(f"✓ Wrote models/preprocessor.joblib")
    say(f"✓ Wrote models/feature_config.joblib")

    report_path = PROCESSED_DIR / "feature_report.txt"
    report_path.write_text("\n".join(REPORT), encoding="utf-8")
    say(f"✓ Wrote {report_path.name}")


if __name__ == "__main__":
    main()
