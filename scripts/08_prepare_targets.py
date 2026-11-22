"""
Phase 2 — Step 1: Target Variable Preparation

Reads the Phase 1 dataset (mandi_dataset.csv), adds the regression target
(future_price_7d), validates that the classification target (spike_7d) and
spike thresholds meet the PDF spec, and writes a modelling-ready file.

Existing Phase 1 columns used as-is (no recalculation):
    future_chg_7d   = price(t+7) / price(t) - 1
    spike_threshold = mean + 1.5*std of future_chg_7d, per crop-variant,
                      computed on training-period observed rows only
    spike_7d        = 1 if future_chg_7d > spike_threshold, else 0
    split           = train / val / test  (70 / 15 / 15 by date)

New column:
    future_price_7d = the actual wholesale price 7 calendar days ahead
                      (the regression target per the project PDF)

Output:
    data/processed/ml_ready.csv    — the modelling-ready dataset
    data/processed/target_report.txt — validation summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- paths (re-use Phase 1 config) -----------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import PROCESSED_DIR, FINAL_DATASET_CSV  # noqa: E402

ML_READY_CSV = PROCESSED_DIR / "ml_ready.csv"
TARGET_REPORT = PROCESSED_DIR / "target_report.txt"

REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


# ---- main -----------------------------------------------------------------------
def main() -> None:
    say("=" * 70)
    say("Phase 2 — Target Variable Preparation")
    say("=" * 70)

    # 1. Load Phase 1 dataset
    df = pd.read_csv(FINAL_DATASET_CSV, parse_dates=["date"])
    say(f"\nLoaded {FINAL_DATASET_CSV.name}: {len(df)} rows, {len(df.columns)} columns")

    # 2. Add regression target: future_price_7d
    #    = price_pkr_per_100kg of the same city-crop-variant, 7 calendar days later
    #    Computed by shifting inside each series, matching Phase 1's approach.
    df = df.sort_values(["city", "crop", "variant", "date"]).reset_index(drop=True)
    df["future_price_7d"] = np.nan

    for key, g in df.groupby(["city", "crop", "variant"]):
        idx = g.index
        price = g.set_index("date")["price_pkr_per_100kg"]
        # Build a lookup of price by date, then map date+7 -> price
        future = price.reindex(g["date"] + pd.Timedelta(days=7))
        df.loc[idx, "future_price_7d"] = future.values

    n_reg = df["future_price_7d"].notna().sum()
    say(f"\nRegression target (future_price_7d) populated: "
        f"{n_reg}/{len(df)} rows ({n_reg/len(df):.1%})")

    # 3. Validate: future_chg_7d should equal future_price_7d / price - 1
    has_both = df["future_price_7d"].notna() & df["future_chg_7d"].notna()
    if has_both.any():
        recomputed = df.loc[has_both, "future_price_7d"] / df.loc[has_both, "price_pkr_per_100kg"] - 1
        diff = (recomputed - df.loc[has_both, "future_chg_7d"]).abs()
        say(f"\nConsistency check (future_price_7d / price - 1  vs  future_chg_7d):")
        say(f"  Max absolute difference: {diff.max():.6f}")
        say(f"  Mean absolute difference: {diff.mean():.8f}")
        if diff.max() < 0.001:
            say("  ✓ Regression and classification targets are consistent.")
        else:
            say("  ⚠ Some rows show discrepancy > 0.001; check data alignment.")

    # 4. Validate spike_threshold was computed from train-only data
    #    (reproduce the calculation and compare)
    train_obs = df[(df["split"] == "train") & (df["is_filled"] == 0)]
    thr_check = train_obs.groupby(["crop", "variant"])["future_chg_7d"].agg(["mean", "std"])
    thr_check["threshold_recomputed"] = thr_check["mean"] + 1.5 * thr_check["std"]
    thr_original = df.groupby(["crop", "variant"])["spike_threshold"].first()
    comparison = pd.DataFrame({
        "original": thr_original,
        "recomputed": thr_check["threshold_recomputed"],
    })
    comparison["match"] = (comparison["original"] - comparison["recomputed"]).abs() < 0.001
    say(f"\nSpike threshold validation (train-only, observed-only, mean + 1.5×std):")
    say(comparison.to_string())
    if comparison["match"].all():
        say("  ✓ All spike thresholds match: computed from training data only.")
    else:
        say("  ⚠ Mismatch detected; thresholds may use non-training data.")

    # 5. Validate spike_7d label
    expected_spike = np.where(
        df["future_chg_7d"].isna(), np.nan,
        (df["future_chg_7d"] > df["spike_threshold"]).astype(float)
    )
    mismatch = ((df["spike_7d"].values != expected_spike)
                & ~(np.isnan(df["spike_7d"].values) & np.isnan(expected_spike)))
    say(f"\nSpike label validation (future_chg_7d > threshold):")
    say(f"  Mismatched rows: {int(mismatch.sum())} / {len(df)}")
    if int(mismatch.sum()) == 0:
        say("  ✓ All spike labels are correct.")

    # 6. Date alignment check: future_price_7d date is exactly +7 days
    sample = df[df["future_price_7d"].notna()].head(5)
    say(f"\nDate alignment sample (first 5 rows with future price):")
    say(sample[["date", "city", "crop", "variant",
                "price_pkr_per_100kg", "future_price_7d", "future_chg_7d", "spike_7d"]].to_string(index=False))

    # 7. Summary statistics
    say(f"\n{'=' * 70}")
    say("TARGET SUMMARY")
    say(f"{'=' * 70}")

    say(f"\nClassification target (spike_7d):")
    for split in ["train", "val", "test"]:
        sub = df[(df["split"] == split) & df["spike_7d"].notna()]
        counts = sub["spike_7d"].value_counts().to_dict()
        total = len(sub)
        spike_pct = counts.get(1.0, 0) / total * 100 if total > 0 else 0
        say(f"  {split:5s}: {total:>6d} rows | "
            f"Normal={int(counts.get(0.0, 0)):>5d} | "
            f"Spike={int(counts.get(1.0, 0)):>4d} | "
            f"Spike rate={spike_pct:.1f}%")

    say(f"\nRegression target (future_price_7d):")
    for split in ["train", "val", "test"]:
        sub = df[(df["split"] == split) & df["future_price_7d"].notna()]
        say(f"  {split:5s}: {len(sub):>6d} rows | "
            f"mean={sub['future_price_7d'].mean():>10.1f} | "
            f"std={sub['future_price_7d'].std():>9.1f} | "
            f"min={sub['future_price_7d'].min():>8.1f} | "
            f"max={sub['future_price_7d'].max():>9.1f}")

    say(f"\nRows with NaN targets (last 7 days of each series → no future price):")
    say(f"  future_chg_7d  NaN: {int(df['future_chg_7d'].isna().sum())}")
    say(f"  future_price_7d NaN: {int(df['future_price_7d'].isna().sum())}")
    say(f"  spike_7d       NaN: {int(df['spike_7d'].isna().sum())}")

    # 8. Write outputs
    # Round new column consistently with Phase 1
    df["future_price_7d"] = df["future_price_7d"].round(4)

    # Reorder columns: insert future_price_7d right after future_chg_7d
    cols = list(df.columns)
    cols.remove("future_price_7d")
    pos = cols.index("future_chg_7d") + 1
    cols.insert(pos, "future_price_7d")
    df = df[cols]

    df.to_csv(ML_READY_CSV, index=False)
    say(f"\n✓ Wrote {ML_READY_CSV.name}: {len(df)} rows, {len(df.columns)} columns")

    TARGET_REPORT.write_text("\n".join(REPORT), encoding="utf-8")
    say(f"✓ Wrote {TARGET_REPORT.name}")


if __name__ == "__main__":
    main()
