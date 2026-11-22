"""
Phase 2 — Step 4: Regression Model Development

Implements and compares two regression models to predict 7-day-ahead crop wholesale price (PKR/100kg):
  1. Linear Regression         (Baseline price prediction)
  2. Random Forest Regressor   (More complex non-linear price prediction)

Per project PDF:
  - Target: future_price_7d (actual wholesale price 7 days after prediction date).
  - Chronological 70% train / 15% validation / 15% test split.
  - Evaluation metrics: MAE, RMSE, R² (and MAPE for intuitive % accuracy).
  - Best model pipeline saved to models/regression_model.joblib for Flask app.

Outputs:
  models/regression_model.joblib          — best regression pipeline
  models/reg_linear_regression.joblib    — linear regression pipeline
  models/reg_random_forest.joblib        — random forest pipeline
  data/processed/regression_report.txt   — detailed evaluation report
  data/processed/regression_metrics.csv  — tabular metrics comparison
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

# ---- paths -----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import PROCESSED_DIR  # noqa: E402

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

ML_FEATURED_CSV = PROCESSED_DIR / "ml_featured.csv"
REPORT_TXT = PROCESSED_DIR / "regression_report.txt"
METRICS_CSV = PROCESSED_DIR / "regression_metrics.csv"

REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


def calc_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_split(y_true, y_pred, split_name="val") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = calc_rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "split": split_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape * 100.0,
    }


def main() -> None:
    say("=" * 75)
    say("Phase 2 — Regression Model Development (7-Day Future Price Forecaster)")
    say("=" * 75)

    # 1. Load dataset & configuration
    df = pd.read_csv(ML_FEATURED_CSV, parse_dates=["date"])
    feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")
    base_preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    num_cols = feature_config["numerical_features"]
    cat_cols = feature_config["categorical_features"]
    all_features = num_cols + cat_cols
    target_col = feature_config["reg_target"]  # "future_price_7d"

    # Filter rows with complete features and valid target
    valid_mask = df[num_cols].notna().all(axis=1) & df[target_col].notna()
    data = df[valid_mask].copy()

    say(f"\nDataset loaded: {len(df):,d} total rows")
    say(f"Complete cases for regression: {len(data):,d} rows")
    say(f"Target variable: {target_col} (PKR per 100 kg)")
    say(f"Total input features: {len(all_features)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)")

    # 2. Split dataset chronologically
    splits = {}
    for s in ("train", "val", "test"):
        sub = data[data["split"] == s]
        X = sub[all_features]
        y = sub[target_col]
        splits[s] = {
            "df": sub,
            "X": X,
            "y": y,
            "dates": (sub["date"].min().date(), sub["date"].max().date()),
        }
        say(f"  {s.capitalize():5s}: {len(sub):>6,d} rows ({splits[s]['dates'][0]} to {splits[s]['dates'][1]}) "
            f"| Mean Price={y.mean():>9.1f} PKR | Min={y.min():>7.1f} | Max={y.max():>8.1f}")

    # 3. Define candidate regression models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            min_samples_split=12,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # 4. Train, evaluate and collect metrics
    results = []
    trained_pipelines = {}

    say("\n" + "=" * 75)
    say("Model Training & Evaluation")
    say("=" * 75)

    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]

    for name, reg in candidate_models.items():
        say(f"\n>>> Training {name}...")

        # Build full pipeline (preprocessor + model)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("regressor", reg),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Save individual model pipeline
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODELS_DIR / f"reg_{slug}.joblib")
        say(f"    Saved models/reg_{slug}.joblib")

        # Evaluate across all splits
        for split_key, X_set, y_set in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            y_pred = pipeline.predict(X_set)
            m = evaluate_split(y_set, y_pred, split_name=split_key)
            m["model"] = name
            results.append(m)

            say(f"    [{split_key.upper():5s}] MAE: {m['mae']:>8.1f} PKR | RMSE: {m['rmse']:>8.1f} PKR | "
                f"R²: {m['r2']:>6.4f} | MAPE: {m['mape_pct']:>5.2f}%")

    # 5. Summary Table & Model Comparison
    metrics_df = pd.DataFrame(results)
    col_order = ["model", "split", "mae", "rmse", "r2", "mape_pct"]
    export_df = metrics_df[col_order].sort_values(["split", "rmse"], ascending=[True, True])
    export_df.to_csv(METRICS_CSV, index=False)

    say("\n" + "=" * 75)
    say("VALIDATION SET COMPARISON (Selection Set)")
    say("=" * 75)
    val_table = export_df[export_df["split"] == "val"].drop(columns=["split"])
    say(val_table.to_string(index=False))

    say("\n" + "=" * 75)
    say("TEST SET PERFORMANCE (Final Holdout Evaluation)")
    say("=" * 75)
    test_table = export_df[export_df["split"] == "test"].drop(columns=["split"])
    say(test_table.to_string(index=False))

    # 6. Select Best Model Based on Validation RMSE & R²
    val_metrics = metrics_df[metrics_df["split"] == "val"].copy()
    best_row = val_metrics.sort_values(by="rmse", ascending=True).iloc[0]
    best_model_name = best_row["model"]
    best_pipeline = trained_pipelines[best_model_name]

    say("\n" + "=" * 75)
    say(f"SELECTED BEST REGRESSION MODEL: {best_model_name}")
    say(f"Validation RMSE: {best_row['rmse']:.1f} PKR | MAE: {best_row['mae']:.1f} PKR | R²: {best_row['r2']:.4f}")
    say("=" * 75)

    # Save to canonical deployment file: models/regression_model.joblib
    primary_save_path = MODELS_DIR / "regression_model.joblib"
    joblib.dump(best_pipeline, primary_save_path)
    say(f"\n✓ Saved best regression pipeline to: {primary_save_path}")

    # 7. Breakdown of Test Performance by Crop for the Selected Model
    test_df = splits["test"]["df"].copy()
    test_df["predicted_price_7d"] = best_pipeline.predict(splits["test"]["X"])
    test_df["error_abs"] = (test_df[target_col] - test_df["predicted_price_7d"]).abs()
    test_df["error_pct"] = test_df["error_abs"] / test_df[target_col] * 100.0

    crop_summary = []
    for crop_name, g in test_df.groupby("crop"):
        c_mae = mean_absolute_error(g[target_col], g["predicted_price_7d"])
        c_rmse = calc_rmse(g[target_col], g["predicted_price_7d"])
        c_r2 = r2_score(g[target_col], g["predicted_price_7d"])
        c_mape = g["error_pct"].mean()
        c_mean_price = g[target_col].mean()
        crop_summary.append({
            "Crop": crop_name,
            "Mean Price (PKR)": round(c_mean_price, 1),
            "MAE (PKR)": round(c_mae, 1),
            "RMSE (PKR)": round(c_rmse, 1),
            "R²": round(c_r2, 4),
            "MAPE (%)": round(c_mape, 2),
        })

    say("\nTest Performance Breakdown by Crop (Selected Model):")
    say(pd.DataFrame(crop_summary).to_string(index=False))

    # Feature Importance for Random Forest
    final_reg = best_pipeline.named_steps["regressor"]
    if hasattr(final_reg, "feature_importances_"):
        out_names = feature_config["all_output_names"]
        importances = pd.Series(final_reg.feature_importances_, index=out_names).sort_values(ascending=False)
        say("\nTop 15 Most Influential Features in Price Prediction:")
        say(importances.head(15).round(4).to_string())

    say(f"\n✓ Saved evaluation metrics to: {METRICS_CSV}")
    REPORT_TXT.write_text("\n".join(REPORT), encoding="utf-8")
    say(f"✓ Saved full report to: {REPORT_TXT}")


if __name__ == "__main__":
    main()
