"""
Phase 2 — Step 3: Classification Model Development

Implements and compares three classification models to predict 7-day price spikes:
  1. Logistic Regression      (Baseline probabilistic classification)
  2. Decision Tree Classifier (Rule-based classification)
  3. Random Forest Classifier (Ensemble classification)

Per project PDF:
  - Chronological 70% train / 15% validation / 15% test split.
  - Class imbalance addressed using class_weight='balanced'.
  - Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
  - Best model pipeline saved to models/classification_model.joblib for Flask app.

Outputs:
  models/classification_model.joblib     — best classification pipeline
  models/clf_logistic_regression.joblib  — logistic regression pipeline
  models/clf_decision_tree.joblib        — decision tree pipeline
  models/clf_random_forest.joblib        — random forest pipeline
  data/processed/classification_report.txt — detailed evaluation report
  data/processed/classification_metrics.csv — tabular metrics comparison
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# ---- paths -----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import PROCESSED_DIR  # noqa: E402

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

ML_FEATURED_CSV = PROCESSED_DIR / "ml_featured.csv"
REPORT_TXT = PROCESSED_DIR / "classification_report.txt"
METRICS_CSV = PROCESSED_DIR / "classification_metrics.csv"

REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


def evaluate_split(y_true, y_pred, y_prob=None, split_name="val") -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

    return {
        "split": split_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def format_cm(cm: np.ndarray) -> str:
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    return f"TN={tn:>5,d}  FP={fp:>4,d} | FN={fn:>4,d}  TP={tp:>4,d}"


def main() -> None:
    say("=" * 75)
    say("Phase 2 — Classification Model Development (Price Spike Forecaster)")
    say("=" * 75)

    # 1. Load dataset & configuration
    df = pd.read_csv(ML_FEATURED_CSV, parse_dates=["date"])
    feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")
    base_preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    num_cols = feature_config["numerical_features"]
    cat_cols = feature_config["categorical_features"]
    all_features = num_cols + cat_cols
    target_col = feature_config["clf_target"]  # "spike_7d"

    # Filter rows with complete features and valid target
    valid_mask = df[num_cols].notna().all(axis=1) & df[target_col].notna()
    data = df[valid_mask].copy()
    data[target_col] = data[target_col].astype(int)

    say(f"\nDataset loaded: {len(df):,d} total rows")
    say(f"Complete cases for classification: {len(data):,d} rows")
    say(f"Total input features: {len(all_features)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)")

    # 2. Split dataset chronologically
    splits = {}
    for s in ("train", "val", "test"):
        sub = data[data["split"] == s]
        X = sub[all_features]
        y = sub[target_col]
        splits[s] = {"X": X, "y": y, "dates": (sub["date"].min().date(), sub["date"].max().date())}
        n_spikes = (y == 1).sum()
        say(f"  {s.capitalize():5s}: {len(sub):>6,d} rows ({splits[s]['dates'][0]} to {splits[s]['dates'][1]}) "
            f"| Normal={(y == 0).sum():>5,d} | Spike={n_spikes:>4,d} ({n_spikes/len(sub)*100:.1f}%)")

    # 3. Define the three classification models
    # Note: class_weight='balanced' addresses the ~15:1 class imbalance
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=7,
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
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

    for name, clf in candidate_models.items():
        say(f"\n>>> Training {name}...")

        # Build full pipeline (preprocessor + model)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("classifier", clf),
            ]
        )

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Save individual model pipeline
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODELS_DIR / f"clf_{slug}.joblib")

        say(f"    Saved models/clf_{slug}.joblib")

        # Evaluate across all splits
        for split_key, X_set, y_set in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            y_pred = pipeline.predict(X_set)
            y_prob = pipeline.predict_proba(X_set)[:, 1] if hasattr(pipeline, "predict_proba") else None
            m = evaluate_split(y_set, y_pred, y_prob, split_name=split_key)
            m["model"] = name
            results.append(m)

            say(f"    [{split_key.upper():5s}] Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | "
                f"Rec: {m['recall']:.4f} | F1: {m['f1']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | "
                f"CM: {format_cm(m['confusion_matrix'])}")

    # 5. Summary Table & Model Comparison
    metrics_df = pd.DataFrame(results)

    # Flatten confusion matrix for CSV export
    metrics_df["tn"] = metrics_df["confusion_matrix"].apply(lambda cm: cm[0, 0])
    metrics_df["fp"] = metrics_df["confusion_matrix"].apply(lambda cm: cm[0, 1])
    metrics_df["fn"] = metrics_df["confusion_matrix"].apply(lambda cm: cm[1, 0])
    metrics_df["tp"] = metrics_df["confusion_matrix"].apply(lambda cm: cm[1, 1])

    # Reorder columns
    col_order = ["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "fn", "tn"]
    export_df = metrics_df[col_order].sort_values(["split", "f1"], ascending=[True, False])
    export_df.to_csv(METRICS_CSV, index=False)

    say("\n" + "=" * 75)
    say("VALIDATION SET COMPARISON (Selection Set)")
    say("=" * 75)
    val_table = export_df[export_df["split"] == "val"].drop(columns=["split"])
    say(val_table.to_string(index=False))

    say("\n" + "=" * 75)
    say("TEST SET PERFORMANCE (Final Evaluation)")
    say("=" * 75)
    test_table = export_df[export_df["split"] == "test"].drop(columns=["split"])
    say(test_table.to_string(index=False))

    # 6. Select Best Model Based on Validation F1-score
    val_metrics = metrics_df[metrics_df["split"] == "val"].copy()
    best_row = val_metrics.sort_values(by="f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_pipeline = trained_pipelines[best_model_name]

    say("\n" + "=" * 75)
    say(f"SELECTED BEST MODEL: {best_model_name}")
    say(f"Validation F1-score: {best_row['f1']:.4f} (Recall: {best_row['recall']:.4f}, Precision: {best_row['precision']:.4f})")
    say("=" * 75)

    # Save to canonical deployment file: models/classification_model.joblib
    primary_save_path = MODELS_DIR / "classification_model.joblib"
    joblib.dump(best_pipeline, primary_save_path)
    say(f"\n✓ Saved best classification pipeline to: {primary_save_path}")

    # Detailed classification report on Test set for the selected model
    y_test_pred = best_pipeline.predict(X_test)
    cr = classification_report(y_test, y_test_pred, target_names=["Normal (0)", "Price Spike (1)"])
    say("\nFinal Test Classification Report (Selected Model):")
    say(cr)

    # Feature Importance for Tree models
    final_clf = best_pipeline.named_steps["classifier"]
    if hasattr(final_clf, "feature_importances_"):
        out_names = feature_config["all_output_names"]
        importances = pd.Series(final_clf.feature_importances_, index=out_names).sort_values(ascending=False)
        say("\nTop 15 Most Important Features in Selected Model:")
        say(importances.head(15).round(4).to_string())

    say(f"\n✓ Saved evaluation metrics to: {METRICS_CSV}")
    REPORT_TXT.write_text("\n".join(REPORT), encoding="utf-8")
    say(f"✓ Saved full report to: {REPORT_TXT}")


if __name__ == "__main__":
    main()
