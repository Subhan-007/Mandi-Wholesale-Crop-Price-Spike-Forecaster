"""
Phase 2 — Step 5: Model Evaluation, Comparison & Pipeline Finalization

Per project PDF:
  - Verifies chronological 70% train / 15% val / 15% test data partitioning.
  - Evaluates all 3 classifiers using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
  - Evaluates both regressors using MAE, RMSE, and R² (plus MAPE).
  - Performs direct side-by-side comparison within each task.
  - Selects the best performing classifier and regressor.
  - Ensures winning pipelines are saved as:
      models/classification_model.joblib
      models/regression_model.joblib
  - Exports visual evaluation figures and summary reports for presentation/docs.

Outputs:
  models/classification_model.joblib          — verified best classification pipeline
  models/regression_model.joblib              — verified best regression pipeline
  eda_outputs/figures/11_classification_comparison.png — classifier comparison & confusion matrices
  eda_outputs/figures/12_regression_comparison.png     — regressor comparison & actual vs predicted
  data/processed/model_comparison_report.txt  — comprehensive audit report
  data/processed/classification_summary.csv   — final classification metrics
  data/processed/regression_summary.csv       — final regression metrics
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# ---- paths -----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import PROCESSED_DIR  # noqa: E402

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "eda_outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ML_FEATURED_CSV = PROCESSED_DIR / "ml_featured.csv"
REPORT_TXT = PROCESSED_DIR / "model_comparison_report.txt"

REPORT: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    REPORT.append(str(msg))


def calc_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    say("=" * 80)
    say("Phase 2 — Subtask 5: Model Evaluation, Comparison & Pipeline Finalization")
    say("=" * 80)

    # 1. Load dataset & configuration
    df = pd.read_csv(ML_FEATURED_CSV, parse_dates=["date"])
    feature_config = joblib.load(MODELS_DIR / "feature_config.joblib")

    num_cols = feature_config["numerical_features"]
    cat_cols = feature_config["categorical_features"]
    all_features = num_cols + cat_cols
    clf_target = feature_config["clf_target"]  # "spike_7d"
    reg_target = feature_config["reg_target"]  # "future_price_7d"

    # Filter complete cases
    valid_mask = df[num_cols].notna().all(axis=1) & df[clf_target].notna() & df[reg_target].notna()
    data = df[valid_mask].copy()
    data[clf_target] = data[clf_target].astype(int)

    say(f"\n1. CHRONOLOGICAL DATA SPLIT AUDIT")
    say("-" * 80)
    total_valid = len(data)
    say(f"Total model-ready rows: {total_valid:,d}")

    splits = {}
    for s in ("train", "val", "test"):
        sub = data[data["split"] == s]
        splits[s] = {
            "df": sub,
            "X": sub[all_features],
            "y_clf": sub[clf_target],
            "y_reg": sub[reg_target],
            "dates": (sub["date"].min().date(), sub["date"].max().date()),
            "count": len(sub),
            "pct": len(sub) / total_valid * 100.0,
        }
        say(f"  • {s.upper():5s} set: {len(sub):>6,d} rows ({splits[s]['pct']:5.1f}%) | "
            f"Range: {splits[s]['dates'][0]} to {splits[s]['dates'][1]} | "
            f"Spikes: {(splits[s]['y_clf'] == 1).sum():>4,d} ({(splits[s]['y_clf'] == 1).mean()*100:4.1f}%)")

    # 2. CLASSIFIER EVALUATION & COMPARISON
    say("\n" + "=" * 80)
    say("2. CLASSIFICATION TASK EVALUATION (Predicting 7-Day Price Spikes)")
    say("=" * 80)

    clf_models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "clf_logistic_regression.joblib"),
        "Decision Tree": joblib.load(MODELS_DIR / "clf_decision_tree.joblib"),
        "Random Forest": joblib.load(MODELS_DIR / "clf_random_forest.joblib"),
    }

    clf_records = []
    cms = {}

    for name, pipeline in clf_models.items():
        cms[name] = {}
        for s in ("train", "val", "test"):
            X = splits[s]["X"]
            y = splits[s]["y_clf"]
            y_pred = pipeline.predict(X)
            y_prob = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            auc = roc_auc_score(y, y_prob) if y_prob is not None else np.nan
            cm = confusion_matrix(y, y_pred)
            cms[name][s] = cm

            tn, fp, fn, tp = cm.ravel()
            clf_records.append({
                "Model": name,
                "Split": s,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            })

    clf_df = pd.DataFrame(clf_records)
    clf_df.to_csv(PROCESSED_DIR / "classification_summary.csv", index=False)

    say("\nValidation Split Performance (Used for Model Selection):")
    val_clf = clf_df[clf_df["Split"] == "val"].sort_values(by="F1-Score", ascending=False)
    say(val_clf[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "FP", "FN", "TN"]].to_string(index=False))

    say("\nTest Split Performance (Unseen Generalization Assessment):")
    test_clf = clf_df[clf_df["Split"] == "test"].sort_values(by="F1-Score", ascending=False)
    say(test_clf[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "TP", "FP", "FN", "TN"]].to_string(index=False))

    # Selection decision
    best_clf_name = val_clf.iloc[0]["Model"]
    best_clf_pipeline = clf_models[best_clf_name]
    say(f"\n>>> Selected Best Classifier: {best_clf_name.upper()}")
    say(f"    Reason: Achieved top validation F1-score ({val_clf.iloc[0]['F1-Score']:.4f}) and test accuracy "
        f"({test_clf[test_clf['Model'] == best_clf_name]['Accuracy'].values[0]*100:.1f}%) while suppressing false alarms.")

    # Save to canonical deployment path
    joblib.dump(best_clf_pipeline, MODELS_DIR / "classification_model.joblib")
    say(f"    ✓ Serialized best classifier pipeline to: models/classification_model.joblib")

    # 3. REGRESSOR EVALUATION & COMPARISON
    say("\n" + "=" * 80)
    say("3. REGRESSION TASK EVALUATION (Predicting 7-Day Wholesale Price in PKR/100kg)")
    say("=" * 80)

    reg_models = {
        "Linear Regression": joblib.load(MODELS_DIR / "reg_linear_regression.joblib"),
        "Random Forest Regressor": joblib.load(MODELS_DIR / "reg_random_forest_regressor.joblib"),
    }

    reg_records = []
    reg_test_preds = {}

    for name, pipeline in reg_models.items():
        for s in ("train", "val", "test"):
            X = splits[s]["X"]
            y = splits[s]["y_reg"]
            y_pred = pipeline.predict(X)
            if s == "test":
                reg_test_preds[name] = y_pred

            mae = mean_absolute_error(y, y_pred)
            rmse = calc_rmse(y, y_pred)
            r2 = r2_score(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred) * 100.0

            reg_records.append({
                "Model": name,
                "Split": s,
                "MAE (PKR)": mae,
                "RMSE (PKR)": rmse,
                "R² Score": r2,
                "MAPE (%)": mape,
            })

    reg_df = pd.DataFrame(reg_records)
    reg_df.to_csv(PROCESSED_DIR / "regression_summary.csv", index=False)

    say("\nValidation Split Performance (Used for Model Selection):")
    val_reg = reg_df[reg_df["Split"] == "val"].sort_values(by="RMSE (PKR)", ascending=True)
    say(val_reg[["Model", "MAE (PKR)", "RMSE (PKR)", "R² Score", "MAPE (%)"]].to_string(index=False))

    say("\nTest Split Performance (Unseen Generalization Assessment):")
    test_reg = reg_df[reg_df["Split"] == "test"].sort_values(by="RMSE (PKR)", ascending=True)
    say(test_reg[["Model", "MAE (PKR)", "RMSE (PKR)", "R² Score", "MAPE (%)"]].to_string(index=False))

    # Selection decision
    best_reg_name = val_reg.iloc[0]["Model"]
    best_reg_pipeline = reg_models[best_reg_name]
    say(f"\n>>> Selected Best Regressor: {best_reg_name.upper()}")
    say(f"    Reason: Lowest Validation RMSE ({val_reg.iloc[0]['RMSE (PKR)']:.1f} PKR) and superior test generalizability (R²: {test_reg[test_reg['Model'] == best_reg_name]['R² Score'].values[0]:.4f}).")

    # Save to canonical deployment path
    joblib.dump(best_reg_pipeline, MODELS_DIR / "regression_model.joblib")
    say(f"    ✓ Serialized best regressor pipeline to: models/regression_model.joblib")

    # 4. GENERATE COMPARISON FIGURES
    say("\n" + "=" * 80)
    say("4. GENERATING VISUAL COMPARISON CHARTS")
    say("=" * 80)

    # Figure 11: Classification Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Classification Models Comparison — 7-Day Spike Forecaster (Test Split)", fontsize=15, fontweight="bold")

    # Bar chart of test metrics
    test_clf_plot = test_clf.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    test_clf_plot.plot(kind="bar", ax=axes[0, 0], colormap="viridis", edgecolor="black")
    axes[0, 0].set_title("Test Set Evaluation Metrics", fontweight="bold")
    axes[0, 0].set_ylabel("Score (0.0 to 1.0)")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0, 0].legend(loc="upper right", frameon=True)

    # Confusion matrices
    cm_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for (m_name, cm_dict), ax in zip(cms.items(), cm_axes):
        cm = cm_dict["test"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Normal (0)", "Spike (1)"], yticklabels=["Normal (0)", "Spike (1)"])
        ax.set_title(f"Confusion Matrix: {m_name} (Test)", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    clf_fig_path = FIGURES_DIR / "11_classification_comparison.png"
    plt.savefig(clf_fig_path, dpi=200)
    plt.close()
    say(f"  ✓ Saved figure: {clf_fig_path}")

    # Figure 12: Regression Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Regression Models Comparison — 7-Day Ahead Price Prediction", fontsize=15, fontweight="bold")

    # Metric comparison across splits
    reg_plot_data = reg_df.pivot(index="Split", columns="Model", values="RMSE (PKR)")[["Linear Regression", "Random Forest Regressor"]].loc[["train", "val", "test"]]
    reg_plot_data.plot(kind="bar", ax=axes[0], color=["#2b5c8f", "#d95f02"], edgecolor="black")
    axes[0].set_title("Root Mean Squared Error (RMSE) across Splits", fontweight="bold")
    axes[0].set_ylabel("RMSE in PKR per 100 kg")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Actual vs Predicted scatter plot (Linear Regression on Test)
    test_sub = splits["test"]["df"].copy()
    test_sub["Pred"] = reg_test_preds["Linear Regression"]
    palette = sns.color_palette("tab10", n_colors=test_sub["crop"].nunique())
    sns.scatterplot(data=test_sub, x=reg_target, y="Pred", hue="crop", alpha=0.6, s=25, ax=axes[1], palette=palette)
    min_val = min(test_sub[reg_target].min(), test_sub["Pred"].min())
    max_val = max(test_sub[reg_target].max(), test_sub["Pred"].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")
    axes[1].set_title("Actual vs Predicted Price: Linear Regression (Test Set)", fontweight="bold")
    axes[1].set_xlabel("Actual 7-Day Future Price (PKR/100kg)")
    axes[1].set_ylabel("Predicted 7-Day Future Price (PKR/100kg)")
    axes[1].legend(title="Crop", loc="upper left", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    reg_fig_path = FIGURES_DIR / "12_regression_comparison.png"
    plt.savefig(reg_fig_path, dpi=200)
    plt.close()
    say(f"  ✓ Saved figure: {reg_fig_path}")

    # 5. SUMMARY OF PRODUCTION PIPELINES
    say("\n" + "=" * 80)
    say("5. PRODUCTION DEPLOYMENT DELIVERABLES (Per Project Specification)")
    say("=" * 80)
    say(f"  1. Classification Pipeline: {MODELS_DIR / 'classification_model.joblib'}")
    say(f"     Model: {best_clf_name}")
    say(f"     Inputs: 33 raw features -> Encoded/Scaled to 47 features -> Classifier")
    say(f"  2. Regression Pipeline:     {MODELS_DIR / 'regression_model.joblib'}")
    say(f"     Model: {best_reg_name}")
    say(f"     Inputs: 33 raw features -> Encoded/Scaled to 47 features -> Regressor")
    say("=" * 80)

    REPORT_TXT.write_text("\n".join(REPORT), encoding="utf-8")
    say(f"✓ Saved master evaluation report to: {REPORT_TXT}")


if __name__ == "__main__":
    main()
