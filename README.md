# Mandi & Wholesale Crop Price Spike Forecaster

**Course:** Introduction to Data Science (Fall 2026)  
**Dataset:** AMIS Punjab Daily Crop Rates & Open-Meteo Weather API (2022–2026)  
**Scope:** 5 Crops (*Onion, Tomato, Potato, Garlic, Green Chilli*) across 5 Punjab Mandis (*Lahore, Faisalabad, Rawalpindi, Multan, Gujranwala*).

---

## Project Status

- [x] **Phase 1: Data Engineering & EDA** (Completed by Member 1)
  * AMIS Punjab scraper & Open-Meteo weather integration.
  * Outlier removal, calendar gap-filling (1–2 days), quality tiering.
  * Comprehensive Exploratory Data Analysis (`notebooks/EDA.ipynb`).
- [x] **Phase 2: Feature Engineering & Machine Learning** (Completed by Member 2)
  * Target variable definitions: 7-day future price (`future_price_7d`) and spike label (`spike_7d`).
  * Feature pipeline: 7d/14d lags, rolling averages, volatility spreads, weather windows, seasonality & Eid dates.
  * 3 Classification models developed & compared (Random Forest selected).
  * 2 Regression models developed & compared (Linear Regression selected).
  * Full scikit-learn deployment pipelines serialized via Joblib.
- [ ] **Phase 3: Web Application & Deployment** (Next step for Member 3)
  * Flask backend connected to trained `.joblib` pipelines.
  * Web interface with crop/city selectors, price predictions, and spike risk probability.
  * Vercel deployment.

---

## Folder Structure

```text
├── data/
│   ├── raw/                             Raw AMIS prices & Open-Meteo weather CSVs
│   └── processed/
│       ├── mandi_dataset.csv            Clean Phase 1 daily dataset (49,252 rows)
│       ├── ml_ready.csv                 Phase 2 dataset with regression & classification targets
│       ├── ml_featured.csv              Feature-engineered modelling dataset (40 columns)
│       ├── classification_summary.csv   Tabular metrics for all 3 classifiers across splits
│       ├── regression_summary.csv       Tabular metrics for both regressors across splits
│       ├── feature_report.txt           Scaling and categorical encoding audit log
│       └── model_comparison_report.txt  Comprehensive model audit and selection report
├── docs/
│   └── data_dictionary.md               Full column definitions, units, and cleaning rules
├── eda_outputs/
│   ├── figures/                         Plots 01–10 (Phase 1 EDA) + 11–12 (Phase 2 Model Evaluation)
│   │   ├── 11_classification_comparison.png
│   │   └── 12_regression_comparison.png
│   └── tables/                          Summary CSV tables from EDA
├── models/                              Production pipelines & serialized models
│   ├── classification_model.joblib      PRIMARY: Selected Random Forest Classifier pipeline
│   ├── regression_model.joblib          PRIMARY: Selected Linear Regression pipeline
│   ├── preprocessor.joblib              Fitted ColumnTransformer (Scaler + OneHotEncoder)
│   ├── feature_config.joblib            Feature column lists and metadata
│   ├── clf_logistic_regression.joblib   Baseline classification pipeline
│   ├── clf_decision_tree.joblib         Decision tree classification pipeline
│   ├── clf_random_forest.joblib         Random forest classification pipeline
│   ├── reg_linear_regression.joblib     Linear regression pipeline
│   └── reg_random_forest_regressor.joblib Random forest regressor pipeline
├── notebooks/
│   └── EDA.ipynb                        Exploratory data analysis notebook
├── scripts/
│   ├── config.py                        Global settings, crop IDs, coordinates, and paths
│   ├── requirements.txt                 Python dependencies
│   ├── 01_probe_amis.py                 AMIS website investigation script
│   ├── 02_scrape_amis.py                AMIS historical web scraper
│   ├── 03_collect_weather.py            Open-Meteo API weather collector
│   ├── 04_clean_merge.py                Legacy v2 cleaning rules
│   ├── 07_build_v3.py                   Phase 1 master builder (raw -> mandi_dataset.csv)
│   ├── 08_prepare_targets.py            Phase 2: Target variable generation & verification
│   ├── 09_feature_engineering.py        Phase 2: Feature creation, scaling & preprocessor fitting
│   ├── 10_train_classification.py       Phase 2: Train & evaluate 3 classification models
│   ├── 11_train_regression.py           Phase 2: Train & evaluate 2 regression models
│   └── 12_model_evaluation_and_comparison.py Phase 2: Master comparison, report & plot generation
└── README.md
```

---

## Machine Learning Results Summary (Phase 2)

Data is partitioned **chronologically** to guarantee zero future data leakage:
* **Train (71.1%):** 2022-01-15 to 2025-06-06 (31,413 rows)
* **Validation (15.9%):** 2025-06-07 to 2026-02-27 (7,004 rows)
* **Test (13.0%):** 2026-02-28 to 2026-11-13 (5,746 rows)

### 1. Classification: 7-Day Price Spike Forecaster
Predicts whether next week's wholesale price will exceed the crop's normal historical increase by $> 1.5\sigma$. Evaluated on unseen **Test Split**:

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Alarms (FP) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **89.8%** | **19.0%** | 30.5% | **0.2344** | **0.7161** | **383** | **Selected Best** |
| Logistic Regression | 65.1% | 9.2% | **65.4%** | 0.1615 | 0.7113 | 1,902 | Baseline |
| Decision Tree | 61.6% | 7.5% | 56.9% | 0.1321 | 0.6677 | 2,081 | Evaluated |

* **Why Random Forest was selected:** Random Forest achieved **90% accuracy** and cut out over 80% of the false alarms generated by linear and single-tree models. For web app users, its high confidence on normal days (**96% precision on normal class**) and solid risk ranking (**0.72 ROC-AUC**) provide a trustworthy early-warning signal.
* **Top Drivers of Spikes:** `day_of_year` (harvest season), `price_vs_city_median` (provincial divergence), `roll14_mean`, `vol_14d` (pre-spike volatility), and `days_to_next_eid`.

### 2. Regression: 7-Day Wholesale Price Predictor
Predicts the actual expected wholesale rate in PKR per 100 kg. Evaluated on unseen **Test Split**:

| Model | Test MAE | Test RMSE | $R^2$ Score | Mean % Error (MAPE) | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** | **1,264.1 PKR** | **1,850.9 PKR** | **0.9737** | **12.52%** | **Selected Best** |
| Random Forest Regressor | 1,319.7 PKR | 1,959.6 PKR | 0.9705 | 13.09% | Evaluated |

* **Why Linear Regression was selected:** In time-series commodity forecasting, current price and rolling averages provide strong linear anchors. Linear Regression yielded the lowest validation RMSE (1,342.9 PKR) and generalized better to continuous upward price trends on test data ($R^2 = 0.9737$, ~12.5% average error).
* **Crop Breakdown (Selected Model on Test Set):**
  * **Garlic:** MAE 1,208.7 PKR | $R^2$ **0.9833** | MAPE **5.6%** (highly predictable)
  * **Onion:** MAE 979.9 PKR | $R^2$ **0.9348** | MAPE **10.0%**
  * **Green Chilli:** MAE 985.2 PKR | $R^2$ 0.5855 | MAPE **11.4%**
  * **Tomato:** MAE 2,238.7 PKR | $R^2$ 0.7944 | MAPE **20.6%** (perishable, high volatility)
  * **Potato:** MAE 482.9 PKR | $R^2$ 0.3885 | MAPE **24.4%** (low base price, seasonal gaps)

---

## How to Run & Reproduce

### 1. Environment Setup
```bash
# Clone the repository and navigate to root
cd "Mandi Phase1_v3"

# Activate the virtual environment
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Re-run Phase 2 Machine Learning Pipeline (Step-by-Step)
You can run the modules individually or end-to-end:

```bash
# Step 1: Create regression target and verify classification targets
python scripts/08_prepare_targets.py

# Step 2: Generate lag features, rolling windows, and fit preprocessor
python scripts/09_feature_engineering.py

# Step 3: Train and evaluate the 3 classification models
python scripts/10_train_classification.py

# Step 4: Train and evaluate the 2 regression models
python scripts/11_train_regression.py

# Step 5: Run master comparison, generate metrics tables and visual figures
python scripts/12_model_evaluation_and_comparison.py
```

---

## Handoff Guide for Member 3 (Web App & Deployment)

Member 3 can directly use the production-ready Joblib pipelines stored in `models/`. Each pipeline is self-contained with its own preprocessor:

```python
import joblib
import pandas as pd

# 1. Load the production models
clf_pipeline = joblib.load("models/classification_model.joblib")
reg_pipeline = joblib.load("models/regression_model.joblib")
feature_config = joblib.load("models/feature_config.joblib")

# 2. Predict on new input data (DataFrame with the raw input features)
sample_input = pd.DataFrame([{
    "city": "Lahore",
    "crop": "Onion",
    "variant": "Onion",
    "price_pkr_per_100kg": 11500.0,
    "price_lag_7d": 11000.0,
    "price_lag_14d": 10500.0,
    "pct_chg_1d": 0.0,
    "pct_chg_7d": 0.045,
    "pct_chg_14d": 0.095,
    "roll7_mean": 11200.0,
    "roll14_mean": 10900.0,
    "roll7_std_pct": 0.04,
    "vol_14d": 0.03,
    "price_vs_city_median": 1.02,
    "rain_mm": 0.0,
    "rain_7d_mm": 5.0,
    "rain_14d_mm": 12.0,
    "days_since_rain": 4,
    "temp_max_c": 32.0,
    "temp_min_c": 18.0,
    "temp_mean_c": 25.0,
    "temp_mean_7d_c": 24.5,
    "heat_day": 0,
    "month": 11,
    "day_of_week": 6,
    "day_of_year": 326,
    "season": "Autumn",
    "is_ramadan": 0,
    "days_to_next_eid": 45,
    "days_since_last_eid": 60,
    "is_pre_eid_10d": 0,
    "is_filled": 0,
    "is_stale": 0
}])

# 3. Generate App Outputs
predicted_price = reg_pipeline.predict(sample_input)[0]
spike_prediction = clf_pipeline.predict(sample_input)[0]      # 0 = Normal, 1 = Price Spike
spike_probability = clf_pipeline.predict_proba(sample_input)[0][1]  # e.g., 0.35 -> 35% risk

print(f"Predicted Price: {predicted_price:.1f} PKR/100kg")
print(f"Spike Risk: {'HIGH' if spike_prediction == 1 else 'LOW'} ({spike_probability * 100:.1f}%)")
```
