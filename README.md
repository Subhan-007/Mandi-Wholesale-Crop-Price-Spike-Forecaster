# Mandi Project - Phase 1: Data Engineering & EDA (Member 1)

Daily wholesale prices of 5 crops (Onion, Tomato, Potato, Garlic, Green Chilli) in 5 Punjab cities (Lahore, Faisalabad,
Rawalpindi, Multan, Gujranwala), 2022-01-01 to 2026-11-20, from AMIS Punjab, merged with Open-Meteo weather. Includes
features, a 7-day-ahead target, a spike label and a train/val/test split for Member 2's machine-learning work.

**Current version: v3.** Use `data/processed/mandi_dataset.csv` and `docs/data_dictionary.md`.
The previous version (v2) is kept untouched in `data/processed/archive_v2/` for comparison only.

## Folder structure
```
├── scripts/
│   ├── config.py                    settings: crops, cities, AMIS IDs, dates, file paths
│   ├── requirements.txt             Python packages
│   ├── 01_probe_amis.py             investigation: AMIS home page + Open-Meteo test
│   ├── 01b_probe_price_trends.py    investigation: the Price Trends form
│   ├── 01c_try_ssrs_export.py       investigation: the report's CSV export
│   ├── 02_scrape_amis.py            AMIS scraper                     -> data/raw/amis_prices_raw.csv, data/raw/exports/
│   ├── 03_collect_weather.py        Open-Meteo weather               -> data/raw/weather_raw.csv
│   ├── 04_clean_merge.py            v2 cleaning; v3 imports its filters. Do not run on its own (see below)
│   ├── 05_diagnose.py               read-only checks on the raw data
│   ├── 06_validate_dataset.py       v2 validator only - does NOT work on the v3 file
│   ├── 07_build_v3.py               v3 build: raw -> cleaned, featured dataset (use this one)
│   └── oneoff/                      one-time check/patch scripts from the v2 clean-up; not needed any more
├── notebooks/EDA.ipynb              exploratory analysis (saves figures + tables)
├── docs/data_dictionary.md          every column and every cleaning rule
├── data/
│   ├── raw/                         amis_prices_raw.csv, weather_raw.csv, exports/ (untouched AMIS downloads)
│   └── processed/                   mandi_dataset.csv (main), mandi_observed_v3.csv, removed_v3.csv,
│                                    cleaning_report_v3.txt, archive_v2/
└── eda_outputs/{figures,tables}/    written by the notebook
```

## How to reproduce
Run from the project root.

```bash
pip install -r scripts/requirements.txt      # also install jupyter if you want to open the notebook
python scripts/07_build_v3.py                # rebuilds the dataset from data/raw (no internet needed)
# then open notebooks/EDA.ipynb and Run All
```
This rebuilds the shipped dataset exactly from the raw files in `data/raw/`.

To refresh the raw data from the internet first (the end date is today's date, so the result will be newer than the shipped
data, which ends 2026-11-20):
```bash
python scripts/02_scrape_amis.py             # ~30 series x 5 yearly chunks; resumable, just re-run if it stops
python scripts/03_collect_weather.py
python scripts/07_build_v3.py
```
Notes:
- Put your own e-mail in `HEADERS` in `scripts/config.py` (polite user-agent for the scraper). `build_request()` in that file is an unused leftover; ignore it.
- `07_build_v3.py` imports its unit conversion and outlier filters from `04_clean_merge.py`. Running `04_clean_merge.py` by itself
  would overwrite `mandi_dataset.csv` with the old v2 format, so do not run it.
- `07_build_v3.py` compares against `data/processed/archive_v2/mandi_dataset.csv` at the end. Keep that file in place.
- `01_*` and `05_*` are investigation scripts; they are not needed to rebuild.

## How the data is collected
- AMIS site: `http://www.amis.pk` (**http**, https does not work). The "Price Trends" report (Crop + City + Start/End date) is an
  ASP.NET / SQL Reporting Services page loaded by JavaScript, so the scraper submits the form and downloads the report's
  built-in **CSV export**. One request returns one year for one crop and city.
- AMIS gives **one** price per day (PKR per 100 kg). No min / max / usual price.
- Weather: Open-Meteo archive API (daily rain and temperature at each city's coordinates).

## What v3 does (details in `docs/data_dictionary.md` and `data/processed/cleaning_report_v3.txt`)
- Backup series are **not** spliced any more: Potato Fresh / Potato Store and Garlic (Local) / Garlic (China) are separate
  series in column `variant`.
- Removes: unit/duplicate/invalid prices, outliers vs the city's own level, persistent city-specific glitches vs the other
  cities, one-day glitches, and not-updated stale repeats. Real market-wide shocks are kept. Every removed row is in `removed_v3.csv`.
- 1-2 day gaps are forward-filled and flagged (`is_filled`); `mandi_observed_v3.csv` is the same data without them.
- Adds percentage changes, rolling stats, cross-city ratio, weather windows, calendar / Ramadan / Eid features, the target
  `future_chg_7d`, the label `spike_7d` and a date-based `split` (train to 2025-06-06, val to 2026-02-27, test after).

## Hand-off notes for Member 2
Give Member 2: `data/processed/mandi_dataset.csv`, `docs/data_dictionary.md`, `notebooks/EDA.ipynb` (+ `eda_outputs/`), `data/raw/`, `scripts/`.

1. **Pick one variant per crop** for Potato (Potato Fresh) and Garlic (Local), or model the variants separately. They are different AMIS series with different price levels.
2. **Lags and rolling windows are computed by calendar date**; gaps longer than 2 days are missing from the file.
3. `future_chg_7d` is the target. **Never use it as a feature.**
4. `quality_tier`: A real, B real but in a stale run (10+ identical prices), C forward-filled. Consider training on A (+B) only.
5. Split by date, never shuffle. The 7-day target of the last days of `train` and `val` looks into the next period; drop the last 7 days of each if you want a strict split.
6. The spike rate is not stable over time (e.g. Tomato and Potato spike more often in `test` than in `train`); evaluate per split.
7. No min/max price exists, so a "daily price spread" feature cannot be built; `roll7_std_pct` is the closest substitute.
8. Potato has real data gaps in several cities (Lahore potato ends 2026-06-19). Prices trend upward: prefer percentage-change features.

## Known limitations
- A small number of glitch rows remain, for example Faisalabad Garlic (Local) around 2026-01-16 to 2026-02-08 (price about 1.9x the other cities, 11 rows flagged by `price_vs_city_median > 1.8`).
- 285 day-to-day moves above 30 % remain; they are kept, not corrected.
- Weather is for the city centre, not for the farms that supply the market.

## Troubleshooting
- `SSL: WRONG_VERSION_NUMBER` -> use `http://`, not `https://`.
- Scraper stopped halfway -> run `02_scrape_amis.py` again; finished requests are skipped (`data/raw/amis_progress.csv`). Failed requests are listed in `data/raw/amis_failed_requests.csv` (created only if something failed).
- AMIS layout changed -> re-run `01b_probe_price_trends.py` and `01c_try_ssrs_export.py` to see what changed.
- `ModuleNotFoundError: hijridate` -> `pip install -r scripts/requirements.txt`.
