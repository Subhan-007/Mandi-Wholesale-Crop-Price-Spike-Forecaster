# Data Dictionary - Mandi Project Phase 1

Scope: 5 crops (Onion, Tomato, Potato, Garlic, Green Chilli) x 5 Punjab cities (Lahore, Faisalabad, Rawalpindi, Multan, Gujranwala), daily, 2022-01-01 to 2026-11-20.
Sources: AMIS Punjab (http://www.amis.pk, report "Price Trends") for prices; Open-Meteo Historical Weather API for weather.

## 1. `data/processed/mandi_dataset.csv` - the clean merged dataset (use this one)

One row = one city, one crop, one calendar day. Sorted by `city`, `crop`, `date`.

| Column | Type | Unit | Description |
|---|---|---|---|
| `date` | date (YYYY-MM-DD) | - | Calendar day of the price. |
| `city` | text | - | Lahore, Faisalabad, Rawalpindi, Multan or Gujranwala (AMIS market name). |
| `crop` | text | - | Onion, Tomato, Potato, Garlic or Green Chilli. |
| `price_source` | text | - | AMIS series the price came from: `Onion`, `Tomato`, `Green Chilli`, `Potato Fresh`, `Potato Store`, `Garlic (Local)`, `Garlic (China)`. Potato and Garlic can switch series (see `is_adjusted`). |
| `price_pkr_per_100kg` | float | PKR per 100 kg | Wholesale price. AMIS publishes **one** price per day (not separate min/max/usual). |
| `is_filled` | 0/1 | - | 1 = no price was published that day (Sunday / short holiday); the previous available price (up to 2 days back) was repeated. |
| `is_adjusted` | 0/1 | - | 1 = price comes from a backup series (Potato Store for Potato, Garlic (China) for Garlic) multiplied by the typical primary/backup price ratio of the surrounding overlapping days. It is an estimate, not an exact AMIS value. |
| `is_stale` | 0/1 | - | 1 = this price is part of a run of **10 or more consecutive trading days with exactly the same price** in AMIS. Probably AMIS not updating the rate, so the price may not reflect the real market; the model team can down-weight or drop these rows. |
| `rain_mm` | float | mm | Total precipitation of that day in that city (Open-Meteo `precipitation_sum`). |
| `temp_max_c` | float | deg C | Daily maximum air temperature at 2 m. |
| `temp_min_c` | float | deg C | Daily minimum air temperature at 2 m. |
| `temp_mean_c` | float | deg C | Daily mean air temperature at 2 m. |

## 2. Raw files (`data/raw/`)

### `amis_prices_raw.csv`
| Column | Description |
|---|---|
| `date`, `city`, `crop` | As above. |
| `crop_variant` | Name of the AMIS series (e.g. `Potato Fresh`). |
| `amis_crop_id` | AMIS dropdown ID of that series (Onion 22, Tomato 25, Potato Fresh 20, Potato Store 21, Garlic (China) 23, Garlic (Local) 69, Green Chilli 80). |
| `price` | Price exactly as exported by AMIS; blank = AMIS listed the day without a price. |
| `unit_raw` | Unit text from AMIS (`Rs./100kg`). **Blank for Green Chilli**; treated as `Rs./100kg` because its price level matches the other crops. |

`exports/` holds the untouched CSV downloaded for every request. `amis_progress.csv` / `amis_failed_requests.csv` are scraper bookkeeping files.

### `weather_raw.csv`
`date, city, precipitation_sum (mm), temperature_2m_max, temperature_2m_min, temperature_2m_mean (deg C)`. One row per city per day, no gaps. City coordinates (Lahore 31.55N 74.34E, Faisalabad 31.45N 73.14E, Rawalpindi 33.57N 73.02E, Multan 30.16N 71.52E, Gujranwala 32.19N 74.19E); time zone Asia/Karachi.

## 3. Cleaning rules (script `04_clean_merge.py`)

1. Prices converted to PKR per 100 kg (all raw data was already per 100 kg).
2. Blank, zero or negative prices removed; exact duplicates removed (none found).
3. **Outliers (two tests):** (a) within each AMIS series, a price more than 2.5x or less than 0.4x the median of the surrounding 15 trading days is removed (typing errors); (b) **cross-city test** - a price is removed if it is more than 1.7x (or less than 0.55x) the median of the *other* cities on the same day **and** more than 30 % away from the same city's own 31-day median. Because both must hold, real market-wide shocks (all cities move together) are kept and only city-specific glitches are removed. Removed rows and the reason are listed in `data/processed/outliers_removed.csv`.
4. **Primary + backup series:** gaps of more than 2 days in the primary series (Potato Fresh, Garlic (Local)) are filled from the backup series (Potato Store, Garlic (China)). The ratio primary/backup is measured on overlapping days at the start and at the end of each gap and interpolated across the gap, so the filled values join the primary series **without a jump** at either end (`is_adjusted = 1`). Gaps are skipped if the ratio is implausible (outside 0.25-4).
5. **Calendar:** a daily calendar is built for each city/crop. Gaps of **1-2 days** are filled with the previous price (`is_filled = 1`).
6. **Stale runs** (10+ trading days with an identical price) are flagged in `is_stale`; they are kept, not changed.
7. **Gaps longer than 2 days are not filled** and those days are absent from the file.
8. Weather merged on (`city`, `date`); no price row was without weather.

The numbers of removed / filled / adjusted rows are in `data/processed/cleaning_report.txt`.

## 4. Known limitations

- Only one price per day exists, so min/max/usual price and a daily price spread are **not available**.
- Potato has real gaps in several cities (AMIS publishes "Potato Fresh" seasonally and "Potato Store" mainly for Faisalabad); Lahore potato ends 2026-06-19. See the coverage chart in the EDA notebook.
- Because long gaps are left out, row position is not equal to calendar position: compute lags/rolling windows **by date**.
- Prices rise over the years; percentage-change features generalise better than raw price levels.
- Weather is for the city centre, not for the farms that supply the market.
