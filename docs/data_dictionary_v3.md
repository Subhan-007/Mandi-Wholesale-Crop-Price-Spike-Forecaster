# Data Dictionary - v3 (replaces section 1 of data_dictionary.md; the raw-file and cleaning-rule sections of the old file describe v2)

Built by `scripts/07_build_v3.py` from the raw AMIS + Open-Meteo files.

## Files (data/processed/)
| File | Content |
|---|---|
| `mandi_dataset.csv` | **Main deliverable.** Daily series with 1-2 day gaps forward-filled and flagged (`is_filled = 1`), as the project plan asks. 49,252 rows. |
| `mandi_observed_v3.csv` | Same rows without the forward-filled days: real AMIS observations only. 43,512 rows. |
| `removed_v3.csv` | Every row removed and why (`own_median`, `cross_city`, `one_day_glitch`, `stale_repeat`). |
| `cleaning_report_v3.txt` | Counts, thresholds, v2 vs v3 comparison. |
| `archive_v2/` | The previous version, untouched. |

## What changed vs v2
- **No spliced series.** Potato Fresh / Potato Store and Garlic (Local) / Garlic (China) are separate series in column `variant`. Choose one variant per crop (e.g. Potato Fresh, Garlic (Local)) or model them separately. `is_adjusted` and `price_source` no longer exist.
- **New glitch filter** for one-day prices far from the city's own neighbours while the other cities did not move (64 rows).
- **Stale runs:** 38 runs where the other cities clearly moved were removed except their first day (621 rows); the other stale runs stay with `is_stale = 1`.
- **Forward-filled days are flagged** (`is_filled`, `quality_tier = C`); `mandi_observed_v3.csv` has none.

## Columns
| Column | Description |
|---|---|
| `date`, `city`, `crop`, `variant` | Series key = (`city`, `crop`, `variant`). |
| `price_pkr_per_100kg` | AMIS wholesale price, one per day. |
| `quality_tier` | `A` real and not stale; `B` real but in a stale run; `C` forward-filled (daily file only). |
| `is_filled`, `is_stale` | Flags as in v2. |
| `pct_chg_1d / 7d / 14d` | Change vs the price exactly 1 / 7 / 14 calendar days earlier (uses filled prices, NaN across long gaps). |
| `roll7_mean`, `roll7_std_pct` | 7-day rolling mean and std/mean. |
| `vol_14d` | Std of the 1-day % change over 14 days. |
| `price_vs_city_median` | Price / median price of all cities for the same crop, variant and day (needs 3+ cities). |
| `rain_mm`, `temp_*_c` | Same-day weather. |
| `rain_7d_mm`, `rain_14d_mm`, `temp_mean_7d_c` | Rolling weather windows. |
| `days_since_rain` | Days since rain >= 1 mm (capped at 30). `heat_day` = max temp >= 40 C. |
| `month`, `day_of_week`, `day_of_year` | Calendar. |
| `is_ramadan`, `days_to_next_eid`, `days_since_last_eid`, `is_pre_eid_10d` | Islamic calendar (Umm al-Qura shifted 1 day for Pakistan; accurate to +/- 1 day). Eid al-Fitr and Eid al-Adha; capped at 60. |
| `future_chg_7d` | **Target:** price 7 days ahead / price - 1. Never use as a feature. |
| `spike_7d` | 1 if `future_chg_7d` > `spike_threshold`. NaN when the future price is unknown. |
| `spike_threshold` | Mean + 1.5 std of `future_chg_7d` per (crop, variant), computed on the training period only. |
| `split` | `train` (to 2025-06-06), `val` (to 2026-02-27), `test` (after). Split by date, never shuffle. |

## Known limits (still true)
- One price per day: no min / max spread. Prices are AMIS only; no fuel, exchange-rate or CPI data yet.
- 285 day-to-day moves above 30% remain; 98 are market-wide, 187 are unclear (often AMIS catching up after a stale price). Kept, not corrected.
- Potato Store has short series in several cities (Rawalpindi 186 rows, 70% stale); Lahore potato ends 2026-06-19.
- The EDA notebook and `eda_outputs/` were regenerated on v3 (primary series: Potato Fresh, Garlic (Local)).
