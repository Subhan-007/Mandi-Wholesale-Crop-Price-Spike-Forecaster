"""
Subtask 4 - Data Cleaning & Integration.

    python scripts\\04_clean_merge.py

Input : data/raw/amis_prices_raw.csv, data/raw/weather_raw.csv
Output: data/processed/mandi_dataset.csv        (the clean merged dataset for Member 2)
        data/processed/outliers_removed.csv     (every price removed as an outlier)
        data/processed/cleaning_report.txt      (counts of everything that was changed)

Steps: types -> units (PKR / 100 kg) -> drop invalid prices -> remove duplicates ->
       outliers: (1) against the series' own recent level, (2) against the OTHER cities (city-specific glitches only;
       market-wide shocks are kept) -> combine primary + backup series (backup rescaled so it joins the primary
       smoothly at both ends of every gap) -> daily calendar, fill 1-2 day gaps -> flag stale runs -> merge weather.
"""
import re

import numpy as np
import pandas as pd

from config import *  # noqa

MAX_FILL_DAYS = 2          # forward-fill gaps of at most this many days (Sundays / short holidays)
OUT_WINDOW = 15            # rolling-median window (trading days) for outlier detection
OUT_HIGH, OUT_LOW = 2.5, 0.4   # price > 2.5x or < 0.4x the local median = outlier (typing error)
MIN_OVERLAP = 20           # days where primary & backup both exist, needed to use the backup
CROSS_HIGH, CROSS_LOW = 1.6, 0.62   # 5-day median of price/median(other cities)
DAY_HIGH, DAY_LOW = 1.45, 0.70      # and the day itself must also be outside this
STALE_MIN = 10             # >= this many consecutive trading days with the identical price -> is_stale = 1

REPORT = []


def say(msg=""):
    print(msg)
    REPORT.append(str(msg))


def unit_factor(unit):
    """Multiplier converting a price quoted in `unit` to PKR per 100 kg."""
    u = str(unit).lower().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)kg", u)
    if m:
        return 100.0 / float(m.group(1))
    if "kg" in u:
        return 100.0
    return 1.0


def remove_outliers(g):
    med = g["price"].rolling(OUT_WINDOW, center=True, min_periods=5).median()
    ratio = g["price"] / med
    is_out = (ratio > OUT_HIGH) | (ratio < OUT_LOW)
    removed = g[is_out].copy()
    removed["local_median"] = med[is_out]
    return g[~is_out], removed


def cross_city_filter(df):
    """Remove city-specific anomalies that persist for several days (price vs median of the OTHER cities).
    Market-wide shocks move all cities together, so they are kept. Rows are removed, not corrected."""
    keep_parts, removed = [], []
    for (crop, variant), g in df.groupby(["crop", "crop_variant"]):
        pv = g.pivot_table(index="date", columns="city", values="price")
        others = pd.DataFrame({c: pv.drop(columns=c).median(axis=1) for c in pv.columns})
        cnt = pd.DataFrame({c: pv.drop(columns=c).notna().sum(axis=1) for c in pv.columns})
        r = pv / others
        r5 = r.rolling(5, center=True, min_periods=2).median()
        bad = ((r5 > CROSS_HIGH) | (r5 < CROSS_LOW)) & ((r > DAY_HIGH) | (r < DAY_LOW)) & (cnt >= 3)
        bad = bad.stack()
        bad = bad[bad].index
        flag = g.set_index(["date", "city"]).index.isin(bad)
        rr = g[flag].copy()
        rr["local_median"] = np.nan
        removed.append(rr.assign(reason="cross_city"))
        keep_parts.append(g[~flag])
    return pd.concat(keep_parts), pd.concat(removed)


def _edge_ratio(orig, sv, idx, last):
    """median primary/backup ratio on overlap days among idx (last=True -> the 5 closest to the end of idx)."""
    idx = [k for k in idx if np.isfinite(orig[k]) and np.isfinite(sv[k]) and sv[k] > 0]
    if not idx:
        return np.nan
    idx = idx[-5:] if last else idx[:5]
    return float(np.median([orig[k] / sv[k] for k in idx]))


def bridge(P, S, name, source, adj):
    """Fill gaps (> MAX_FILL_DAYS) of primary P from backup S. The ratio primary/backup is measured at the start and
    at the end of every gap and interpolated across it, so the filled values join the primary without a jump."""
    both = P.notna() & S.notna()
    if both.sum() < MIN_OVERLAP:
        return P, source, adj
    glob = float((P[both] / S[both]).median())
    orig, sv = P.to_numpy(float), S.to_numpy(float)
    out, src, ad = orig.copy(), source.to_numpy(object).copy(), adj.to_numpy(int).copy()
    miss, n, i = np.isnan(orig), len(orig), 0
    while i < n:
        if not miss[i]:
            i += 1
            continue
        j = i
        while j < n and miss[j]:
            j += 1                                                   # gap = [i, j)
        L = j - i
        if L > MAX_FILL_DAYS:
            ra = _edge_ratio(orig, sv, range(max(0, i - 10), i), last=True)
            rb = _edge_ratio(orig, sv, range(j, min(n, j + 10)), last=False)
            if np.isnan(ra) and np.isnan(rb):
                ra = rb = glob
            ra = rb if np.isnan(ra) else ra
            rb = ra if np.isnan(rb) else rb
            if 0.25 <= ra <= 4 and 0.25 <= rb <= 4:
                for k in range(i, j):
                    if np.isfinite(sv[k]):
                        out[k] = sv[k] * (ra + (rb - ra) * (k - i + 0.5) / L)
                        src[k], ad[k] = name, 1
        i = j
    return pd.Series(out, index=P.index), pd.Series(src, index=P.index, dtype=object), pd.Series(ad, index=P.index)


def combine_variants(series_by_variant, order):
    """Returns daily (price, source, adjusted) Series on the full date range; NaN price where nothing is available."""
    variants = [v for v in order if v in series_by_variant]
    full = pd.date_range(min(s.index.min() for s in series_by_variant.values()),
                         max(s.index.max() for s in series_by_variant.values()), name="date")
    P = series_by_variant[variants[0]].reindex(full)
    source = pd.Series(np.where(P.notna(), variants[0], None), index=full, dtype=object)
    adj = pd.Series(0, index=full)
    for v in variants[1:]:
        P, source, adj = bridge(P, series_by_variant[v].reindex(full), v, source, adj)
    return P, source, adj


SPIKE_HIGH, SPIKE_LOW = 1.8, 0.55   # a single day this far from the other cities is a glitch


def final_cross_check(df, passes=8):
    """Cross-city check on the FINAL daily series (after backup fill), per crop. Rows are dropped, never refilled."""
    removed = []
    for _ in range(passes):
        keep = []
        for crop, g in df.groupby("crop"):
            pv = g.pivot_table(index="date", columns="city", values="price")
            others = pd.DataFrame({c: pv.drop(columns=c).median(axis=1) for c in pv.columns})
            cnt = pd.DataFrame({c: pv.drop(columns=c).notna().sum(axis=1) for c in pv.columns})
            r = pv / others
            r5 = r.rolling(5, center=True, min_periods=2).median()
            persist = ((r5 > CROSS_HIGH) | (r5 < CROSS_LOW)) & ((r > DAY_HIGH) | (r < DAY_LOW))
            spike = (r > SPIKE_HIGH) | (r < SPIKE_LOW)
            bad = ((persist | spike) & (cnt >= 3)).stack()
            idx = bad[bad].index
            flag = g.set_index(["date", "city"]).index.isin(idx)
            removed.append(g[flag])
            keep.append(g[~flag])
        n_before = len(df)
        df = pd.concat(keep, ignore_index=True)
        if len(df) == n_before:
            break
    return df, pd.concat(removed, ignore_index=True)


def main():
    # ------------------------------------------------------------ load prices
    df = pd.read_csv(RAW_PRICES_CSV)
    say(f"Raw price rows: {len(df)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["city"] = df["city"].str.strip()
    df["crop"] = df["crop"].str.strip()
    say(f"Rows with unreadable date: {int(df['date'].isna().sum())} (dropped)")
    df = df.dropna(subset=["date"])

    # ------------------------------------------------------------ units
    blank = df["unit_raw"].isna() | (df["unit_raw"].astype(str).str.strip() == "")
    say(f"\nUnits found in raw data:\n{df['unit_raw'].value_counts(dropna=False).to_string()}")
    say(f"Rows with BLANK unit (AMIS leaves it empty for: {sorted(df[blank]['crop_variant'].unique())}) "
        f"-> assumed Rs./100kg because their price level matches the other crops: {int(blank.sum())}")
    df["unit_raw"] = df["unit_raw"].where(~blank, "Rs./100kg")
    df["price"] = df["price"] * df["unit_raw"].map(unit_factor)

    # ------------------------------------------------------------ missing / invalid prices
    bad = df["price"].isna() | (df["price"] <= 0)
    say(f"\nMissing or non-positive prices removed: {int(bad.sum())}")
    df = df[~bad]

    # ------------------------------------------------------------ duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    say(f"Exact duplicate rows removed: {n0 - len(df)}")
    key = ["city", "crop", "crop_variant", "date"]
    n1 = len(df)
    df = df.groupby(key, as_index=False).agg(price=("price", "mean"))
    say(f"Rows merged because the same day had conflicting prices: {n1 - len(df)}")

    # ------------------------------------------------------------ outliers
    kept, removed = [], []
    for _, g in df.sort_values("date").groupby(["city", "crop", "crop_variant"]):
        k, r = remove_outliers(g)
        kept.append(k)
        removed.append(r.assign(reason="own_median"))
    df = pd.concat(kept)
    n_own = sum(len(r) for r in removed)
    df, r_cross = cross_city_filter(df)
    removed.append(r_cross)
    out = pd.concat(removed)
    out.to_csv(PROCESSED_DIR / "outliers_removed.csv", index=False)
    say(f"\nOutliers removed: {len(out)} = {n_own} against the series' own level + {len(r_cross)} city-specific "
        f"glitches vs the other cities (listed in outliers_removed.csv)")

    # ------------------------------------------------------------ primary + backup series, daily calendar
    order_by_crop = {crop: [name for name, _ in variants] for crop, variants in CROP_IDS.items()}
    frames, n_backup, n_long_gap, n_filled = [], 0, 0, 0
    for (city, crop), g in df.groupby(["city", "crop"]):
        series = {v: gg.set_index("date")["price"].sort_index() for v, gg in g.groupby("crop_variant")}
        price, source, adjusted = combine_variants(series, order_by_crop[crop])
        out_df = pd.DataFrame({"price": price, "price_source": source, "is_adjusted": adjusted})
        out_df = out_df.loc[out_df["price"].first_valid_index():out_df["price"].last_valid_index()]
        was_missing = out_df["price"].isna()
        out_df["price"] = out_df["price"].ffill(limit=MAX_FILL_DAYS)           # previous trading day's price
        out_df["price_source"] = out_df["price_source"].ffill(limit=MAX_FILL_DAYS)
        out_df["is_adjusted"] = out_df["is_adjusted"].ffill(limit=MAX_FILL_DAYS).fillna(0).astype(int)
        out_df["is_filled"] = (was_missing & out_df["price"].notna()).astype(int)
        # stale = >= STALE_MIN consecutive trading days (real, unfilled rows) with exactly the same price
        nf = out_df[out_df["is_filled"] == 0]["price"].dropna()
        run = (nf != nf.shift()).cumsum()
        stale = pd.Series(0, index=out_df.index)
        for r_id, sz in nf.groupby(run).size().items():
            if sz >= STALE_MIN:
                d = nf.index[run == r_id]
                stale.loc[d.min():d.max()] = 1
        out_df["is_stale"] = stale
        n_backup += int(out_df["is_adjusted"].sum())
        n_filled += int(out_df["is_filled"].sum())
        n_long_gap += int(out_df["price"].isna().sum())
        out_df["city"], out_df["crop"] = city, crop
        frames.append(out_df.dropna(subset=["price"]).reset_index())
    df = pd.concat(frames, ignore_index=True)
    df, r_final = final_cross_check(df)
    r_final.to_csv(PROCESSED_DIR / "final_cross_removed.csv", index=False)
    say(f"Final cross-city check removed {len(r_final)} rows (incl. backup-filled days): final_cross_removed.csv")
    say(f"Days filled from a BACKUP series (scaled to the primary): {n_backup}")
    say(f"Gap days filled with the previous price (<= {MAX_FILL_DAYS} days): {n_filled}")
    say(f"Days still missing after filling (longer gaps, left out of the dataset): {n_long_gap}")

    # ------------------------------------------------------------ merge weather
    w = pd.read_csv(RAW_WEATHER_CSV, parse_dates=["date"])
    w = w.rename(columns={"precipitation_sum": "rain_mm", "temperature_2m_max": "temp_max_c",
                          "temperature_2m_min": "temp_min_c", "temperature_2m_mean": "temp_mean_c"})
    wcols = ["rain_mm", "temp_max_c", "temp_min_c", "temp_mean_c"]
    w = w.sort_values(["city", "date"])
    w[wcols] = w.groupby("city")[wcols].transform(lambda s: s.interpolate(limit=3, limit_direction="both"))
    df = df.merge(w[["city", "date"] + wcols], on=["city", "date"], how="left")
    no_weather = df[wcols].isna().any(axis=1)
    say(f"\nPrice rows without weather (dates outside the weather range): {int(no_weather.sum())} (dropped)")
    df = df[~no_weather]

    # ------------------------------------------------------------ final dataset
    df = df.rename(columns={"price": "price_pkr_per_100kg"})
    df["price_pkr_per_100kg"] = df["price_pkr_per_100kg"].round(2)
    cols = ["date", "city", "crop", "price_source", "price_pkr_per_100kg", "is_filled", "is_adjusted", "is_stale"] + wcols
    df = df[cols].sort_values(["city", "crop", "date"]).reset_index(drop=True)
    df.to_csv(FINAL_DATASET_CSV, index=False)

    say(f"\nFINAL: {len(df)} rows, {df['city'].nunique()} cities, {df['crop'].nunique()} crops "
        f"-> {FINAL_DATASET_CSV}")
    summary = df.groupby(["crop", "city"]).agg(
        rows=("date", "size"), first=("date", "min"), last=("date", "max"),
        filled_pct=("is_filled", lambda s: round(100 * s.mean(), 1)),
        backup_pct=("is_adjusted", lambda s: round(100 * s.mean(), 1)),
        stale_pct=("is_stale", lambda s: round(100 * s.mean(), 1)),
        median_price=("price_pkr_per_100kg", "median"))
    say("\n" + summary.to_string())
    (PROCESSED_DIR / "cleaning_report.txt").write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()
