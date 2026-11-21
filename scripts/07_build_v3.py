"""
Phase 1 - v3 dataset build.   Run from anywhere:   python scripts/07_build_v3.py

Input : data/raw/amis_prices_raw.csv, data/raw/weather_raw.csv
Output (data/processed/):
    mandi_dataset.csv          MAIN deliverable: daily series with 1-2 day gaps forward-filled (is_filled = 1) + features
    mandi_observed_v3.csv      same rows without the forward-filled days (real AMIS observations only)
    removed_v3.csv             every row removed, with the reason
    cleaning_report_v3.txt     counts + before/after comparison with v2

What changed compared with v2 (04_clean_merge.py):
  1. NO splicing of backup series: Potato Fresh / Potato Store and Garlic (Local) / Garlic (China) are separate
     series (column `variant`), so there are no artificial scaled prices and no jumps at join points.
  2. Extra glitch filter: a one-day price that is >1.5x (or <0.67x) its own neighbours while the OTHER cities did not
     move is removed (real market-wide shocks move all cities and are kept).
  3. Stale runs (10+ identical prices): if the other cities clearly moved (>10 %) during the run, the repeated
     days are removed (AMIS did not update); otherwise they stay and are flagged (is_stale = 1).
  4. Forward-filled days (1-2 day gaps, as the project plan asks) are flagged; a file without them is also written.
  5. New features: % changes, rolling stats, cross-city ratio, weather windows, calendar / Ramadan / Eid,
     7-day-ahead target + spike label (threshold from the first 70 % of dates only), train/val/test split,
     quality_tier (A real, B real but stale-flagged, C forward-filled).
"""
import importlib

import numpy as np
import pandas as pd
from hijridate import Gregorian

from config import *  # noqa

cm = importlib.import_module("04_clean_merge")      # re-use unit conversion, own-median and cross-city filters

MAX_FILL_DAYS = 2
REV_HIGH, REV_LOW = 1.5, 1 / 1.5      # one-day glitch: price vs median of its 3 previous + 3 next observations
OTHERS_BAND = (0.85, 1.18)            # ... and the other cities' own ratio stays inside this band (they did not move)
MIN_OTHERS = 2
STALE_MIN = 10
STALE_OTHERS_MOVE = 0.10              # other cities moved by more than 10 % during the run -> run is not real
SPIKE_K = 1.5                         # spike: 7-day-ahead change > mean + 1.5 std (per series, first 70 % of dates)

REPORT = []


def say(msg=""):
    print(msg)
    REPORT.append(str(msg))


# --------------------------------------------------------------------------------------------- loading
def load_raw():
    df = pd.read_csv(RAW_PRICES_CSV)
    say(f"Raw price rows: {len(df)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["city"], df["crop"] = df["city"].str.strip(), df["crop"].str.strip()
    df = df.dropna(subset=["date"])
    blank = df["unit_raw"].isna() | (df["unit_raw"].astype(str).str.strip() == "")
    df["unit_raw"] = df["unit_raw"].where(~blank, "Rs./100kg")          # Green Chilli: blank unit, same price level
    df["price"] = df["price"] * df["unit_raw"].map(cm.unit_factor)
    bad = df["price"].isna() | (df["price"] <= 0)
    say(f"Missing / non-positive prices removed: {int(bad.sum())}")
    df = df[~bad].drop_duplicates()
    df = df.groupby(["city", "crop", "crop_variant", "date"], as_index=False).agg(price=("price", "mean"))
    return df


# --------------------------------------------------------------------------------------------- glitch filter
def reversal_filter(df, passes=3):
    """Remove one-day glitches: far from the city's own neighbouring observations while the other cities stayed put."""
    removed = []
    for _ in range(passes):
        keep, n_rem = [], 0
        for (crop, variant), g in df.groupby(["crop", "crop_variant"]):
            pv = g.pivot_table(index="date", columns="city", values="price").sort_index()
            ratio = pd.DataFrame(np.nan, index=pv.index, columns=pv.columns)
            for c in pv.columns:
                s = pv[c].dropna()
                nb = pd.concat([s.shift(k) for k in (1, 2, 3, -1, -2, -3)], axis=1)
                ratio.loc[s.index, c] = (s / nb.median(axis=1)).where(nb.notna().sum(axis=1) >= 3)
            others = pd.DataFrame({c: ratio.drop(columns=c).median(axis=1) for c in ratio.columns})
            cnt = pd.DataFrame({c: ratio.drop(columns=c).notna().sum(axis=1) for c in ratio.columns})
            bad = ((ratio > REV_HIGH) | (ratio < REV_LOW)) & others.ge(OTHERS_BAND[0]) & others.le(OTHERS_BAND[1]) \
                & (cnt >= MIN_OTHERS)
            bad = bad.stack()
            flag = g.set_index(["date", "city"]).index.isin(bad[bad].index)
            removed.append(g[flag].assign(reason="one_day_glitch"))
            keep.append(g[~flag])
            n_rem += int(flag.sum())
        df = pd.concat(keep, ignore_index=True)
        if n_rem == 0:
            break
    return df, pd.concat(removed, ignore_index=True)


# --------------------------------------------------------------------------------------------- stale runs
def stale_filter(df):
    """Flag runs of >= STALE_MIN identical consecutive prices; drop the repeated days when other cities moved."""
    df = df.sort_values(["crop", "crop_variant", "city", "date"]).copy()
    df["is_stale"] = 0
    drop_idx, n_runs, n_conf = [], 0, 0
    for (crop, variant), g in df.groupby(["crop", "crop_variant"]):
        pv = g.pivot_table(index="date", columns="city", values="price").sort_index()
        pvf = pv.ffill(limit=7)
        for city, gc in g.groupby("city"):
            gc = gc.sort_values("date")
            run = (gc["price"] != gc["price"].shift()).cumsum()
            for _, r in gc.groupby(run):
                if len(r) < STALE_MIN:
                    continue
                n_runs += 1
                d0, d1 = r["date"].iloc[0], r["date"].iloc[-1]
                mv = [pvf.at[d1, o] / pvf.at[d0, o] for o in pv.columns
                      if o != city and pd.notna(pvf.at[d0, o]) and pd.notna(pvf.at[d1, o])]
                if len(mv) >= 2 and abs(np.median(mv) - 1) > STALE_OTHERS_MOVE:
                    drop_idx.extend(r.index[1:])           # keep the first (real) day, drop the repeats
                    n_conf += 1
                else:
                    df.loc[r.index, "is_stale"] = 1
    removed = df.loc[drop_idx].assign(reason="stale_repeat")
    say(f"Stale runs (>= {STALE_MIN} identical prices): {n_runs}; confirmed not-updated (others moved) and "
        f"repeated days removed: {n_conf} runs / {len(removed)} rows; the rest stay, flagged is_stale = 1")
    return df.drop(index=drop_idx), removed


# --------------------------------------------------------------------------------------------- calendar features
def islamic_calendar(dates):
    """Ramadan flag and distance to Eid al-Fitr / Eid al-Adha. Pakistan usually sights the moon ~1 day after
    Saudi Arabia, so the Umm al-Qura date of (day - 1) is used. Accuracy: +/- 1 day."""
    scan = pd.date_range("2020-10-01", "2027-12-31")
    fitr, adha, ram = [], [], set()
    for d in scan:
        h = Gregorian.fromdate((d - pd.Timedelta(days=1)).date()).to_hijri()
        if h.month == 10 and h.day == 1:
            fitr.append(d)
        if h.month == 12 and h.day == 10:
            adha.append(d)
        if h.month == 9:
            ram.add(d)
    eids = pd.DatetimeIndex(sorted(fitr + adha))
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    out["is_ramadan"] = out.index.isin(ram).astype(int)
    pos = eids.searchsorted(out.index)
    nxt = eids[np.minimum(pos, len(eids) - 1)]
    prv = eids[np.maximum(pos - 1, 0)]
    # a date equal to an Eid counts as "days_to_next_eid = 0"
    out["days_to_next_eid"] = np.minimum((nxt - out.index).days, 60)
    out["days_since_last_eid"] = np.minimum(np.where(out.index == nxt, 0, (out.index - prv).days), 60)
    out["is_pre_eid_10d"] = (out["days_to_next_eid"] <= 10).astype(int)
    return out


# --------------------------------------------------------------------------------------------- build
def build_daily(obs):
    frames = []
    for (city, crop, variant), g in obs.groupby(["city", "crop", "crop_variant"]):
        g = g.sort_values("date")
        s = g.set_index("date")["price"]
        full = pd.date_range(s.index.min(), s.index.max(), name="date")
        d = pd.DataFrame({"raw": s.reindex(full)})
        d["price"] = d["raw"].ffill(limit=MAX_FILL_DAYS)
        d["is_filled"] = (d["raw"].isna() & d["price"].notna()).astype(int)
        d["is_stale"] = g.set_index("date")["is_stale"].reindex(full).fillna(0).astype(int)
        p = d["price"]
        d["pct_chg_1d"] = p / p.shift(1) - 1
        d["pct_chg_7d"] = p / p.shift(7) - 1
        d["pct_chg_14d"] = p / p.shift(14) - 1
        d["roll7_mean"] = p.rolling(7, min_periods=4).mean()
        d["roll7_std_pct"] = p.rolling(7, min_periods=4).std() / d["roll7_mean"]
        d["vol_14d"] = d["pct_chg_1d"].rolling(14, min_periods=7).std()
        d["future_chg_7d"] = p.shift(-7) / p - 1
        d["city"], d["crop"], d["variant"] = city, crop, variant
        frames.append(d.dropna(subset=["price"]).drop(columns="raw").reset_index())
    return pd.concat(frames, ignore_index=True)


def weather():
    w = pd.read_csv(RAW_WEATHER_CSV, parse_dates=["date"]).rename(columns={
        "precipitation_sum": "rain_mm", "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c", "temperature_2m_mean": "temp_mean_c"}).sort_values(["city", "date"])
    wc = ["rain_mm", "temp_max_c", "temp_min_c", "temp_mean_c"]
    w[wc] = w.groupby("city")[wc].transform(lambda s: s.interpolate(limit=3, limit_direction="both"))
    g = w.groupby("city")
    w["rain_7d_mm"] = g["rain_mm"].transform(lambda s: s.rolling(7, min_periods=7).sum())
    w["rain_14d_mm"] = g["rain_mm"].transform(lambda s: s.rolling(14, min_periods=14).sum())
    w["temp_mean_7d_c"] = g["temp_mean_c"].transform(lambda s: s.rolling(7, min_periods=7).mean())
    w["heat_day"] = (w["temp_max_c"] >= 40).astype(int)

    def since_rain(s):
        last, out = None, []
        for i, v in enumerate(s.to_numpy()):
            if v >= 1:
                last = i
            out.append(min(i - last, 30) if last is not None else 30)
        return pd.Series(out, index=s.index)
    w["days_since_rain"] = g["rain_mm"].transform(since_rain)
    return w


def extreme_share(df, key, filled_col="is_filled", price="price", max_gap=3):
    """Share of day-to-day changes > 30 % / > 50 % between consecutive REAL observations <= max_gap days apart."""
    d = df[df[filled_col] == 0].sort_values(key + ["date"]).copy()
    d["chg"] = d.groupby(key)[price].pct_change().abs()
    d["gap"] = d.groupby(key)["date"].diff().dt.days
    d = d[d["gap"] <= max_gap]
    return len(d), (d["chg"] > 0.30).mean() * 100, (d["chg"] > 0.50).mean() * 100


def main():
    df = load_raw()
    removed = []

    kept, own = [], []
    for _, g in df.sort_values("date").groupby(["city", "crop", "crop_variant"]):
        k, r = cm.remove_outliers(g)
        kept.append(k)
        own.append(r.assign(reason="own_median"))
    df, rc = cm.cross_city_filter(pd.concat(kept))
    own = pd.concat(own)
    removed += [own, rc.assign(reason="cross_city")]
    say(f"Outliers vs own level: {len(own)} | vs other cities (persistent): {len(rc)}")

    df, rr = reversal_filter(df)
    removed.append(rr)
    say(f"One-day glitches removed (others did not move): {len(rr)}")

    df, rs = stale_filter(df)
    removed.append(rs)

    obs = df.copy()
    daily = build_daily(obs)

    # weather + calendar + cross-city ratio
    w = weather()
    daily = daily.merge(w.drop(columns=[]), on=["city", "date"], how="left")
    n0 = len(daily)
    daily = daily.dropna(subset=["rain_mm", "temp_mean_c"])
    say(f"Rows dropped for missing weather: {n0 - len(daily)}")
    daily["price_vs_city_median"] = daily["price"] / daily.groupby(["crop", "variant", "date"])["price"].transform("median")
    n_cities = daily.groupby(["crop", "variant", "date"])["price"].transform("count")
    daily.loc[n_cities < 3, "price_vs_city_median"] = np.nan
    cal = islamic_calendar(daily["date"].unique())
    daily = daily.merge(cal, left_on="date", right_index=True, how="left")
    daily["month"], daily["day_of_week"], daily["day_of_year"] = daily["date"].dt.month, daily["date"].dt.dayofweek, daily["date"].dt.dayofyear

    # split, spike label (threshold from the first 70 % of dates only), quality tier
    dates = np.sort(daily["date"].unique())
    c70, c85 = dates[int(len(dates) * 0.70)], dates[int(len(dates) * 0.85)]
    daily["split"] = np.where(daily["date"] <= c70, "train", np.where(daily["date"] <= c85, "val", "test"))
    tr = daily[(daily["split"] == "train") & (daily["is_filled"] == 0)]
    thr = tr.groupby(["crop", "variant"])["future_chg_7d"].agg(["mean", "std"])
    thr["spike_threshold"] = thr["mean"] + SPIKE_K * thr["std"]
    daily = daily.merge(thr[["spike_threshold"]], left_on=["crop", "variant"], right_index=True, how="left")
    daily["spike_7d"] = np.where(daily["future_chg_7d"].isna(), np.nan, (daily["future_chg_7d"] > daily["spike_threshold"]).astype(float))
    daily["quality_tier"] = np.where(daily["is_filled"] == 1, "C", np.where(daily["is_stale"] == 1, "B", "A"))

    cols = ["date", "city", "crop", "variant", "price", "quality_tier", "is_filled", "is_stale",
            "pct_chg_1d", "pct_chg_7d", "pct_chg_14d", "roll7_mean", "roll7_std_pct", "vol_14d", "price_vs_city_median",
            "rain_mm", "rain_7d_mm", "rain_14d_mm", "days_since_rain", "temp_max_c", "temp_min_c", "temp_mean_c",
            "temp_mean_7d_c", "heat_day", "month", "day_of_week", "day_of_year",
            "is_ramadan", "days_to_next_eid", "days_since_last_eid", "is_pre_eid_10d",
            "future_chg_7d", "spike_7d", "spike_threshold", "split"]
    daily = daily[cols].rename(columns={"price": "price_pkr_per_100kg"})
    daily = daily.sort_values(["city", "crop", "variant", "date"]).reset_index(drop=True)
    num = daily.select_dtypes("float").columns
    daily[num] = daily[num].round(4)
    observed = daily[daily["is_filled"] == 0].reset_index(drop=True)

    daily.to_csv(FINAL_DATASET_CSV, index=False)
    observed.to_csv(PROCESSED_DIR / "mandi_observed_v3.csv", index=False)
    rem = pd.concat(removed, ignore_index=True)[["city", "crop", "crop_variant", "date", "price", "reason"]]
    rem.to_csv(PROCESSED_DIR / "removed_v3.csv", index=False)

    say(f"\nREMOVED total: {len(rem)}  ->  {rem['reason'].value_counts().to_dict()}")
    say(f"Train/val cut-offs: {pd.Timestamp(c70).date()} / {pd.Timestamp(c85).date()}")
    say("Spike thresholds (7-day-ahead change, first 70 % of dates):\n" + (thr["spike_threshold"] * 100).round(1).to_string())
    say(f"\nFINAL observed-only file: {len(observed)} rows | mandi_dataset.csv (with 1-2 day fills): {len(daily)} rows")
    say("Quality tiers (daily file): " + daily["quality_tier"].value_counts().to_dict().__repr__())
    say("\nSpike share by split (observed): " + observed.groupby("split")["spike_7d"].mean().round(3).to_dict().__repr__())

    # ---- before / after vs v2
    v2 = pd.read_csv(PROCESSED_DIR / "archive_v2" / "mandi_dataset.csv", parse_dates=["date"])
    n2, a2, b2 = extreme_share(v2, ["city", "crop"], price="price_pkr_per_100kg")
    n3, a3, b3 = extreme_share(observed, ["city", "crop", "variant"], price="price_pkr_per_100kg")
    say("\nBEFORE / AFTER (day-to-day change between consecutive real observations <= 3 days apart)")
    say(f"v2: {n2} pairs | >30 %: {a2:.2f} % | >50 %: {b2:.2f} %    rows: {len(v2)} (filled {v2['is_filled'].mean():.1%}, "
        f"adjusted {v2['is_adjusted'].mean():.1%}, stale {v2['is_stale'].mean():.1%})")
    say(f"v3: {n3} pairs | >30 %: {a3:.2f} % | >50 %: {b3:.2f} %    rows: {len(observed)} (filled 0 %, adjusted 0 %, "
        f"stale {observed['is_stale'].mean():.1%})")
    summ = observed.groupby(["crop", "variant", "city"]).agg(rows=("date", "size"), first=("date", "min"), last=("date", "max"),
                                                              stale_pct=("is_stale", lambda s: round(100 * s.mean(), 1)))
    say("\n" + summ.to_string())
    (PROCESSED_DIR / "cleaning_report_v3.txt").write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()
