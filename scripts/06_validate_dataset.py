"""
06 - Validate mandi_dataset.csv before it is handed to the modelling team (read-only).

    python scripts\\06_validate_dataset.py

Prints a compact report (also saved to data/processed/validation_report.txt). Paste the printed report to Claude.
"""
import numpy as np
import pandas as pd

from config import *  # noqa

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
LINES, STATUS = [], []


def rnd(df):
    """round only numeric columns (dates stay untouched)"""
    return df.apply(lambda s: s.round(1) if s.dtype.kind == "f" else s)


def say(x=""):
    print(x)
    LINES.append(str(x))


def status(name, n, note=""):
    STATUS.append((name, "OK" if n == 0 else f"CHECK ({n})", note))


ds = pd.read_csv(FINAL_DATASET_CSV, parse_dates=["date"]).sort_values(["city", "crop", "date"]).reset_index(drop=True)
raw = pd.read_csv(RAW_PRICES_CSV, parse_dates=["date"])
raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
P = "price_pkr_per_100kg"
G = ["city", "crop"]

# ------------------------------------------------------------------ A. structure
say("##### A. STRUCTURE #####")
say(f"rows={len(ds)} | {ds['date'].min().date()} -> {ds['date'].max().date()} | cities={ds['city'].nunique()} crops={ds['crop'].nunique()}")
say(f"columns: {list(ds.columns)}")
dups = int(ds.duplicated(["city", "crop", "date"]).sum())
nulls = int(ds.isna().sum().sum())
nonpos = int((ds[P] <= 0).sum())
say(f"duplicate (city,crop,date): {dups} | null cells: {nulls} | prices <= 0: {nonpos}")
status("no duplicate city/crop/date", dups)
status("no null cells", nulls)
status("all prices > 0", nonpos)

# ------------------------------------------------------------------ B. match with raw AMIS data
say("\n##### B. FINAL vs RAW AMIS DATA #####")
real = ds[(ds["is_filled"] == 0) & (ds["is_adjusted"] == 0)]
m = real.merge(raw[["city", "crop", "crop_variant", "date", "price"]].rename(columns={"price": "raw_price"}),
               left_on=["city", "crop", "price_source", "date"], right_on=["city", "crop", "crop_variant", "date"], how="left")
no_raw = int(m["raw_price"].isna().sum())
diff = int(((m[P] - m["raw_price"]).abs() > 0.5).sum())
say(f"genuine rows checked: {len(real)} | not found in raw: {no_raw} | price differs from raw: {diff}")
status("genuine rows match raw AMIS prices", no_raw + diff)
prev = ds.groupby(G)[P].shift(1)
fill_bad = int(((ds["is_filled"] == 1) & ((ds[P] - prev).abs() > 0.01)).sum())
say(f"forward-filled rows not equal to previous day's price: {fill_bad}")
status("filled rows repeat previous price", fill_bad)

# ------------------------------------------------------------------ C. extreme changes
say("\n##### C. EXTREME PRICE CHANGES #####")
ds["prev_price"] = prev
ds["chg1"] = (ds[P] / ds["prev_price"] - 1) * 100
ds.loc[ds["date"].diff().dt.days != 1, "chg1"] = np.nan                      # only consecutive days
ds.loc[ds["city"] != ds["city"].shift(1), "chg1"] = np.nan
ds.loc[ds["crop"] != ds["crop"].shift(1), "chg1"] = np.nan
real_chg = ds[ds["is_filled"] == 0]
big50 = int((real_chg["chg1"].abs() > 50).sum())
big30 = int((real_chg["chg1"].abs() > 30).sum())
say(f"day-to-day changes > 30%: {big30} | > 50%: {big50}  (out of {int(real_chg['chg1'].notna().sum())})")
say("Top 12 day-to-day changes:")
say(real_chg.reindex(real_chg["chg1"].abs().sort_values(ascending=False).index)
    [["city", "crop", "date", "prev_price", P, "chg1", "price_source"]].head(12).pipe(rnd).to_string(index=False))
status("day-to-day change > 50%", big50, "look at the top rows above")

base = ds[G + ["date", P]]
fut = base.rename(columns={P: "price_next7"}).copy()
fut["date"] -= pd.Timedelta(days=7)
wk = base.merge(fut, on=G + ["date"])
wk["chg7"] = (wk["price_next7"] / wk[P] - 1) * 100
say("\n7-day change (%) by crop:")
say(wk.groupby("crop")["chg7"].describe(percentiles=[.01, .05, .95, .99]).round(1).to_string())
say("Top 8 7-day changes:")
say(wk.reindex(wk["chg7"].abs().sort_values(ascending=False).index).head(8).pipe(rnd).to_string(index=False))
big7 = int((wk["chg7"].abs() > 100).sum())
status("7-day change > 100%", big7)

# ------------------------------------------------------------------ D. series switches (Potato / Garlic)
say("\n##### D. SERIES SWITCHES (backup series joins) #####")
sw = ds[(ds["price_source"] != ds["price_source"].shift(1)) & (ds["city"] == ds["city"].shift(1)) & (ds["crop"] == ds["crop"].shift(1))]
sw = sw[sw["chg1"].notna()] if len(sw) else sw
typical = ds.loc[ds["is_adjusted"] == 0, "chg1"].abs().median()
say(f"switch points: {len(sw)} | median |jump| at switch: {sw['chg1'].abs().median() if len(sw) else 0:.1f}% | typical daily |change|: {typical:.1f}%")
jumpy = sw[sw["chg1"].abs() > 25]
if len(jumpy):
    say(jumpy[["city", "crop", "date", "price_source", "prev_price", P, "chg1"]].head(10).pipe(rnd).to_string(index=False))
status("series switch jump > 25%", len(jumpy))
adj_vol = ds.assign(adj=ds["is_adjusted"]).groupby(["crop", "adj"])["chg1"].apply(lambda s: s.abs().mean()).unstack().round(2)
say("mean |daily change| % for genuine (0) vs backup-adjusted (1) rows:")
say(adj_vol.to_string())

# ------------------------------------------------------------------ E. flat / stuck prices
say("\n##### E. STUCK PRICES (same value on many consecutive trading days) #####")
flat_rows = []
for (city, crop), g in ds[ds["is_filled"] == 0].groupby(G):
    run = (g[P] != g[P].shift()).cumsum()
    sizes = g.groupby(run).size()
    flat_rows.append((city, crop, int(sizes.max()), int((sizes >= 10).sum())))
flat = pd.DataFrame(flat_rows, columns=["city", "crop", "longest_run", "runs>=10"]).sort_values("longest_run", ascending=False)
say(flat.head(6).to_string(index=False))
status("stuck-price runs >= 10 days", int(flat["runs>=10"].sum()))
if "is_stale" in ds.columns:
    say(f"rows flagged is_stale=1: {ds['is_stale'].mean() * 100:.1f}% overall; by crop (%):")
    say(ds.groupby("crop")["is_stale"].mean().mul(100).round(1).to_string())

# ------------------------------------------------------------------ F. cross-city consistency
say("\n##### F. CROSS-CITY CONSISTENCY (price / median of cities on the same day) #####")
pv = ds.pivot_table(index=["crop", "date"], columns="city", values=P)
med = pv.median(axis=1)
ratio = pv.div(med, axis=0)[pv.notna().sum(axis=1) >= 4]
flag = ratio.stack()
flag = flag[(flag > 1.8) | (flag < 0.55)]
say(f"city prices > 1.8x or < 0.55x the same-day median of the other cities: {len(flag)} of {int(ratio.notna().sum().sum())}")
if len(flag):
    say(flag.rename("ratio").reset_index().groupby(["crop", "city"]).size().unstack(fill_value=0).to_string())
    fl = flag.rename("ratio").reset_index().sort_values(["crop", "city", "date"])
    fl["ep"] = ((fl["date"].diff().dt.days != 1) | (fl["city"] != fl["city"].shift()) | (fl["crop"] != fl["crop"].shift())).cumsum()
    eps = fl.groupby("ep").agg(crop=("crop", "first"), city=("city", "first"), start=("date", "min"),
                               end=("date", "max"), days=("date", "size"), ratio=("ratio", "max"))
    say(f"as episodes (runs of consecutive days): {len(eps)}; 12 longest:")
    say(eps.sort_values("days", ascending=False).head(12).pipe(rnd).to_string(index=False))
status("cross-city price outliers", len(flag))
say("median price ratio of each city to the all-city median (1.00 = typical level):")
say((pv.div(med, axis=0)).groupby("crop").median().round(2).to_string())

# ------------------------------------------------------------------ G. level jumps
say("\n##### G. MONTHLY LEVEL JUMPS (month median vs previous month, > 50%) #####")
mm = ds.assign(ym=ds["date"].dt.to_period("M")).groupby(G + ["ym"])[P].median().reset_index()
mm["mom"] = mm.groupby(G)[P].pct_change() * 100
jumps = mm[mm["mom"].abs() > 50]
say(f"months with |median change| > 50%: {len(jumps)}")
if len(jumps):
    say(jumps.reindex(jumps["mom"].abs().sort_values(ascending=False).index).head(10).pipe(rnd).to_string(index=False))
status("monthly median jump > 50%", len(jumps))

# ------------------------------------------------------------------ H. coverage / gaps
say("\n##### H. COVERAGE (days still missing after cleaning) #####")
rows = []
for (city, crop), g in ds.groupby(G):
    d = g["date"]
    span = (d.max() - d.min()).days + 1
    gap = d.diff().dt.days.max()
    rows.append((crop, city, len(g), span - len(g), int(gap) - 1 if gap == gap else 0, d.max().date()))
cov = pd.DataFrame(rows, columns=["crop", "city", "rows", "missing_days", "longest_gap", "last_date"])
say(cov.sort_values(["crop", "city"]).to_string(index=False))
status("series with > 10% missing days", int(((cov["missing_days"] / (cov["rows"] + cov["missing_days"])) > 0.10).sum()),
       "Potato in some cities is expected")

# ------------------------------------------------------------------ I. weather
say("\n##### I. WEATHER SANITY #####")
wbad = int(((ds["temp_min_c"] > ds["temp_mean_c"] + 0.01) | (ds["temp_mean_c"] > ds["temp_max_c"] + 0.01)).sum())
say(f"rows where min/mean/max temperature are inconsistent: {wbad} | negative rain: {int((ds['rain_mm'] < 0).sum())} "
    f"| max daily rain: {ds['rain_mm'].max():.0f} mm | temp range: {ds['temp_mean_c'].min():.1f} to {ds['temp_mean_c'].max():.1f} C")
status("weather values consistent", wbad + int((ds["rain_mm"] < 0).sum()) + int((ds["temp_mean_c"].abs() > 55).sum()))
mon = ds.assign(m=ds["date"].dt.month)
say("mean temperature by month (C):")
say(mon.pivot_table(index="city", columns="m", values="temp_mean_c", aggfunc="mean").round(0).astype(int).to_string())
say("average rain by month (mm per month):")
w = pd.read_csv(RAW_WEATHER_CSV, parse_dates=["date"])
tot = w.assign(m=w["date"].dt.month, y=w["date"].dt.year).groupby(["city", "y", "m"])["precipitation_sum"].sum().reset_index()
say(tot.pivot_table(index="city", columns="m", values="precipitation_sum", aggfunc="mean").round(0).astype(int).to_string())

# ------------------------------------------------------------------ J. target readiness
say("\n##### J. TARGET READINESS (7-day-ahead price and spike class balance) #####")
avail = ds.merge(wk[G + ["date"]].assign(has7=1), on=G + ["date"], how="left")
say("share of rows that have a price exactly 7 days later (%):")
say((avail.groupby("crop")["has7"].apply(lambda s: s.notna().mean() * 100)).round(1).to_string())
cut = wk["date"].quantile(0.70)
thr = wk[wk["date"] <= cut].groupby("crop")["chg7"].agg(["mean", "std"])
thr["thr_%"] = thr["mean"] + 1.5 * thr["std"]
wk = wk.merge(thr[["thr_%"]], left_on="crop", right_index=True)
wk["spike"] = (wk["chg7"] > wk["thr_%"]).astype(int)
wk["part"] = np.where(wk["date"] <= cut, "train70", "later30")
say(f"train/later split date: {cut.date()}  (threshold = train mean + 1.5 x std)")
say(thr.join(wk.pivot_table(index="crop", columns="part", values="spike", aggfunc="mean").mul(100)).round(2).to_string())
say("spike rate by city (%, whole period):")
say(wk.pivot_table(index="crop", columns="city", values="spike", aggfunc="mean").mul(100).round(1).to_string())

# ------------------------------------------------------------------ summary
say("\n##### SUMMARY #####")
for name, st, note in STATUS:
    say(f"[{st:>10}] {name} {('- ' + note) if note and st != 'OK' else ''}")
(PROCESSED_DIR / "validation_report.txt").write_text("\n".join(LINES), encoding="utf-8")
print("\nSaved:", PROCESSED_DIR / "validation_report.txt")
