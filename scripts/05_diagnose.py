"""
05 - Diagnostics on the raw AMIS data (read-only). Run:  python scripts\\05_diagnose.py
Prints: units/blank prices per series, price ranges, longest gaps, and rows per month for seasonal series.
"""
import pandas as pd

from config import *  # noqa

pd.set_option("display.width", 200)
raw = pd.read_csv(RAW_PRICES_CSV, parse_dates=["date"])

print("##### 1. Per AMIS series #####")
print(raw.groupby(["crop_variant", "amis_crop_id"]).agg(
    rows=("price", "size"),
    blank_price=("price", lambda s: int(s.isna().sum())),
    blank_unit=("unit_raw", lambda s: int(s.isna().sum())),
    units=("unit_raw", lambda s: ",".join(sorted(set(s.dropna())))),
    min=("price", "min"), median=("price", "median"), max=("price", "max")).to_string())

print("\n##### 2. Header of an export from each series with a blank unit (Lahore, first chunk) #####")
for cid in raw[raw["unit_raw"].isna()]["amis_crop_id"].unique():
    files = sorted((RAW_DIR / "exports").glob(f"{cid}_Lahore_*.csv"))
    if files:
        print(f"-- crop id {cid}: {files[0].name}")
        for line in files[0].read_text(encoding="utf-8").splitlines()[:4]:
            print("   ", line[:120])

print("\n##### 3. Rows per calendar month (all cities, 2022-2026) for the series with two variants #####")
sel = raw[raw["crop_variant"].isin(["Potato Fresh", "Garlic (China)", "Garlic (Local)"]) & raw["price"].notna()]
print(sel.groupby([sel["date"].dt.month.rename("month"), "crop_variant"]).size().unstack(fill_value=0).to_string())

print("\n##### 4. Ten longest gaps between consecutive priced days (> 2 days missing) #####")
gaps = []
for (city, var), g in raw.dropna(subset=["price"]).groupby(["city", "crop_variant"]):
    d = g["date"].sort_values().reset_index(drop=True)
    miss = d.diff().dt.days - 1
    for i in miss[miss > 2].index:
        gaps.append((int(miss[i]), city, var, d[i - 1].date(), d[i].date()))
gaps.sort(reverse=True)
print(f"total gaps > 2 days: {len(gaps)}")
for g in gaps[:10]:
    print("  ", g)
print("\nGaps > 2 days per series:")
print(pd.DataFrame(gaps, columns=["days", "city", "variant", "from", "to"]).groupby("variant")["days"]
      .agg(["count", "sum", "max"]).to_string())
