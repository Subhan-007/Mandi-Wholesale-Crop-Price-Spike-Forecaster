import re, pathlib
p = pathlib.Path("04_clean_merge.py")
t = p.read_text(encoding="utf-8")
assert "DAY_HIGH" not in t, "already patched"

t = re.sub(r"CROSS_HIGH, CROSS_LOW = .*\n.*OWN_HIGH, OWN_LOW = .*\n",
 "CROSS_HIGH, CROSS_LOW = 1.6, 0.62   # 5-day median of price/median(other cities)\n"
 "DAY_HIGH, DAY_LOW = 1.45, 0.70      # and the day itself must also be outside this\n", t, count=1)

new = '''def cross_city_filter(df):
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


'''
s = t.index("def cross_city_filter"); e = t.index("def _edge_ratio")
t = t[:s] + new + t[e:]
p.write_text(t, encoding="utf-8")
print("patched")