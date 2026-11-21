import pandas as pd
o = pd.read_csv("data/processed/outliers_removed.csv", parse_dates=["date"])
c = o[o.reason == "cross_city"].sort_values(["crop", "city", "date"])
c["gap"] = c.groupby(["crop", "city"]).date.diff().dt.days.fillna(99)
c["ep"] = ((c.gap > 3)).cumsum()
ep = c.groupby("ep").agg(crop=("crop", "first"), city=("city", "first"),
                         start=("date", "min"), end=("date", "max"),
                         days=("date", "size"), mean_price=("price", "mean"))
print(ep.sort_values("days", ascending=False).head(25).to_string())
print("\nRemoved inside shock windows:")
for a, b in [("2026-06-01", "2026-07-31"), ("2022-08-15", "2022-09-05"), ("2023-08-25", "2023-09-10")]:
    m = c[c.date.between(a, b)]
    print(a, b, m.groupby(["crop", "city"]).size().to_dict())