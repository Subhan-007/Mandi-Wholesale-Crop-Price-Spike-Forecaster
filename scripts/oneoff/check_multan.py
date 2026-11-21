import pandas as pd
d = pd.read_csv("data/processed/mandi_dataset.csv", parse_dates=["date"])
g = d[(d.crop == "Garlic") & (d.city == "Multan") & d.date.between("2024-12-05", "2025-03-09")]
print(g.groupby(["price_source", "is_adjusted", "is_filled"]).size())
print(g.price_pkr_per_100kg.describe())
print(d[(d.crop == "Garlic") & (d.date == "2025-01-15")][["city", "price_source", "price_pkr_per_100kg", "is_adjusted"]])
o = pd.read_csv("data/processed/outliers_removed.csv", parse_dates=["date"])
print(o[(o.city == "Multan") & (o.crop == "Garlic")].groupby("reason").size())