import pandas as pd
d = pd.read_csv("data/raw/amis_prices_raw.csv")
g = d[(d.city == "Faisalabad") & (d.crop == "Garlic") & d.date.between("2025-12-20", "2026-02-15")]
print(g.pivot_table(index="date", columns="crop_variant", values="price").to_string())