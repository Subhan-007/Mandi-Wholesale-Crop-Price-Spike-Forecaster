"""
Subtask 3 - Weather Data Collection (Open-Meteo historical archive API).

    python scripts\\03_collect_weather.py

Output: data/raw/weather_raw.csv
    date, city, precipitation_sum, temperature_2m_max, temperature_2m_min, temperature_2m_mean
(one row per city per day; matched to market dates later, in 04_clean_merge.py)
"""
import time
from datetime import date, timedelta

import pandas as pd
import requests

from config import *  # noqa


def fetch_city(city, info, start, end, tries=4):
    params = {
        "latitude": info["lat"], "longitude": info["lon"],
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(WEATHER_DAILY_VARS), "timezone": TIMEZONE,
    }
    for attempt in range(tries):
        try:
            r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC * 2)
            if r.status_code == 429:                       # rate limited -> wait and retry
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            daily = r.json()["daily"]
            df = pd.DataFrame(daily).rename(columns={"time": "date"})
            df.insert(1, "city", city)
            return df
        except Exception as e:  # noqa
            print(f"  attempt {attempt + 1} failed for {city}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo failed for {city}")


def main():
    end = min(END_DATE, date.today() - timedelta(days=1))   # archive is not available for today
    frames = []
    for city, info in CITIES.items():
        print(f"Fetching {city} {START_DATE} -> {end} ...")
        frames.append(fetch_city(city, info, START_DATE, end))
        time.sleep(2)
    weather = pd.concat(frames, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.drop_duplicates(["city", "date"]).sort_values(["city", "date"])
    weather.to_csv(RAW_WEATHER_CSV, index=False)
    print(f"\nSaved {len(weather)} rows to {RAW_WEATHER_CSV}")
    print(weather.groupby("city").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max"),
                                      missing_rain=("precipitation_sum", lambda s: int(s.isna().sum()))))


if __name__ == "__main__":
    main()
