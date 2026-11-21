"""
Subtask 2 - AMIS Data Collection (Requests + BeautifulSoup), using the Price Trends report's CSV export.

    python scripts\\02_scrape_amis.py --test          # Onion/Lahore for the WHOLE date range in one request (checks limits)
    python scripts\\02_scrape_amis.py --chunk-days 365  # full run (resumable - just run it again if it stops)

Per (crop series, city, date chunk):  GET page -> POST form -> read ExportUrlBase -> GET ...Format=CSV -> parse.

Output
    data/raw/amis_prices_raw.csv      date, city, crop, crop_variant, amis_crop_id, price, unit_raw
    data/raw/exports/*.csv            the untouched CSV downloaded for every request (for traceability)
    data/raw/amis_progress.csv        finished requests (used to resume)
    data/raw/amis_failed_requests.csv requests that failed after retries
"""
import argparse
import csv
import json
import re
import time
from datetime import date, datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import *  # noqa

EXPORT_DIR = RAW_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
COLUMNS = ["date", "city", "crop", "crop_variant", "amis_crop_id", "price", "unit_raw"]
P = "ReportViewer1$ctl08$"
F_CROP, F_CITY, F_START, F_END, F_GO = (P + "ctl03$ddValue", P + "ctl05$ddValue",
                                        P + "ctl07$txtValue", P + "ctl09$txtValue", P + "ctl00")


def mdy(d):  # AMIS accepts M/D/YYYY
    return f"{d.month}/{d.day}/{d.year}"


# ------------------------------------------------------------------ parsing
def parse_export_csv(text):
    """Return (unit, [(date, price), ...]) from the SSRS CSV export."""
    unit, rows = "", []
    for r in csv.reader(text.splitlines()):
        if len(r) >= 5 and r[0] and r[0] != "CropName_1" and "/" in r[2] and r[4].lower().startswith("rs"):
            unit = r[4]                                   # e.g. 'Rs./100kg'
        if len(r) == 3 and r[0] == "Price":
            try:
                d = datetime.strptime(r[1].strip(), "%d %b %y").date()
            except ValueError:
                continue
            try:
                price = float(r[2].replace(",", ""))
            except ValueError:
                price = None                              # blank / NaN price
            rows.append((d, price))
    return unit, rows


# ------------------------------------------------------------------ HTTP
def hidden(soup):
    return {i["name"]: i.get("value", "") for i in soup.find_all("input", type="hidden") if i.get("name")}


def fetch_csv(session, crop_id, city_id, start, end):
    """Return CSV text for one crop/city/date range (raises on failure)."""
    page = session.get(AMIS_TRENDS_URL, timeout=REQUEST_TIMEOUT_SEC)
    page.raise_for_status()
    data = hidden(BeautifulSoup(page.text, "html.parser"))
    data.update({F_CROP: crop_id, F_CITY: city_id, F_START: mdy(start), F_END: mdy(end), F_GO: "View Report"})
    post = session.post(AMIS_TRENDS_URL, data=data, timeout=REQUEST_TIMEOUT_SEC * 2)
    post.raise_for_status()
    m = re.search(r'"ExportUrlBase"\s*:\s*"([^"]+)"', post.text)
    if not m:
        raise RuntimeError("ExportUrlBase not found in response (report did not render)")
    url = urljoin(AMIS_BASE_URL, json.loads('"' + m.group(1) + '"')) + "CSV"
    r = session.get(url, timeout=REQUEST_TIMEOUT_SEC * 2)
    r.raise_for_status()
    return r.content.decode("utf-8-sig", errors="replace")


def fetch_with_retries(session, *args, tries=3):
    for i in range(tries):
        try:
            return fetch_csv(session, *args)
        except Exception as e:  # noqa
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


# ------------------------------------------------------------------ storage
def append_rows(path, rows, fields):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def load_done():
    if not PROGRESS_CSV.exists():
        return set()
    with open(PROGRESS_CSV, encoding="utf-8") as f:
        return {(r["crop_id"], r["city"], r["start"]) for r in csv.DictReader(f)}


def chunks(a, b, days):
    s = a
    while s <= b:
        e = min(s + timedelta(days=days - 1), b)
        yield s, e
        s = e + timedelta(days=1)


# ------------------------------------------------------------------ main
def run(start, end, chunk_days, test=False):
    session = requests.Session()
    session.headers.update(HEADERS)
    done = set() if test else load_done()
    jobs = [(crop, name, cid, city, city_id)
            for crop, variants in CROP_IDS.items() for name, cid in variants
            for city, city_id in CITY_IDS.items()]
    if test:
        jobs = [j for j in jobs if j[0] == "Onion" and j[3] == "Lahore"]
    n = len(jobs) * len(list(chunks(start, end, chunk_days)))
    print(f"{len(jobs)} crop/city series x chunks = {n} requests")
    k = 0
    for crop, name, cid, city, city_id in jobs:
        for s, e in chunks(start, end, chunk_days):
            k += 1
            key = (cid, city, s.isoformat())
            if key in done:
                continue
            try:
                text = fetch_with_retries(session, cid, city_id, s, e)
                unit, rows = parse_export_csv(text)
                if test:
                    print(f"{name}/{city} {s}..{e}: {len(rows)} rows; first {rows[:1]} last {rows[-1:]} unit={unit}")
                    continue
                (EXPORT_DIR / f"{cid}_{city}_{s}.csv").write_text(text, encoding="utf-8")
                recs = [{"date": d.isoformat(), "city": city, "crop": crop, "crop_variant": name,
                         "amis_crop_id": cid, "price": p, "unit_raw": unit} for d, p in rows]
                if recs:
                    append_rows(RAW_PRICES_CSV, recs, COLUMNS)
                append_rows(PROGRESS_CSV, [{"crop_id": cid, "city": city, "start": s.isoformat(),
                                            "end": e.isoformat(), "rows": len(recs)}],
                            ["crop_id", "city", "start", "end", "rows"])
                print(f"[{k}/{n}] {name} | {city} | {s}..{e}: {len(recs)} rows")
            except Exception as ex:  # noqa
                print(f"[{k}/{n}] FAILED {name} | {city} | {s}: {ex}")
                append_rows(FAILED_CSV, [{"crop_id": cid, "city": city, "start": s.isoformat(),
                                          "error": str(ex)[:200]}], ["crop_id", "city", "start", "error"])
            time.sleep(REQUEST_DELAY_SEC)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=START_DATE)
    ap.add_argument("--end", type=date.fromisoformat, default=END_DATE)
    ap.add_argument("--chunk-days", type=int, default=365)
    ap.add_argument("--test", action="store_true", help="Onion/Lahore only, whole range in ONE request")
    a = ap.parse_args()
    if a.test:
        run(a.start, a.end, (a.end - a.start).days + 1, test=True)
    else:
        run(a.start, a.end, a.chunk_days)
