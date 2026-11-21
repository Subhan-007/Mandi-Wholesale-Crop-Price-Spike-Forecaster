"""
01b - Test the AMIS 'Price Trends' report form (Crop + City + Start/End date) and look at HistoricalPrices.aspx.

Run (Colab):  !python scripts/01b_probe_price_trends.py
Then paste the printed output (it is short) back to Claude.
"""
import re

import requests
from bs4 import BeautifulSoup

from config import *  # noqa

BASE = "http://www.amis.pk/"
TRENDS = BASE + "Price%20Trends.aspx"
HIST = BASE + "HistoricalPrices.aspx"
PREFIX = "ReportViewer1$ctl08$"
CROP_FIELD = PREFIX + "ctl03$ddValue"
CITY_FIELD = PREFIX + "ctl05$ddValue"
START_FIELD = PREFIX + "ctl07$txtValue"
END_FIELD = PREFIX + "ctl09$txtValue"
SUBMIT_FIELD = PREFIX + "ctl00"


def clean(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def short(s, n=70):
    s = clean(s)
    return s if len(s) <= n else s[:n] + "..."


def hidden_fields(soup):
    return {i["name"]: i.get("value", "") for i in soup.find_all("input", type="hidden") if i.get("name")}


def options(soup, name):
    sel = soup.find("select", attrs={"name": name})
    return {clean(o.get_text()): o.get("value") for o in sel.find_all("option")} if sel else {}


def numeric_table_rows(soup):
    """Rows (anywhere) that contain >=2 numeric cells and at most 12 cells -> likely real data rows."""
    rows = []
    for tr in soup.find_all("tr"):
        cells = [clean(c.get_text()) for c in tr.find_all(["td", "th"])]
        nums = [c for c in cells if re.fullmatch(r"[\d,]+(\.\d+)?", c)]
        if 2 <= len(nums) and len(cells) <= 12 and all(len(c) < 40 for c in cells):
            rows.append(cells)
    return rows


def main():
    s = requests.Session()
    s.headers.update(HEADERS)

    # ---------- Price Trends form
    print("##### Price Trends.aspx #####")
    r = s.get(TRENDS, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    crops = options(soup, CROP_FIELD)
    cities = options(soup, CITY_FIELD)
    print("crop options:", len(crops), "| city options:", len(cities))
    print("Matching our crops:")
    for crop, keys in CROPS.items():
        hits = {n: v for n, v in crops.items() if any(k in n.lower() for k in keys)}
        print(f"  {crop}: {hits}")
    print("Matching our cities:", {c: cities.get(c) for c in CITIES})

    onion = crops.get("Onion", "22")
    lahore = cities.get("Lahore", "1")
    for fmt_name, start, end in [("M/D/YYYY", "1/1/2024", "1/31/2024"), ("DD/MM/YYYY", "01/01/2024", "31/01/2024")]:
        data = hidden_fields(soup)
        data.update({CROP_FIELD: onion, CITY_FIELD: lahore, START_FIELD: start, END_FIELD: end, SUBMIT_FIELD: "View Report"})
        rr = s.post(TRENDS, data=data, timeout=60)
        out = PROBE_DIR / f"price_trends_response_{fmt_name.replace('/', '')}.html"
        out.write_text(rr.text, encoding="utf-8")
        s2 = BeautifulSoup(rr.text, "html.parser")
        rows = numeric_table_rows(s2)
        print(f"\nPOST test, date format {fmt_name}: HTTP {rr.status_code}, {len(rr.text)} bytes, saved {out.name}")
        print("  mentions 'Reserved.ReportViewerWebControl.axd' (AJAX-loaded report):", "Reserved.ReportViewerWebControl" in rr.text)
        print("  numeric data rows found directly in HTML:", len(rows))
        for row in rows[:6]:
            print("   ", [short(c, 25) for c in row])
        err = s2.find(string=re.compile(r"error|invalid|exception", re.I))
        if err:
            print("  possible error text:", short(str(err), 120))

    # ---------- HistoricalPrices.aspx
    print("\n##### HistoricalPrices.aspx #####")
    r = s.get(HIST, timeout=30)
    (PROBE_DIR / "historical_prices.html").write_text(r.text, encoding="utf-8")
    h = BeautifulSoup(r.text, "html.parser")
    print("title:", short(h.title.get_text() if h.title else ""))
    for sel in h.find_all("select"):
        opts = [(o.get("value"), short(o.get_text(), 30)) for o in sel.find_all("option")]
        print(f"  <select name={sel.get('name')}> {len(opts)} options, first 8: {opts[:8]}")
    for inp in h.find_all("input"):
        if inp.get("type") not in ("hidden", "image"):
            print(f"  <input name={inp.get('name')} type={inp.get('type')} value={short(str(inp.get('value')), 30)}>")
    for a in h.find_all("a", href=True)[:15]:
        print("  link:", short(a.get_text(), 40), "->", a["href"][:80])
    rows = numeric_table_rows(h)
    print("  numeric data rows:", len(rows), [short(c, 20) for c in rows[0]] if rows else "")


if __name__ == "__main__":
    main()
