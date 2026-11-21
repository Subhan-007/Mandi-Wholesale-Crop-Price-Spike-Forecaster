"""
01c - Try to get Price Trends data through the report's built-in export (CSV/Excel),
and probe Reports/YearMonthWiseDrillDown.aspx (May-2007 onwards).

Run:  python scripts/01c_try_ssrs_export.py
Paste the (short) output back to Claude.
"""
import json
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import *  # noqa

BASE = "http://www.amis.pk/"
TRENDS = BASE + "Price%20Trends.aspx"
P = "ReportViewer1$ctl08$"
CROP, CITY, START, END, GO = P + "ctl03$ddValue", P + "ctl05$ddValue", P + "ctl07$txtValue", P + "ctl09$txtValue", P + "ctl00"


def clean(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def hidden(soup):
    return {i["name"]: i.get("value", "") for i in soup.find_all("input", type="hidden") if i.get("name")}


s = requests.Session()
s.headers.update(HEADERS)

print("##### 1. POST Price Trends (Onion, Lahore, 1/1/2024 - 1/31/2024) #####")
soup = BeautifulSoup(s.get(TRENDS, timeout=30).text, "html.parser")
data = hidden(soup)
data.update({CROP: "22", CITY: "1", START: "1/1/2024", END: "1/31/2024", GO: "View Report"})
html = s.post(TRENDS, data=data, timeout=60).text
(PROBE_DIR / "trends_post.html").write_text(html, encoding="utf-8")

print("Reserved.ReportViewerWebControl.axd URLs in page (first 3, cut to 220 chars):")
for m in list(re.finditer(r"Reserved\.ReportViewerWebControl\.axd[^\"'<> ]{0,400}", html))[:3]:
    print("   ", m.group()[:220])
for key in ["ExportUrlBase", "ReportSession", "ControlID", "AsyncRendering", "ReportStack"]:
    m = re.search(rf'"{key}"\s*:\s*("([^"]*)"|true|false|\d+)', html)
    print(f"  {key}:", (m.group(1)[:160] if m else "NOT FOUND"))

m = re.search(r'"ExportUrlBase"\s*:\s*"([^"]+)"', html)
if m:
    base = json.loads('"' + m.group(1) + '"')
    print("\n##### 2. Trying export formats #####")
    for fmt in ["CSV", "EXCELOPENXML", "XML"]:
        url = urljoin(BASE, base) + fmt
        try:
            r = s.get(url, timeout=60)
            ctype = r.headers.get("Content-Type", "")
            print(f"  {fmt}: HTTP {r.status_code}, {len(r.content)} bytes, type={ctype}")
            if r.ok and fmt == "CSV":
                (PROBE_DIR / "export_test.csv").write_bytes(r.content)
                print("  --- first 600 chars ---")
                print(r.content.decode("utf-8-sig", errors="replace")[:600])
        except Exception as e:  # noqa
            print(f"  {fmt}: failed {e}")
else:
    print("\nNo ExportUrlBase in the POST response -> export link not available this way.")

print("\n##### 3. YearMonthWiseDrillDown.aspx #####")
r = s.get(BASE + "Reports/YearMonthWiseDrillDown.aspx", timeout=30)
d = BeautifulSoup(r.text, "html.parser")
(PROBE_DIR / "yearmonth.html").write_text(r.text, encoding="utf-8")
print("HTTP", r.status_code, "| title:", clean(d.title.get_text()) if d.title else None)
for sel in d.find_all("select"):
    opts = [(o.get("value"), clean(o.get_text())[:25]) for o in sel.find_all("option")]
    print(f"  <select {sel.get('name')}> {len(opts)} options; first 6: {opts[:6]}")
for inp in d.find_all("input"):
    if inp.get("type") in ("text", "submit"):
        print(f"  <input {inp.get('name')} type={inp.get('type')}>")
print("  mentions Reserved.ReportViewerWebControl.axd:", "Reserved.ReportViewerWebControl" in r.text)
