"""
Subtask 1 - Data Source Investigation.

Run this FIRST, on your own computer (it needs internet):
    python scripts/01_probe_amis.py

It will
  1. download the AMIS home page and save the raw HTML in data/probe/
  2. print every form, dropdown (<select>) with its option values, input field and table it finds
  3. follow links that look like "rates / prices / daily" and do the same
  4. test whether Open-Meteo's historical API works for your cities and date range

Copy the printed report (or data/probe/probe_report.txt) - you need it to fill in
build_request() in config.py, and you can paste it to Claude to get the scraper tuned.
"""
import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import *  # noqa

REPORT = []


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    REPORT.append(line)


def get(url, **kw):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC, **kw)
        r.raise_for_status()
        return r
    except Exception as e:  # noqa
        say(f"   !! failed to fetch {url}: {e}")
        return None


def describe_page(url, html, tag):
    soup = BeautifulSoup(html, "html.parser")
    (PROBE_DIR / f"{tag}.html").write_text(html, encoding="utf-8")
    say(f"\n=== {url}  (saved as data/probe/{tag}.html)")
    say("title:", soup.title.get_text(strip=True) if soup.title else None)

    for i, form in enumerate(soup.find_all("form")):
        say(f"-- form #{i}: method={form.get('method')} action={form.get('action')}")
    for sel in soup.find_all("select"):
        opts = [(o.get("value"), o.get_text(strip=True)) for o in sel.find_all("option")]
        say(f"-- <select name={sel.get('name')} id={sel.get('id')}> {len(opts)} options; first 40: {opts[:40]}")
    for inp in soup.find_all("input"):
        if inp.get("type") != "hidden":
            say(f"-- <input name={inp.get('name')} type={inp.get('type')} value={inp.get('value')}>")
    hidden = [i.get("name") for i in soup.find_all("input", type="hidden")]
    if hidden:
        say("-- hidden fields (ASP.NET sites need these on POST):", hidden)
    tables = soup.find_all("table")
    say(f"-- {len(tables)} <table> element(s)")
    for t_i, t in enumerate(tables[:5]):
        rows = t.find_all("tr")
        say(f"   table #{t_i}: {len(rows)} rows")
        for r in rows[:4]:
            say("     ", [c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])])
    return soup


def probe_amis():
    say("##### AMIS probe #####")
    r = get(AMIS_BASE_URL)
    if r is None:
        say("Could not reach AMIS. Check internet / try again later / open the site in a browser.")
        return
    soup = describe_page(AMIS_BASE_URL, r.text, "home")

    keywords = re.compile(r"rate|price|daily|commodit|market|veg|fruit", re.I)
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = urljoin(AMIS_BASE_URL, a["href"])
        if "amis.pk" in href and href not in seen and (keywords.search(text) or keywords.search(href)):
            seen.add(href)
            links.append((text, href))
    say(f"\nCandidate links ({len(links)}):")
    for text, href in links[:40]:
        say("  ", text, "->", href)

    for i, (text, href) in enumerate(links[:14]):
        rr = get(href)
        if rr is not None:
            describe_page(href, rr.text, f"link_{i}")


def probe_open_meteo():
    say("\n##### Open-Meteo probe #####")
    for city, info in CITIES.items():
        params = {
            "latitude": info["lat"], "longitude": info["lon"],
            "start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat(),
            "daily": ",".join(WEATHER_DAILY_VARS), "timezone": TIMEZONE,
        }
        r = get(OPEN_METEO_ARCHIVE_URL, params=params)
        if r is None:
            say(f"  {city}: FAILED")
            continue
        d = r.json().get("daily", {})
        n = len(d.get("time", []))
        missing = sum(v is None for v in d.get("precipitation_sum", []))
        say(f"  {city}: {n} daily rows, {missing} missing rain values, {d['time'][0]} -> {d['time'][-1]}")


if __name__ == "__main__":
    probe_amis()
    probe_open_meteo()
    (PROBE_DIR / "probe_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say("\nReport saved to data/probe/probe_report.txt")
