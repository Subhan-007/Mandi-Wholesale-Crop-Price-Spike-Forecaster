"""Shared settings for Mandi Project - Phase 1 (Data Engineering & EDA)."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROBE_DIR = ROOT / "data" / "probe"
PROCESSED_DIR = ROOT / "data" / "processed"
for _d in (RAW_DIR, PROBE_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- scope
# Crop name -> keywords used to match the commodity name printed on AMIS.
# (lower-case; a row matches if ANY keyword is contained in its name)
CROPS = {
    "Onion": ["onion", "piyaz", "pyaz"],
    "Tomato": ["tomato", "tamatar"],
    "Potato": ["potato", "aloo"],
    "Garlic": ["garlic", "lehsan", "lehsun"],
    "Green Chilli": ["green chilli", "green chili", "hari mirch", "chillies green"],
}

# City -> coordinates (used for Open-Meteo) and the name AMIS uses for the market.
CITIES = {
    "Lahore":     {"lat": 31.5497, "lon": 74.3436, "amis_name": "Lahore"},
    "Faisalabad": {"lat": 31.4504, "lon": 73.1350, "amis_name": "Faisalabad"},
    "Rawalpindi": {"lat": 33.5651, "lon": 73.0169, "amis_name": "Rawalpindi"},
    "Multan":     {"lat": 30.1575, "lon": 71.5249, "amis_name": "Multan"},
    "Gujranwala": {"lat": 32.1877, "lon": 74.1945, "amis_name": "Gujranwala"},
}

START_DATE = date(2022, 1, 1)   # ~5 years of history; shrink if AMIS has less
END_DATE = date.today()

# ---------------------------------------------------------------- AMIS
AMIS_BASE_URL = "http://www.amis.pk/"          # note: http, not https
AMIS_TRENDS_URL = AMIS_BASE_URL + "Price%20Trends.aspx"

# IDs taken from the dropdowns on Price Trends.aspx (found by 01b_probe_price_trends.py)
CITY_IDS = {"Lahore": "1", "Faisalabad": "2", "Gujranwala": "3", "Rawalpindi": "6", "Multan": "7"}
CROP_IDS = {   # crop -> [(AMIS name, id), ...]  FIRST variant = primary series, later ones = backups used to fill gaps
    "Onion": [("Onion", "22")],
    "Tomato": [("Tomato", "25")],
    "Potato": [("Potato Fresh", "20"), ("Potato Store", "21")],   # Fresh is seasonal -> Store fills the gaps
    "Garlic": [("Garlic (Local)", "69"), ("Garlic (China)", "23")],
    "Green Chilli": [("Green Chilli", "80")],
}
REQUEST_DELAY_SEC = 1.0          # polite pause between requests
REQUEST_TIMEOUT_SEC = 30
HEADERS = {
    "User-Agent": "Mozilla/5.0 (university data-science project; contact: l252580@lhr.nu.edu.pk)"
}


def build_request(day: date, city: str, session=None):
    """
    >>> EDIT THIS AFTER RUNNING 01_probe_amis.py <<<
    Return (method, url, payload) for ONE (date, city) page on AMIS.

    Example A - plain GET with query string:
        return "GET", AMIS_BASE_URL + "SomeRates.aspx", {"date": day.isoformat(), "market": city}

    Example B - ASP.NET form POST (use scrape helper `aspnet_hidden_fields`):
        see 02_scrape_amis.py -> `fetch_page` (mode "POST").

    The probe script prints the real form field names and dropdown values.
    """
    raise NotImplementedError("Run 01_probe_amis.py first, then edit build_request() in config.py")


# ---------------------------------------------------------------- Open-Meteo
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_DAILY_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
TIMEZONE = "Asia/Karachi"

# ---------------------------------------------------------------- files
RAW_PRICES_CSV = RAW_DIR / "amis_prices_raw.csv"
PROGRESS_CSV = RAW_DIR / "amis_progress.csv"
FAILED_CSV = RAW_DIR / "amis_failed_requests.csv"
RAW_WEATHER_CSV = RAW_DIR / "weather_raw.csv"
FINAL_DATASET_CSV = PROCESSED_DIR / "mandi_dataset.csv"
