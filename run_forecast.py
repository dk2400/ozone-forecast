#!/usr/bin/env python3
"""
Automated Daily Ozone Forecast Pipeline
========================================
Runs at 2 AM daily via cron. Correct data sequence per site:

  Day -2 (day before yesterday) ozone  →  Lag Ozone for Day -1 rows
  Day -1 (yesterday) ozone             →  Ozone for Day -1 rows
                                           AND Lag Ozone for Day 0 rows
  Day -1 (yesterday) met               →  observed met for Day -1 rows
  Day  0 (today) met                   →  forecast met for Day 0 rows
  48-row input (Day -1 + Day 0) fed into sliding-window RF model
  → predicts today's hourly ozone → peak ppb → AQI category
  → updates index.html → pushes to GitHub Pages

Setup
-----
1. Install dependencies:
       pip install requests pandas openpyxl joblib scikit-learn numpy PyGithub

2. Place your four trained .joblib model files in the same folder as this script:
       Site43_RF_LagO3_Overlapping.joblib        (McMillan)
       Site3001_RF_LagO3_Overlapping.joblib      (Rockville)
       Site30_RF_LagO3_Overlapping.joblib        (Beltsville)
       Site8003_RF_LagO3_Overlapping.joblib      (PG Equestrian Center)

3. Fill in YOUR values in the CONFIG section below.

4. Schedule with cron (runs at 2 AM every day):
       crontab -e
       0 2 * * * /usr/bin/python3 /path/to/run_forecast.py >> /path/to/forecast.log 2>&1
"""

# ============================================================
#  CONFIG — fill these in once
# ============================================================
import os
AIRNOW_API_KEY   = os.environ.get("AIRNOW_API_KEY", "5313DE41-540E-4C8F-8F68-196E4E0303FB")
GITHUB_TOKEN     = os.environ.get("GITHUB_TOKEN", "YOUR_NEW_GITHUB_TOKEN")  # ← paste your new token here for local testing only
GITHUB_REPO      = "dk2400/aqforecast"
INDEX_HTML_PATH  = "index.html"
MODEL_DIR        = os.path.dirname(os.path.abspath(__file__))  # same folder as this script
# ============================================================

import sys
import io
import urllib.request
import logging
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import joblib
from github import Github, Auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
#  Site definitions
# ============================================================
SITES = [
    {
        "name":        "McMillan",
        "display":     "McMillan",
        "state_code":  11,
        "county_code": 1,
        "site_num":    43,
        "lat":         38.921848,
        "lon":        -77.013176,
        "model_file":  "Site43_2017-22_RF_LagO3_Overlapping.joblib",
    },
    {
        "name":        "Rockville",
        "display":     "Rockville",
        "state_code":  24,
        "county_code": 31,
        "site_num":    3001,
        "lat":         39.114399,
        "lon":        -77.106903,
        "model_file":  "Site3001_2017-22_RF_LagO3_Overlapping.joblib",
    },
    {
        "name":        "Beltsville",
        "display":     "Beltsville",
        "state_code":  24,
        "county_code": 33,
        "site_num":    30,
        "lat":         39.055302,
        "lon":        -76.878304,
        "model_file":  "Site30_2017-22_RF_LagO3_Overlapping.joblib",
    },
    {
        "name":        "Prince_Georges_Equestrian_Center",
        "display":     "Prince George's Equestrian Center",
        "state_code":  24,
        "county_code": 33,
        "site_num":    8003,
        "lat":         38.812069,
        "lon":        -76.744186,
        "model_file":  "Site8003_2017-22_RF_LagO3_Overlapping.joblib",
    },
]

# AirNow bounding box covering all four sites
BBOX = "-77.106903,38.812069,-76.744186,39.114399"

# Features the model was trained on (must match training order)
NUMERIC_COLS = [
    "Wind Direction - Resultant",
    "Wind Speed - Resultant",
    "Temp",
    "Press",
    "RH",
    "Lag Ozone",
    "Year", "Month", "Day", "Hour",
]

WINDOW_SIZE = 24  # sliding window size used during training

# ============================================================
#  AQI breakpoints for ozone (ppb)
# ============================================================
def ppb_to_aqi_category(peak_ppb: float) -> str:
    """Map peak predicted ozone (ppb) to AQI category keyword used in index.html."""
    if peak_ppb < 55:
        return "good"
    elif peak_ppb < 71:
        return "moderate"
    elif peak_ppb < 86:
        return "usg"
    elif peak_ppb < 106:
        return "unhealthy"
    elif peak_ppb < 200:
        return "very"
    else:
        return "hazardous"

# ============================================================
#  Step 1 — Fetch ozone from AirNow
# ============================================================
def fetch_airnow_ozone(day: date) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [site_num, hour, ozone_ppm]
    for all 24 hours of `day` for the four sites.
    Uses nearest-site matching instead of fixed tolerance so Beltsville
    and PG Equestrian Center (same county, different coords) are never confused.
    """
    log.info(f"Fetching AirNow ozone for {day} …")
    url = (
        f"https://airnowapi.org/aq/data/"
        f"?startdate={day}t0"
        f"&enddate={day}t23"
        f"&parameters=o3"
        f"&bbox={BBOX}"
        f"&datatype=c"
        f"&format=text/csv"
        f"&api_key={AIRNOW_API_KEY}"
    )
    with urllib.request.urlopen(url) as resp:
        csv_data = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(csv_data), header=None)
    df.columns = ["Latitude", "Longitude", "DateTime", "Parameter", "Value", "Unit"]

    # Match each returned row to the NEAREST known site (avoids county-code confusion)
    def nearest_site_num(lat, lon):
        best_snum, best_dist = None, float("inf")
        for s in SITES:
            dist = (lat - s["lat"]) ** 2 + (lon - s["lon"]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_snum = s["site_num"]
        # Only accept if within ~0.05° (~5 km) — rejects unrelated monitors in bbox
        return best_snum if best_dist < 0.05 ** 2 else None

    df["site_num"] = df.apply(lambda r: nearest_site_num(r["Latitude"], r["Longitude"]), axis=1)
    df = df[df["site_num"].notna()].copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["ozone_ppm"] = df["Value"] / 1000.0   # AirNow returns ppb; model trained on ppm
    log.info(f"  Got ozone rows per site: { df.groupby('site_num')['hour'].count().to_dict() }")
    return df[["site_num", "hour", "ozone_ppm"]]


# ============================================================
#  Step 2 — Fetch meteorology from Open-Meteo
# ============================================================
OPEN_METEO_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
]

def fetch_met(site: dict, day: date, is_forecast: bool = False) -> pd.DataFrame:
    """
    Returns a 24-row DataFrame with met columns for `day`.
    For observed days (is_forecast=False): tries forecast endpoint first
    (works up to ~2 days back), then falls back to the archive endpoint.
    For forecast days (is_forecast=True): uses forecast endpoint only.
    """
    label = "forecast" if is_forecast else "observed"
    log.info(f"  Fetching {label} met for {site['name']} on {day} …")

    base_params = {
        "latitude":         site["lat"],
        "longitude":        site["lon"],
        "hourly":           ",".join(OPEN_METEO_VARS),
        "start_date":       day.isoformat(),
        "end_date":         day.isoformat(),
        "timezone":         "America/New_York",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "kn",        # knots — matches training data
        # pressure_msl from Open-Meteo is already in hPa (= millibars) — no unit param needed
    }

    endpoints = ["https://api.open-meteo.com/v1/forecast"]
    if not is_forecast:
        # Archive endpoint as fallback for older observed days
        endpoints.append("https://archive-api.open-meteo.com/v1/archive")

    last_err = None
    for endpoint in endpoints:
        try:
            r = requests.get(endpoint, params=base_params, timeout=60)
            r.raise_for_status()
            data = r.json().get("hourly")
            if not data:
                raise RuntimeError("No hourly data in response")
            df = pd.DataFrame({
                "DateTime":                   pd.to_datetime(data["time"]),
                "Wind Direction - Resultant": data["wind_direction_10m"],
                "Wind Speed - Resultant":     data["wind_speed_10m"],
                "Temp":                       data["temperature_2m"],
                "Press":                      data["pressure_msl"],
                "RH":                         data["relative_humidity_2m"],
            })
            df = df[df["DateTime"].dt.date == day].reset_index(drop=True)
            if len(df) == 24:
                return df
            log.warning(f"  {endpoint} returned {len(df)} rows for {day}; trying next …")
        except Exception as e:
            last_err = e
            log.warning(f"  {endpoint} failed: {e}; trying next …")

    raise RuntimeError(
        f"Could not fetch met for {site['name']} on {day} from any endpoint. "
        f"Last error: {last_err}"
    )


# ============================================================
#  Step 3 — Build model input for one site
# ============================================================

# Exact column order the model was trained on (after dropping State/County/Site/Date/Time)
FEATURE_COLS = [
    "Wind Direction - Resultant",
    "Wind Speed - Resultant",
    "Temp",
    "Press",
    "RH",
    "Lag Ozone",
    "Ozone",
]

def build_site_input(site: dict,
                     day_minus_2: date,
                     yesterday: date,
                     today: date,
                     ozone_d2: pd.DataFrame,
                     ozone_d1: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a 48-row DataFrame in exactly the training data format:

      State Code | County Code | Site Num | Date Local | Time Local |
      Wind Direction - Resultant | Wind Speed - Resultant | Temp | Press | RH |
      Lag Ozone | Ozone

    Row layout:
      Rows  0-23  → yesterday (Day -1)
        Ozone     = yesterday actual obs (ppm)   from AirNow Day -1
        Lag Ozone = day-before-yesterday (ppm)   from AirNow Day -2
        Met       = yesterday observed met       from Open-Meteo Day -1

      Rows 24-47  → today (Day 0)
        Ozone     = NaN  (not yet observed)
        Lag Ozone = yesterday actual obs (ppm)   from AirNow Day -1
        Met       = today forecast met           from Open-Meteo Day  0
    """
    snum = site["site_num"]

    def get_oz_series(oz_df: pd.DataFrame, label: str) -> np.ndarray:
        """Extract sorted 24-hour ozone array (ppm) for this site; fill gaps if needed."""
        oz = oz_df[oz_df["site_num"] == snum].copy()
        # Deduplicate — AirNow sometimes returns duplicate rows for the same hour
        oz = oz.groupby("hour")["ozone_ppm"].mean().reset_index()
        oz = oz.sort_values("hour").reset_index(drop=True)
        if len(oz) < 24:
            log.warning(
                f"  Only {len(oz)} ozone obs for {site['name']} on {label}; "
                "filling missing hours with column mean."
            )
            full = pd.DataFrame({"hour": range(24)})
            oz = full.merge(oz, on="hour", how="left")
            oz["ozone_ppm"] = oz["ozone_ppm"].fillna(oz["ozone_ppm"].mean())
        return oz["ozone_ppm"].values  # shape (24,), already in ppm

    oz_d2_vals = get_oz_series(ozone_d2, str(day_minus_2))   # Day -2 → Lag Ozone for Day -1 rows
    oz_d1_vals = get_oz_series(ozone_d1, str(yesterday))     # Day -1 → Ozone for Day -1 rows
                                                              #         → Lag Ozone for Day  0 rows

    # Fetch met for both days
    met_yest  = fetch_met(site, yesterday, is_forecast=False)  # Day -1 observed
    met_today = fetch_met(site, today,     is_forecast=True)   # Day  0 forecast
    met = pd.concat([met_yest, met_today], ignore_index=True)

    # Build Date Local and Time Local in training format: "3/1/2023", "0:00"
    datetimes = list(
        pd.date_range(start=str(yesterday), periods=24, freq="h")
    ) + list(
        pd.date_range(start=str(today), periods=24, freq="h")
    )
    dt_index = pd.DatetimeIndex(datetimes)

    date_local = [f"{d.month}/{d.day}/{d.year}" for d in dt_index]
    time_local = [f"{d.hour}:00"               for d in dt_index]

    # Ozone and Lag Ozone columns
    ozone_vals = np.concatenate([oz_d1_vals, np.full(24, np.nan)])  # today = NaN
    lag_vals   = np.concatenate([oz_d2_vals, oz_d1_vals])

    # Assemble in exact training column order
    df = pd.DataFrame({
        "State Code":                site["state_code"],
        "County Code":               site["county_code"],
        "Site Num":                  site["site_num"],
        "Date Local":                date_local,
        "Time Local":                time_local,
        "Wind Direction - Resultant": met["Wind Direction - Resultant"].values,
        "Wind Speed - Resultant":     met["Wind Speed - Resultant"].values,
        "Temp":                       met["Temp"].values,
        "Press":                      met["Press"].values,
        "RH":                         met["RH"].values,
        "Lag Ozone":                  lag_vals,
        "Ozone":                      ozone_vals,
    })

    log.info(f"  Built 48-row input for {site['name']} — sample row 0: "
             f"LagO3={lag_vals[0]:.4f} O3={oz_d1_vals[0]:.4f} "
             f"Temp={met['Temp'].values[0]:.1f}")

    return df


# ============================================================
#  Step 4 — Sliding window + predict
# ============================================================
def create_sliding_windows(data: pd.DataFrame, features: list,
                            window_size: int = 24) -> np.ndarray:
    X = []
    for i in range(window_size, len(data)):
        window = data[features].iloc[i - window_size:i].values.flatten()
        X.append(window)
    return np.array(X)


def predict_site(site: dict, df: pd.DataFrame) -> float:
    """
    Replicates the predictor script exactly:
      1. Parse DateTime from Date Local + Time Local
      2. Sort by DateTime
      3. Extract Year, Month, Day, Hour
      4. Drop State Code, County Code, Site Num, Date Local, Time Local, DateTime
      5. Create sliding windows on NUMERIC_COLS
      6. Predict → convert ppm → ppb → return today's peak
    """
    model_path = os.path.join(MODEL_DIR, site["model_file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    log.info(f"  Loaded model: {site['model_file']}")

    # Replicate predictor script preprocessing exactly
    data = df.copy()
    data["DateTime"] = pd.to_datetime(
        data["Date Local"].astype(str) + " " + data["Time Local"].astype(str)
    )
    data = data.sort_values("DateTime").reset_index(drop=True)

    data["Year"]  = data["DateTime"].dt.year
    data["Month"] = data["DateTime"].dt.month
    data["Day"]   = data["DateTime"].dt.day
    data["Hour"]  = data["DateTime"].dt.hour

    data = data.drop(columns=[
        "State Code", "County Code", "Site Num",
        "Date Local", "Time Local", "DateTime"
    ])

    # NUMERIC_COLS must match training order exactly
    numeric_cols = [
        "Wind Direction - Resultant", "Wind Speed - Resultant",
        "Temp", "Press", "RH", "Lag Ozone",
        "Year", "Month", "Day", "Hour",
    ]

    # Drop rows with NaN in feature cols (today's Ozone=NaN is fine; it's not a feature)
    data = data.dropna(subset=numeric_cols).reset_index(drop=True)

    if len(data) < WINDOW_SIZE:
        raise RuntimeError(
            f"Not enough rows after NaN drop for {site['name']}: "
            f"need {WINDOW_SIZE}, got {len(data)}"
        )

    X = create_sliding_windows(data, numeric_cols, WINDOW_SIZE)
    log.info(f"  Sliding windows: X.shape={X.shape}")

    y_pred_ppm = pipeline.predict(X)
    y_pred_ppb = y_pred_ppm * 1000.0

    # Last 24 predictions correspond to today's hours (rows 24-47 after windowing)
    today_preds = y_pred_ppb[-24:] if len(y_pred_ppb) >= 24 else y_pred_ppb
    peak_ppb = float(np.max(today_preds))
    log.info(f"  Peak predicted ozone for {site['name']}: {peak_ppb:.1f} ppb  "
             f"(hourly range: {today_preds.min():.1f}–{today_preds.max():.1f} ppb)")
    return peak_ppb


# ============================================================
#  Step 5 — Update index.html
# ============================================================
def update_index_html(html: str, today: date,
                      categories: dict) -> str:
    """
    Replace the DATE and FORECAST lines in the EDIT ONLY section.
    categories = { "McMillan": "moderate", "Rockville": "good", ... }
    """
    import re

    # Update date string
    date_str = f"{today.month}/{today.day}/{today.year}"
    html = re.sub(r'const DATE\s*=\s*"[^"]*"', f'const DATE = "{date_str}"', html)

    # Update each site's category in the FORECAST array
    for site in SITES:
        display = site["display"].replace("'", "\\'")
        cat = categories[site["name"]]
        # Match:  { name: "...", location: "...", category: "old" }
        pattern = (
            r'(\{\s*name:\s*"' + re.escape(site["display"]) +
            r'"[^}]+category:\s*")[^"]*(")'
        )
        replacement = r'\g<1>' + cat + r'\2'
        html = re.sub(pattern, replacement, html)

    return html


# ============================================================
#  Step 6 — Push to GitHub
# ============================================================
def push_to_github(new_html: str):
    log.info("Pushing updated index.html to GitHub …")
    g    = Github(auth=Auth.Token(GITHUB_TOKEN))
    repo = g.get_repo(GITHUB_REPO)
    file = repo.get_contents(INDEX_HTML_PATH)
    repo.update_file(
        path    = INDEX_HTML_PATH,
        message = f"Auto forecast update {date.today().isoformat()}",
        content = new_html,
        sha     = file.sha,
    )
    log.info("GitHub push complete.")


# ============================================================
#  Main
# ============================================================
def main():
    today       = date.today()
    yesterday   = today - timedelta(days=1)
    day_minus_2 = today - timedelta(days=2)

    log.info(f"=== Ozone Forecast Pipeline  |  Forecast date: {today} ===")
    log.info(f"    Day -2 (lag source) : {day_minus_2}")
    log.info(f"    Day -1 (ozone+met)  : {yesterday}")
    log.info(f"    Day  0 (forecast)   : {today}")

    # Step 1a — Fetch Day -2 ozone (used as Lag Ozone for yesterday rows)
    try:
        log.info("Step 1a: Fetching Day -2 ozone from AirNow …")
        ozone_d2 = fetch_airnow_ozone(day_minus_2)
    except Exception as e:
        log.error(f"Failed to fetch Day -2 AirNow ozone: {e}")
        sys.exit(1)

    # Step 1b — Fetch Day -1 ozone (used as Ozone for yesterday rows
    #            AND as Lag Ozone for today rows)
    try:
        log.info("Step 1b: Fetching Day -1 ozone from AirNow …")
        ozone_d1 = fetch_airnow_ozone(yesterday)
    except Exception as e:
        log.error(f"Failed to fetch Day -1 AirNow ozone: {e}")
        sys.exit(1)

    categories = {}

    for site in SITES:
        log.info(f"--- Processing {site['name']} ---")
        try:
            # Step 2+3: fetch met + assemble 48-row input with correct lag
            df = build_site_input(
                site, day_minus_2, yesterday, today, ozone_d2, ozone_d1
            )
            # Step 4: predict
            peak = predict_site(site, df)
            cat  = ppb_to_aqi_category(peak)
            log.info(f"  AQI category → {cat}")
            categories[site["name"]] = cat
        except Exception as e:
            log.error(f"Error processing {site['name']}: {e}")
            log.error(traceback.format_exc())
            categories[site["name"]] = "moderate"   # safe fallback

    log.info(f"Final categories: {categories}")

    # Step 5 — Fetch current index.html from GitHub
    log.info("Fetching index.html from GitHub …")
    try:
        g    = Github(auth=Auth.Token(GITHUB_TOKEN))
        repo = g.get_repo(GITHUB_REPO)
        file = repo.get_contents(INDEX_HTML_PATH)
        html = file.decoded_content.decode("utf-8")
    except Exception as e:
        log.error(f"Failed to fetch index.html from GitHub: {e}")
        sys.exit(1)

    # Step 6 — Update HTML and push
    updated_html = update_index_html(html, today, categories)
    try:
        push_to_github(updated_html)
    except Exception as e:
        log.error(f"Failed to push to GitHub: {e}")
        sys.exit(1)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
