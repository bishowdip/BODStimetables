"""Fetch hourly historical weather for the study window from Open-Meteo.

Open-Meteo's archive API is free and needs no key (stated in the report as the
reason there are no credentials in the repo). We pull hourly precipitation,
temperature and wind for the centre of the WY bbox -- one point is enough as a
band-level covariate; the delay signal we care about is regional weather, not
street-level.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
import requests

from src.common import load_config, project_path

log = logging.getLogger("fetch_weather")

HOURLY_VARS = ["precipitation", "temperature_2m", "wind_speed_10m"]


def fetch(cfg) -> pd.DataFrame:
    bbox = cfg["region"]["bbox"]
    lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
    dates = sorted(cfg["window"]["dates"])

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": dates[0],
        "end_date": dates[-1],
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/London",
    }
    url = cfg["sources"]["open_meteo_archive"]
    log.info("requesting %s %s..%s", url, dates[0], dates[-1])
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = pd.DataFrame(hourly).rename(
        columns={
            "time": "ts_local",
            "precipitation": "precip_mm",
            "temperature_2m": "temp_c",
            "wind_speed_10m": "wind_kph",
        }
    )
    df["ts_local"] = pd.to_datetime(df["ts_local"])
    df["service_date"] = df["ts_local"].dt.date.astype(str)
    df["hour"] = df["ts_local"].dt.hour
    return df


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()

    df = fetch(cfg)
    out = project_path(cfg["paths"]["parquet"]) / "weather"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "weather.parquet", index=False)
    log.info("wrote %d hourly weather rows", len(df))


if __name__ == "__main__":
    main()
