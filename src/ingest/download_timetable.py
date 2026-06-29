"""Download the current BODS regional GTFS timetable (the scheduled truth).

Timetables are published as a single current GTFS zip per region, not per date,
so one download covers the whole study window (the calendar.txt inside maps
trips to service days). Saved to data/raw/timetable.zip, which load_static.py
picks up.

    export BODS_API_KEY=...      # only if your account/region requires it
    python -m src.ingest.download_timetable
"""
from __future__ import annotations

import argparse
import logging
import os

import requests

from src.common import load_config, project_path

log = logging.getLogger("download_timetable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="override the timetable URL from config")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    bods = cfg["sources"]["bods"]
    url = args.url or bods["timetable_url"]
    params = {}
    key = os.environ.get(bods["api_key_env"])
    if key:
        params["api_key"] = key

    dest = project_path(cfg["paths"]["raw"]) / "timetable.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading timetable from %s", url)

    session = requests.Session()
    session.headers.update({"User-Agent": "st5011cem-research-project"})
    with session.get(url, params=params, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(1 << 16):
                fh.write(chunk)
        tmp.rename(dest)
    log.info("saved %s (%d bytes)", dest, dest.stat().st_size)


if __name__ == "__main__":
    main()
