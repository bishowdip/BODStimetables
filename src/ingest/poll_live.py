"""Self-archive the live BODS GTFS-RT feed (bbox-filtered) at a fixed cadence.

This is the realistic ingestion path: the dated NDL archive only kept ~64 days,
so for a current study window we accumulate our own snapshots. BODS lets us
filter server-side with boundingBox, so we only ever pull West Yorkshire.

Each poll writes one protobuf snapshot to data/raw/<service_date>/<epoch>.pb,
which is exactly what parse_gtfsrt.py already consumes. Run it for the length of
your window (typically in the background):

    export BODS_API_KEY=...                 # free account at data.bus-data.dft.gov.uk
    nohup python -m src.ingest.poll_live --hours 168 > poll.log 2>&1 &   # 7 days

It is interrupt-safe (Ctrl-C stops cleanly) and resumable (just start it again).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import requests

from src.common import load_config, project_path

log = logging.getLogger("poll_live")
_stop = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log.info("stop requested; finishing current cycle")


def _bbox_param(bbox: dict) -> str:
    # BODS expects minLon,minLat,maxLon,maxLat
    return f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"


def poll_once(url: str, params: dict, raw_root: Path, session: requests.Session) -> bool:
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    now = datetime.now()
    out_dir = raw_root / now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{int(now.timestamp())}.pb"
    dest.write_bytes(resp.content)
    log.info("snapshot %s (%d bytes)", dest.name, len(resp.content))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=168.0, help="how long to poll (default 7 days)")
    ap.add_argument("--every", type=int, help="seconds between polls (default from config)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    bods = cfg["sources"]["bods"]
    api_key = os.environ.get(bods["api_key_env"])
    if not api_key:
        raise SystemExit(f"set {bods['api_key_env']} (your free BODS API key) before polling")

    cadence = args.every or cfg["window"]["gtfsrt_sample_seconds"]
    params = {"api_key": api_key, "boundingBox": _bbox_param(cfg["region"]["bbox"])}
    raw_root = project_path(cfg["paths"]["raw"])
    session = requests.Session()
    session.headers.update({"User-Agent": "st5011cem-research-project"})

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    deadline = time.monotonic() + args.hours * 3600
    ok = fail = 0
    log.info("polling %s every %ds for %.1fh (WY bbox only)", bods["gtfsrt_url"], cadence, args.hours)
    while not _stop and time.monotonic() < deadline:
        cycle_start = time.monotonic()
        try:
            poll_once(bods["gtfsrt_url"], params, raw_root, session)
            ok += 1
        except requests.RequestException as exc:
            fail += 1
            log.warning("poll failed: %s", exc)
        # keep the cadence regardless of request time
        sleep_for = cadence - (time.monotonic() - cycle_start)
        if sleep_for > 0 and not _stop:
            time.sleep(sleep_for)

    log.info("done: %d snapshots saved, %d failures", ok, fail)


if __name__ == "__main__":
    main()
