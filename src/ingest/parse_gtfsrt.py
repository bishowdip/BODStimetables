"""Parse archived GTFS-RT protobuf snapshots, filter to West Yorkshire, write Parquet.

The national feed is large, so each snapshot file is parsed and discarded in turn
(parse-and-stream) rather than held in memory all at once -- this is the part the
report flags as *why* we cannot just load everything into pandas. Only the small,
bbox-filtered result is written out, partitioned by service_date so Spark can
prune by date later.

Input layout (produced by download_archive.py):
    data/raw/<service_date>/*.pb | *.dat   (one or many FeedMessage files)
Output:
    data/parquet/positions/service_date=<date>/part-*.parquet
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.transit import gtfs_realtime_pb2

from src.common import load_config, project_path

log = logging.getLogger("parse_gtfsrt")

SNAPSHOT_GLOBS = ("*.pb", "*.dat", "*.bin")


def _in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    return (
        bbox["min_lat"] <= lat <= bbox["max_lat"]
        and bbox["min_lon"] <= lon <= bbox["max_lon"]
    )


def _iter_positions(path: Path, bbox: dict):
    """Yield bbox-filtered vehicle-position rows from one FeedMessage file."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(path.read_bytes())
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        pos = v.position
        if not _in_bbox(pos.latitude, pos.longitude, bbox):
            continue
        ts = v.timestamp or feed.header.timestamp
        yield {
            "vehicle_id": v.vehicle.id or entity.id,
            "trip_id": v.trip.trip_id or None,
            "route_id": v.trip.route_id or None,
            "lat": float(pos.latitude),
            "lon": float(pos.longitude),
            "bearing": float(pos.bearing) if pos.HasField("bearing") else None,
            "ts": int(ts),
            "ts_utc": datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None,
        }


def parse_date(date: str, raw_root: Path, out_root: Path, bbox: dict) -> int:
    src_dir = raw_root / date
    if not src_dir.exists():
        log.warning("no raw dir for %s -- skipping", date)
        return 0

    files: list[Path] = []
    for pattern in SNAPSHOT_GLOBS:
        files.extend(sorted(src_dir.glob(pattern)))
    if not files:
        log.warning("no snapshot files under %s", src_dir)
        return 0

    rows: list[dict] = []
    for fp in files:
        try:
            rows.extend(_iter_positions(fp, bbox))
        except Exception as exc:  # a corrupt snapshot shouldn't kill the run
            log.warning("could not parse %s: %s", fp.name, exc)

    if not rows:
        log.info("%s: no in-bbox positions", date)
        return 0

    df = pd.DataFrame(rows)
    df["service_date"] = date
    out_dir = out_root / "positions" / f"service_date={date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns="service_date").to_parquet(out_dir / "part-0.parquet", index=False)
    log.info("%s: wrote %d positions from %d files", date, len(df), len(files))
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="*")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    dates = args.dates or cfg["window"]["dates"]
    bbox = cfg["region"]["bbox"]
    raw_root = project_path(cfg["paths"]["raw"])
    out_root = project_path(cfg["paths"]["parquet"])

    total = sum(parse_date(d, raw_root, out_root, bbox) for d in dates)
    log.info("total WY positions written: %d", total)


if __name__ == "__main__":
    main()
