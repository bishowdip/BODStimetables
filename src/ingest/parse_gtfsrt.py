"""Parse captured GTFS-RT snapshots into per-day position Parquet.

Snapshots arrive one per minute, but vehicles report on their own cadence, so
consecutive snapshots repeat pings. We deduplicate on (vehicle_id, timestamp):
the same report seen twice is one observation. Rows are accumulated as tuples
rather than dicts to keep a full day (~2M pings) cheap in memory, and each day
is written independently so a failed day can be re-run alone.

Input:   data/raw/<date>/*.pb            (one FeedMessage per snapshot)
Output:  data/parquet/positions/service_date=<date>/part-0.parquet
         docs/results/parse_quality.json (per-day quality counters)

The feed is already server-side filtered to the WY bounding box; the bbox check
here is a guard against feed glitches, and anything it drops is counted.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.transit import gtfs_realtime_pb2

from src.common import load_config, project_path

log = logging.getLogger("parse_gtfsrt")

COLUMNS = ["vehicle_id", "trip_id", "route_id", "lat", "lon", "bearing", "ts"]


def _parse_snapshot(path: Path, bbox: dict) -> tuple[list[tuple], int]:
    """One FeedMessage -> (rows, n_dropped_out_of_bbox)."""
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(path.read_bytes())
    rows: list[tuple] = []
    dropped = 0
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        pos = v.position
        if not (
            bbox["min_lat"] <= pos.latitude <= bbox["max_lat"]
            and bbox["min_lon"] <= pos.longitude <= bbox["max_lon"]
        ):
            dropped += 1
            continue
        rows.append(
            (
                v.vehicle.id or entity.id,
                v.trip.trip_id or None,
                v.trip.route_id or None,
                float(pos.latitude),
                float(pos.longitude),
                float(pos.bearing) if pos.HasField("bearing") else None,
                int(v.timestamp or feed.header.timestamp),
            )
        )
    return rows, dropped


def parse_day(date: str, raw_root: Path, out_root: Path, bbox: dict) -> dict | None:
    """Parse every snapshot for one service date; return quality counters."""
    files = sorted((raw_root / date).glob("*.pb"))
    if not files:
        return None

    rows: list[tuple] = []
    corrupt = out_of_bbox = 0
    for fp in files:
        try:
            got, dropped = _parse_snapshot(fp, bbox)
            rows.extend(got)
            out_of_bbox += dropped
        except Exception as exc:  # one corrupt snapshot must not kill the day
            corrupt += 1
            log.warning("unreadable snapshot %s: %s", fp.name, exc)

    raw_n = len(rows)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates(subset=["vehicle_id", "ts"]).sort_values(["vehicle_id", "ts"])
    df["ts_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    out_dir = out_root / "positions" / f"service_date={date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "part-0.parquet", index=False)

    stats = {
        "date": date,
        "snapshots": len(files),
        "corrupt_snapshots": corrupt,
        "raw_pings": raw_n,
        "deduped_pings": len(df),
        "duplicate_share": round(1 - len(df) / raw_n, 4) if raw_n else 0.0,
        "out_of_bbox": out_of_bbox,
        "with_trip_id": int(df["trip_id"].notna().sum()),
        "trip_id_share": round(float(df["trip_id"].notna().mean()), 4) if len(df) else 0.0,
    }
    log.info(
        "%s: %d snapshots -> %s pings (%.1f%% duplicates dropped, %.1f%% with trip_id)",
        date, len(files), f"{len(df):,}",
        100 * stats["duplicate_share"], 100 * stats["trip_id_share"],
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="*", help="override the window dates from config")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    dates = args.dates or cfg["window"]["dates"]
    raw_root = project_path(cfg["paths"]["raw"])
    out_root = project_path(cfg["paths"]["parquet"])
    bbox = cfg["region"]["bbox"]

    all_stats = [s for d in dates if (s := parse_day(d, raw_root, out_root, bbox))]
    total = sum(s["deduped_pings"] for s in all_stats)

    results = project_path("docs", "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "parse_quality.json").write_text(json.dumps(all_stats, indent=2))
    log.info("done: %s positions across %d days", f"{total:,}", len(all_stats))


if __name__ == "__main__":
    main()
