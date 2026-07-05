"""Print measured record counts for every dataset in the pipeline.

The big-data scale claim in the report must be a measured number, not an
estimate, so this script counts what is actually on disk and prints a table
ready to paste. Re-run after each ingestion stage.

    python -m tools.scale_report
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.common import load_config, project_path


def count_parquet(path: Path) -> int | None:
    files = glob.glob(str(path / "**" / "*.parquet"), recursive=True)
    if not files:
        return None
    return sum(len(pd.read_parquet(f)) for f in files)


def count_positions_raw(raw: Path) -> tuple[int, int]:
    """(snapshot files, total vehicle positions) across captured GTFS-RT."""
    from google.transit import gtfs_realtime_pb2

    files = sorted(glob.glob(str(raw / "*" / "*.pb")))
    total = 0
    for f in files:
        feed = gtfs_realtime_pb2.FeedMessage()
        try:
            feed.ParseFromString(open(f, "rb").read())
        except Exception:
            continue
        total += sum(1 for e in feed.entity if e.HasField("vehicle"))
    return len(files), total


def main() -> None:
    cfg = load_config()
    pq = project_path(cfg["paths"]["parquet"])
    raw = project_path(cfg["paths"]["raw"])

    rows: list[tuple[str, str]] = []
    for label, path in [
        ("GTFS stop_times (WY)", pq / "gtfs" / "stop_times"),
        ("GTFS trips (WY)", pq / "gtfs" / "trips"),
        ("GTFS stops (WY)", pq / "gtfs" / "stops"),
        ("GTFS routes (WY)", pq / "gtfs" / "routes"),
        ("IMD 2019 LSOAs", pq / "imd"),
        ("LSOA boundaries (WY)", pq / "lsoa"),
        ("Weather (hourly)", pq / "weather"),
        ("Stop->IMD join", pq / "stop_imd"),
        ("Scheduled headways", pq / "headway_sched"),
        ("Disruption situations", pq / "disruptions"),
        ("Positions (parsed)", pq / "positions"),
        ("Trip-stop delays", pq / "trip_stop_delay"),
    ]:
        n = count_parquet(path)
        rows.append((label, f"{n:,}" if n is not None else "-"))

    snaps, pos = count_positions_raw(raw)
    rows.append(("GTFS-RT snapshots captured (raw)", f"{snaps:,}"))
    rows.append(("Vehicle positions captured (raw)", f"{pos:,}"))

    width = max(len(r[0]) for r in rows)
    print(f"{'Dataset':<{width}}  Records (measured)")
    print("-" * (width + 20))
    total = 0
    for label, val in rows:
        print(f"{label:<{width}}  {val}")
        if val != "-" and "snapshots" not in label:
            total += int(val.replace(",", ""))
    print("-" * (width + 20))
    print(f"{'TOTAL (excl. snapshot count)':<{width}}  {total:,}")


if __name__ == "__main__":
    main()
