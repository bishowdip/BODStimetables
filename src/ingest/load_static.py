"""Load the static / supplementary datasets to Parquet.

  * GTFS timetable txt files  -> read with Spark (stop_times is the big one),
    filter stops to the WY bbox and keep only the trips/stop_times that touch
    those stops, write Parquet.
  * IMD 2019 deprivation CSV   -> small, pandas.
  * LSOA 2011 boundaries       -> small, geopandas (geometry kept for the join).

The GTFS zips are unzipped per date under data/raw/<date>/gtfs/. Because the
timetable is the same across days for most operators, we read every date and
deduplicate on the natural keys.
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

from src.common import load_config, project_path

log = logging.getLogger("load_static")

GTFS_TABLES = ("stops", "routes", "trips", "stop_times", "calendar")


def _unzip_timetables(dates, raw_root: Path) -> list[Path]:
    # Per-date archive zips, plus a single current GTFS (the live timetable path).
    candidates = [(raw_root / d / "timetable.zip", raw_root / d / "gtfs") for d in dates]
    candidates.append((raw_root / "timetable.zip", raw_root / "gtfs"))

    gtfs_dirs = []
    for zpath, out in candidates:
        if not zpath.exists():
            continue
        if not out.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(out)
            log.info("unzipped %s", zpath)
        gtfs_dirs.append(out)
    return gtfs_dirs


def load_gtfs(spark, gtfs_dirs, bbox, out_root: Path) -> None:
    from pyspark.sql import functions as F

    def read_table(name):
        paths = [str(d / f"{name}.txt") for d in gtfs_dirs if (d / f"{name}.txt").exists()]
        if not paths:
            return None
        return spark.read.option("header", True).option("inferSchema", True).csv(paths)

    stops = read_table("stops")
    if stops is None:
        log.error("no stops.txt found -- did the download/unzip run?")
        return

    stops = stops.dropDuplicates(["stop_id"]).filter(
        (F.col("stop_lat") >= bbox["min_lat"]) & (F.col("stop_lat") <= bbox["max_lat"])
        & (F.col("stop_lon") >= bbox["min_lon"]) & (F.col("stop_lon") <= bbox["max_lon"])
    )
    wy_stop_ids = stops.select("stop_id").distinct()

    out = out_root / "gtfs"
    stops.write.mode("overwrite").parquet(str(out / "stops"))

    routes = read_table("routes")
    if routes is not None:
        routes.dropDuplicates(["route_id"]).write.mode("overwrite").parquet(str(out / "routes"))

    trips = read_table("trips")
    stop_times = read_table("stop_times")
    if stop_times is not None and trips is not None:
        # keep only stop_times at WY stops, then the trips that survive
        st = stop_times.join(F.broadcast(wy_stop_ids), "stop_id")
        wy_trip_ids = st.select("trip_id").distinct()
        st.write.mode("overwrite").parquet(str(out / "stop_times"))
        trips.join(F.broadcast(wy_trip_ids), "trip_id").dropDuplicates(["trip_id"]) \
            .write.mode("overwrite").parquet(str(out / "trips"))

    calendar = read_table("calendar")
    if calendar is not None:
        calendar.write.mode("overwrite").parquet(str(out / "calendar"))
    log.info("GTFS static tables written under %s", out)


def load_imd(cfg, out_root: Path) -> None:
    import pandas as pd

    path = cfg["sources"].get("imd_csv")
    if not path or not Path(path).exists():
        log.warning("IMD csv not configured/found (sources.imd_csv) -- skipping")
        return
    imd = pd.read_csv(path)
    out = out_root / "imd"
    out.mkdir(parents=True, exist_ok=True)
    imd.to_parquet(out / "imd.parquet", index=False)
    log.info("IMD written: %d rows", len(imd))


def load_lsoa(cfg, out_root: Path) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        log.warning("geopandas not installed -- skipping LSOA boundaries")
        return
    path = cfg["sources"].get("lsoa_geojson")
    if not path or not Path(path).exists():
        log.warning("LSOA boundaries not configured/found (sources.lsoa_geojson) -- skipping")
        return
    gdf = gpd.read_file(path)
    out = out_root / "lsoa"
    out.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out / "lsoa.parquet")
    log.info("LSOA boundaries written: %d polygons", len(gdf))


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

    gtfs_dirs = _unzip_timetables(dates, raw_root)
    if gtfs_dirs:
        from src.common import get_spark

        spark = get_spark("load_static")
        try:
            load_gtfs(spark, gtfs_dirs, bbox, out_root)
        finally:
            spark.stop()
    else:
        log.warning("no timetable zips found -- skipping GTFS load")

    load_imd(cfg, out_root)
    load_lsoa(cfg, out_root)


if __name__ == "__main__":
    main()
