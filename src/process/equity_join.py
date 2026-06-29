"""Attach deprivation to stops and routes (the originality / equity layer).

The spatial join is stop -> LSOA polygon. There are only a few thousand unique
WY stops, so this join domain is *small* -- we deliberately drop out of Spark and
use GeoPandas here, and say so in the report: distributed tooling is justified by
data scale, and at this scale it is not. The result is then lifted back into
Spark to roll deprivation up to route level over the (large) stop_times table.

Outputs:
  * data/parquet/stop_imd       stop_id -> lsoa_code, imd_decile, imd_score
  * data/parquet/route_imd      route_id -> median IMD decile of stops served
"""
from __future__ import annotations

import logging

from src.common import get_spark, load_config, project_path

log = logging.getLogger("equity_join")


def stops_to_imd(cfg) -> "object":
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    pq = project_path(cfg["paths"]["parquet"])
    stops = pd.read_parquet(pq / "gtfs" / "stops").drop_duplicates("stop_id")
    lsoa = gpd.read_parquet(pq / "lsoa" / "lsoa.parquet").to_crs(4326)

    gstops = gpd.GeoDataFrame(
        stops,
        geometry=[Point(xy) for xy in zip(stops.stop_lon, stops.stop_lat)],
        crs=4326,
    )
    joined = gpd.sjoin(gstops, lsoa, how="left", predicate="within")

    # column names vary by LSOA file; pick the first that looks like the code
    code_col = next((c for c in joined.columns if "lsoa" in c.lower() and "cd" in c.lower()), None)
    if code_col is None:
        code_col = next((c for c in joined.columns if "lsoa" in c.lower()), "index_right")
    joined = joined.rename(columns={code_col: "lsoa_code"})

    imd = pd.read_parquet(pq / "imd" / "imd.parquet")
    imd_code = next((c for c in imd.columns if "lsoa" in c.lower() and "code" in c.lower()), imd.columns[0])
    dec_col = next((c for c in imd.columns if "decile" in c.lower()), None)
    sco_col = next((c for c in imd.columns if "score" in c.lower()), None)
    imd = imd.rename(columns={imd_code: "lsoa_code"})
    keep = ["lsoa_code"] + [c for c in (dec_col, sco_col) if c]
    out = joined[["stop_id", "lsoa_code"]].merge(imd[keep], on="lsoa_code", how="left")
    out = out.rename(columns={dec_col: "imd_decile", sco_col: "imd_score"})

    dest = pq / "stop_imd"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest / "stop_imd.parquet", index=False)
    log.info("stop_imd written: %d stops, %d matched to an LSOA", len(out), out.lsoa_code.notna().sum())
    return out


def route_deprivation(spark, cfg) -> None:
    from pyspark.sql import functions as F

    pq = project_path(cfg["paths"]["parquet"])
    stop_imd = spark.read.parquet(str(pq / "stop_imd" / "stop_imd.parquet"))
    stop_times = spark.read.parquet(str(pq / "gtfs" / "stop_times"))
    trips = spark.read.parquet(str(pq / "gtfs" / "trips")).select("trip_id", "route_id")

    route_stops = (
        stop_times.select("trip_id", "stop_id").distinct()
        .join(F.broadcast(trips), "trip_id")
        .join(stop_imd, "stop_id")
        .select("route_id", "stop_id", "imd_decile")
        .distinct()
    )
    route_imd = route_stops.groupBy("route_id").agg(
        F.expr("percentile_approx(imd_decile, 0.5)").alias("route_imd_decile"),
        F.min("imd_decile").alias("route_min_imd_decile"),
        F.countDistinct("stop_id").alias("n_stops_with_imd"),
    )
    route_imd.write.mode("overwrite").parquet(str(pq / "route_imd"))
    log.info("route_imd written: %d routes", route_imd.count())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    stops_to_imd(cfg)

    spark = get_spark("equity_join")
    try:
        route_deprivation(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
