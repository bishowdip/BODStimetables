"""Match live vehicle positions to scheduled stop times (the big-data core).

This is the canonical millions x thousands distributed join, so it lives in
Spark. The small tables (stops, routes) are broadcast; positions are the large
side and stay partitioned by service_date.

Algorithm (trip_id path):
  1. Join positions to that trip's scheduled stop_times on trip_id.
  2. Attach each stop's lat/lon (broadcast stops) and compute the haversine
     distance from every ping to every stop on the trip.
  3. For each (trip_id, stop) keep the nearest ping inside the match radius --
     its timestamp is the inferred pass time.
  4. delay = inferred_pass_time - scheduled_arrival_time.

Positions with no trip_id are counted but not matched here; the report documents
the resulting match rate and the spatial-temporal fallback as future work.

Output: data/parquet/trip_stop_delay (intermediate Parquet checkpoint, reused by
the reliability and equity stages -- this is the persistence-between-stages
strategy cited in the optimisation section).
"""
from __future__ import annotations

import logging

from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common import get_spark, load_config, project_path

log = logging.getLogger("trip_match")

EARTH_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres as a Spark column expression."""
    rlat1, rlat2 = F.radians(lat1), F.radians(lat2)
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.sin(dlat / 2) ** 2 + F.cos(rlat1) * F.cos(rlat2) * F.sin(dlon / 2) ** 2
    return F.lit(EARTH_M) * 2 * F.asin(F.sqrt(a))


def _gtfs_time_to_seconds(col):
    """GTFS times can exceed 24:00:00 (after-midnight trips). Parse to seconds."""
    parts = F.split(col, ":")
    return (
        parts.getItem(0).cast(T.IntegerType()) * 3600
        + parts.getItem(1).cast(T.IntegerType()) * 60
        + parts.getItem(2).cast(T.IntegerType())
    )


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    radius = cfg["reliability"]["stop_match_radius_m"]

    positions = spark.read.parquet(str(pq / "positions"))
    stops = spark.read.parquet(str(pq / "gtfs" / "stops"))
    stop_times = spark.read.parquet(str(pq / "gtfs" / "stop_times"))

    # scheduled stop events with coordinates (small enough to broadcast stops)
    sched = (
        stop_times.select("trip_id", "stop_id", "stop_sequence", "arrival_time")
        .join(F.broadcast(stops.select("stop_id", "stop_lat", "stop_lon")), "stop_id")
        .withColumn("sched_sec", _gtfs_time_to_seconds(F.col("arrival_time")))
    )

    matched_only = positions.filter(F.col("trip_id").isNotNull())

    # join pings to their trip's scheduled stops, distance every ping->stop
    joined = (
        matched_only.alias("p")
        .join(sched.alias("s"), "trip_id")
        .withColumn(
            "dist_m",
            haversine_m(F.col("p.lat"), F.col("p.lon"), F.col("s.stop_lat"), F.col("s.stop_lon")),
        )
        .filter(F.col("dist_m") <= radius)
    )

    # nearest ping per (trip, stop) -> inferred pass time
    from pyspark.sql.window import Window

    w = Window.partitionBy("trip_id", "stop_id").orderBy(F.col("dist_m").asc())
    nearest = (
        joined.withColumn("rk", F.row_number().over(w))
        .filter(F.col("rk") == 1)
        .withColumn("inferred_sec", F.hour("ts_utc") * 3600 + F.minute("ts_utc") * 60 + F.second("ts_utc"))
    )

    delay = nearest.select(
        "trip_id",
        F.col("route_id"),
        "service_date",
        "stop_id",
        "stop_sequence",
        "sched_sec",
        "inferred_sec",
        F.round((F.col("inferred_sec") - F.col("sched_sec")) / 60.0, 2).alias("delay_min"),
        F.col("dist_m"),
    )

    out = pq / "trip_stop_delay"
    delay.write.mode("overwrite").partitionBy("service_date").parquet(str(out))

    total_pings = positions.count()
    with_trip = matched_only.count()
    log.info(
        "positions=%d, with trip_id=%d (%.1f%%), matched stop-events=%d",
        total_pings, with_trip, 100.0 * with_trip / max(total_pings, 1), delay.count(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("trip_match")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
