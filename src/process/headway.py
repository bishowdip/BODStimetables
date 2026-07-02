"""Headway Regularity (the brief's second metric) -- scheduled and observed.

Headway = interval between successive vehicles of the same route at the same
stop. The brief's bar: SD of observed headway <= 20% of the scheduled headway.

Two sides, same Spark computation:
  * scheduled -- from stop_times (real timetable, available immediately):
    successive scheduled arrivals per (route, stop), diffed, aggregated to
    route x time-band. Also exported as leakage-safe ML features (the schedule
    is known before any trip runs).
  * observed  -- same, but over inferred pass times from trip_stop_delay
    (exists only after trip matching has run on captured positions).

Output:
  data/parquet/headway_sched  (route_id, time_band, sched_headway_mean_min, sched_headway_sd_min)
  data/parquet/headway_obs    (route_id, service_date, time_band, obs_headway_sd_min,
                               headway_regular flag)  -- when positions exist
"""
from __future__ import annotations

import logging

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common import get_spark, load_config, project_path
from src.process.compute_reliability import time_band_col

log = logging.getLogger("headway")


def _headways(df, time_col: str, extra_keys: list[str]):
    """Diff successive events per (route, stop [, date]) -> headway minutes."""
    keys = ["route_id", "stop_id"] + extra_keys
    w = Window.partitionBy(*keys).orderBy(F.col(time_col))
    return (
        df.withColumn("prev_t", F.lag(time_col).over(w))
        .filter(F.col("prev_t").isNotNull())
        .withColumn("headway_min", (F.col(time_col) - F.col("prev_t")) / 60.0)
        # ignore gaps > 3h: those are service breaks, not operating headway
        .filter((F.col("headway_min") > 0) & (F.col("headway_min") <= 180))
    )


def scheduled(spark, cfg, pq) -> None:
    bands = cfg["reliability"]["time_bands"]
    stop_times = spark.read.parquet(str(pq / "gtfs" / "stop_times"))
    trips = spark.read.parquet(str(pq / "gtfs" / "trips")).select("trip_id", "route_id")

    from src.process.trip_match import _gtfs_time_to_seconds

    ev = (
        stop_times.select("trip_id", "stop_id", "arrival_time")
        .join(F.broadcast(trips), "trip_id")
        .withColumn("sched_sec", _gtfs_time_to_seconds(F.col("arrival_time")))
        .withColumn("hour", (F.col("sched_sec") / 3600).cast("int") % 24)
        .withColumn("time_band", time_band_col(F.col("hour"), bands))
        .filter(F.col("time_band").isNotNull())
    )
    hw = _headways(ev, "sched_sec", ["time_band"])
    out = (
        hw.groupBy("route_id", "time_band")
        .agg(
            F.avg("headway_min").alias("sched_headway_mean_min"),
            F.stddev("headway_min").alias("sched_headway_sd_min"),
            F.count("*").alias("n_headways"),
        )
    )
    out.write.mode("overwrite").parquet(str(pq / "headway_sched"))
    log.info("scheduled headway: %d route-bands", out.count())


def observed(spark, cfg, pq) -> None:
    src = pq / "trip_stop_delay"
    if not src.exists():
        log.info("no trip_stop_delay yet -- observed headway will run after trip matching")
        return
    bands = cfg["reliability"]["time_bands"]
    ev = (
        spark.read.parquet(str(src))
        .withColumn("hour", (F.col("sched_sec") / 3600).cast("int") % 24)
        .withColumn("time_band", time_band_col(F.col("hour"), bands))
        .filter(F.col("time_band").isNotNull())
    )
    hw = _headways(ev, "inferred_sec", ["service_date", "time_band"])
    sched = spark.read.parquet(str(pq / "headway_sched")).select(
        "route_id", "time_band", "sched_headway_mean_min"
    )
    out = (
        hw.groupBy("route_id", "service_date", "time_band")
        .agg(F.stddev("headway_min").alias("obs_headway_sd_min"), F.count("*").alias("n_obs"))
        .join(F.broadcast(sched), ["route_id", "time_band"], "left")
        # brief's bar: SD <= 20% of scheduled headway
        .withColumn(
            "headway_regular",
            (F.col("obs_headway_sd_min") <= 0.2 * F.col("sched_headway_mean_min")).cast("int"),
        )
    )
    out.write.mode("overwrite").parquet(str(pq / "headway_obs"))
    log.info("observed headway: %d route-band-days", out.count())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    pq = project_path(cfg["paths"]["parquet"])
    spark = get_spark("headway")
    try:
        scheduled(spark, cfg, pq)
        observed(spark, cfg, pq)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
