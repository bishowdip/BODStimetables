"""Roll the per-stop delays up to the compliance label used by the model.

Two levels:
  * fact_trip_delay        -- one row per (trip, service_date): the trip's
    on-time status and AVL-confirmation.
  * fact_route_band_day    -- route x time-band x service-day with pct_on_time
    and the compliant flag (>=85% of trips on time, ±2 min) -- the ML target.

A trip counts as on-time when its median stop-level delay sits inside the
primary band. Median (percentile_approx) is robust to the odd GPS jump that a
mean would chase. The ±3 / ±5 min sensitivity bands are computed alongside so
the report can show the headline isn't an artefact of one threshold.
"""
from __future__ import annotations

import logging

from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path

log = logging.getLogger("compute_reliability")


def time_band_col(hour_col, bands: dict):
    expr = F.lit(None)
    for name, (lo, hi) in bands.items():
        expr = F.when((hour_col >= lo) & (hour_col < hi), F.lit(name)).otherwise(expr)
    return expr


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    rel = cfg["reliability"]
    lo, hi = rel["ontime_lower_min"], rel["ontime_upper_min"]

    delay = spark.read.parquet(str(pq / "trip_stop_delay"))
    raw_n = delay.count()
    delay = delay.filter(F.abs(F.col("delay_min")) <= rel["max_abs_delay_min"]).cache()
    kept_n = delay.count()
    log.info("outlier filter (|delay| <= %d min): kept %s of %s stop events (%.2f%% dropped)",
             rel["max_abs_delay_min"], f"{kept_n:,}", f"{raw_n:,}", 100 * (1 - kept_n / raw_n))

    # ---- per-trip on-time (primary + sensitivity bands) ----
    band_min = [(lo, hi)] + [(-b, b) for b in rel["sensitivity_bands_min"]]
    trip = delay.groupBy("trip_id", "route_id", "service_date").agg(
        F.expr("percentile_approx(delay_min, 0.5)").alias("median_delay"),
        F.count("*").alias("stops_observed"),
        F.min("sched_sec").alias("first_sched_sec"),
    )
    for blo, bhi in band_min:
        tag = f"on_time_{abs(blo)}_{bhi}".replace("-", "")
        trip = trip.withColumn(
            tag, ((F.col("median_delay") >= blo) & (F.col("median_delay") <= bhi)).cast("int")
        )

    trip = trip.withColumn("hour", (F.col("first_sched_sec") / 3600).cast("int") % 24)
    trip = trip.withColumn("time_band", time_band_col(F.col("hour"), rel["time_bands"]))
    trip = trip.withColumn("day_of_week", F.date_format(F.to_date("service_date"), "EEEE"))

    trip.write.mode("overwrite").partitionBy("service_date").parquet(str(pq / "fact_trip_delay"))

    # ---- AVL-confirmed operation rate (the honest no-show proxy) ----
    scheduled = (
        spark.read.parquet(str(pq / "gtfs" / "stop_times"))
        .select("trip_id").distinct()
    )
    matched = trip.select("trip_id").distinct()
    confirmed = matched.count()
    sched_n = scheduled.count()
    log.info(
        "AVL-confirmed trips: %d of %d scheduled (%.1f%% floor)",
        confirmed, sched_n, 100.0 * confirmed / max(sched_n, 1),
    )

    # ---- route x time-band x day compliance label ----
    primary = f"on_time_{abs(lo)}_{hi}".replace("-", "")
    rbd = (
        trip.filter(F.col("time_band").isNotNull())
        .groupBy("route_id", "service_date", "time_band", "day_of_week")
        .agg(
            F.count("*").alias("n_trips"),
            F.avg(primary).alias("pct_on_time"),
            F.avg("median_delay").alias("mean_trip_delay"),
            F.stddev("median_delay").alias("sd_trip_delay"),
        )
        .withColumn(
            "compliant", (F.col("pct_on_time") >= rel["compliance_threshold"]).cast("int")
        )
    )
    rbd.write.mode("overwrite").partitionBy("service_date").parquet(str(pq / "fact_route_band_day"))

    n = rbd.count()
    comp = rbd.filter(F.col("compliant") == 1).count()
    log.info("route-band-days: %d, compliant: %d (%.1f%%)", n, comp, 100.0 * comp / max(n, 1))

    # ---- Travel Time Variability (brief metric: CV of trip duration, <=15%) ----
    # actual duration per trip from inferred pass times; CV per route over the window
    durations = (
        delay.groupBy("trip_id", "route_id")
        .agg(
            ((F.max("inferred_sec") - F.min("inferred_sec")) / 60.0).alias("actual_dur_min"),
            F.count("*").alias("stops_seen"),
        )
        .filter((F.col("stops_seen") >= 3) & (F.col("actual_dur_min") > 0))
    )
    ttv = (
        durations.groupBy("route_id")
        .agg(
            F.avg("actual_dur_min").alias("mean_dur_min"),
            F.stddev("actual_dur_min").alias("sd_dur_min"),
            F.count("*").alias("n_trips_measured"),
        )
        .withColumn("ttv_cv", F.col("sd_dur_min") / F.col("mean_dur_min"))
        .withColumn("ttv_ok", (F.col("ttv_cv") <= 0.15).cast("int"))
    )
    ttv.write.mode("overwrite").parquet(str(pq / "travel_time_variability"))
    log.info("travel-time variability written for %d routes", ttv.count())
    delay.unpersist()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("compute_reliability")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
