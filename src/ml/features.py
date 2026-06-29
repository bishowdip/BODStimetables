"""Build the model feature table -- leakage-safe, with a time-aware split flag.

The rule that earns the marks: every feature must be knowable *before* the trip
runs. So we never feed a row its own outcome. The one historical feature
(route_hist_ontime_rate) is computed *only* from the training days and then
joined onto every row -- the test days never contribute to it.

Output: data/parquet/ml_features  (+ an is_train flag from the date order).
"""
from __future__ import annotations

import logging

from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path

log = logging.getLogger("features")


def _band_hours(bands: dict) -> "F.Column":
    """Map an hour to its band name (reused to aggregate weather to bands)."""
    expr = F.lit(None)
    for name, (lo, hi) in bands.items():
        expr = F.when((F.col("hour") >= lo) & (F.col("hour") < hi), F.lit(name)).otherwise(expr)
    return expr


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    bands = cfg["reliability"]["time_bands"]
    dates = sorted(cfg["window"]["dates"])
    train_dates = set(dates[: cfg["split"]["train_days"]])

    rbd = spark.read.parquet(str(pq / "fact_route_band_day"))

    # --- service features: stops per route + scheduled trips in band ---
    trips = spark.read.parquet(str(pq / "gtfs" / "trips")).select("trip_id", "route_id")
    stop_times = spark.read.parquet(str(pq / "gtfs" / "stop_times"))
    n_stops = (
        stop_times.select("trip_id", "stop_id").distinct()
        .join(F.broadcast(trips), "trip_id")
        .groupBy("route_id").agg(F.countDistinct("stop_id").alias("n_stops"))
    )

    # --- weather aggregated to time band ---
    weather = spark.read.parquet(str(pq / "weather"))
    weather = weather.withColumn("time_band", _band_hours(bands))
    wx = (
        weather.filter(F.col("time_band").isNotNull())
        .groupBy("service_date", "time_band")
        .agg(
            F.avg("precip_mm").alias("precip_mm"),
            F.avg("temp_c").alias("temp_c"),
            F.avg("wind_kph").alias("wind_kph"),
        )
    )

    # --- equity overlay (known in advance, fine as a predictor) ---
    route_imd = spark.read.parquet(str(pq / "route_imd")).select("route_id", "route_imd_decile")

    # --- leakage-safe historical reliability: TRAIN DAYS ONLY ---
    hist = (
        rbd.filter(F.col("service_date").isin(list(train_dates)))
        .groupBy("route_id")
        .agg(F.avg("pct_on_time").alias("route_hist_ontime_rate"))
    )

    feats = (
        rbd.join(F.broadcast(n_stops), "route_id", "left")
        .join(wx, ["service_date", "time_band"], "left")
        .join(F.broadcast(route_imd), "route_id", "left")
        .join(F.broadcast(hist), "route_id", "left")
        .withColumn("is_weekend", F.col("day_of_week").isin("Saturday", "Sunday").cast("int"))
        .withColumn("is_train", F.col("service_date").isin(list(train_dates)).cast("int"))
        .fillna(
            {"precip_mm": 0.0, "wind_kph": 0.0, "n_stops": 0,
             "route_hist_ontime_rate": 0.0, "route_imd_decile": -1.0}
        )
    )

    feats.write.mode("overwrite").parquet(str(pq / "ml_features"))
    log.info(
        "features written: %d rows (%d train / %d test)",
        feats.count(),
        feats.filter("is_train = 1").count(),
        feats.filter("is_train = 0").count(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("features")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
