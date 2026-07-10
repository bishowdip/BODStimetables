"""Match-rate and AVL-confirmed operation statistics (the honest no-show floor).

For each study date, GTFS calendar (day-of-week flags + validity range) gives
the trips genuinely scheduled to run that day. Comparing with the trips that
left at least one matched position yields the AVL-confirmed operation rate.

Framing matters and is deliberate: a scheduled trip with no live trace is
*unconfirmed*, not proven cancelled -- a bus can run without broadcasting AVL.
So the rate is a floor on operation, reported per day and per operator, with
low-coverage operators visible so the report can restrict claims to operators
whose feeds are trustworthy.

Output: docs/results/match_stats.json  (+ a readable log summary)
"""
from __future__ import annotations

import json
import logging
from datetime import date

from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path

log = logging.getLogger("match_stats")

WEEKDAY_COLS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def scheduled_trips_per_date(spark, pq, dates: list[str]):
    """(service_date, trip_id) rows for every trip scheduled on each window date."""
    cal = spark.read.parquet(str(pq / "gtfs" / "calendar"))
    trips = spark.read.parquet(str(pq / "gtfs" / "trips")).select("trip_id", "route_id", "service_id")

    per_date = None
    for d in dates:
        dow = WEEKDAY_COLS[date.fromisoformat(d).weekday()]
        yyyymmdd = int(d.replace("-", ""))
        active = cal.filter(
            (F.col(dow) == 1)
            & (F.col("start_date") <= yyyymmdd)
            & (F.col("end_date") >= yyyymmdd)
        ).select("service_id")
        day = trips.join(F.broadcast(active), "service_id").withColumn("service_date", F.lit(d))
        per_date = day if per_date is None else per_date.unionByName(day)
    return per_date


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    dates = cfg["window"]["dates"]

    scheduled = scheduled_trips_per_date(spark, pq, dates).cache()
    matched = (
        spark.read.parquet(str(pq / "trip_stop_delay"))
        .select("service_date", "trip_id").distinct()
        .withColumn("confirmed", F.lit(1))
    )

    joined = scheduled.join(matched, ["service_date", "trip_id"], "left").fillna({"confirmed": 0})

    by_day = (
        joined.groupBy("service_date")
        .agg(F.count("*").alias("scheduled"), F.sum("confirmed").alias("confirmed"))
        .withColumn("avl_confirmed_rate", F.round(F.col("confirmed") / F.col("scheduled"), 4))
        .orderBy("service_date")
    )

    routes = spark.read.parquet(str(pq / "gtfs" / "routes")).select("route_id", "agency_id")
    by_operator = (
        joined.join(F.broadcast(routes), "route_id", "left")
        .groupBy("agency_id")
        .agg(F.count("*").alias("scheduled"), F.sum("confirmed").alias("confirmed"))
        .withColumn("avl_confirmed_rate", F.round(F.col("confirmed") / F.col("scheduled"), 4))
        .filter(F.col("scheduled") >= 100)  # ignore tiny operators: rates are noise
        .orderBy(F.desc("scheduled"))
    )

    day_rows = [r.asDict() for r in by_day.collect()]
    op_rows = [r.asDict() for r in by_operator.collect()]
    total_sched = sum(r["scheduled"] for r in day_rows)
    total_conf = sum(r["confirmed"] for r in day_rows)

    out = {
        "window": {"dates": dates},
        "total_scheduled_trip_days": total_sched,
        "total_avl_confirmed": total_conf,
        "overall_avl_confirmed_rate": round(total_conf / total_sched, 4) if total_sched else None,
        "by_day": day_rows,
        "by_operator": op_rows,
    }
    results = project_path("docs", "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "match_stats.json").write_text(json.dumps(out, indent=2))

    log.info("scheduled trip-days: %s | AVL-confirmed: %s (floor %.1f%%)",
             f"{total_sched:,}", f"{total_conf:,}", 100 * out["overall_avl_confirmed_rate"])
    for r in day_rows:
        log.info("  %s: %s/%s (%.1f%%)", r["service_date"], f"{r['confirmed']:,}",
                 f"{r['scheduled']:,}", 100 * r["avl_confirmed_rate"])
    scheduled.unpersist()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("match_stats")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
