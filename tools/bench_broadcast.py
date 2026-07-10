"""Measure the effect of the broadcast join used in trip matching.

Runs the positions <-> scheduled-stops join twice on the real data -- once with
the small stops table broadcast (as the pipeline does) and once with broadcasting
disabled -- and records wall-clock for each. This is the before/after evidence
behind the optimisation claims in the report.

    python -m tools.bench_broadcast
"""
from __future__ import annotations

import json
import logging
import time

from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path
from src.process.trip_match import _gtfs_time_to_seconds, haversine_m

log = logging.getLogger("bench_broadcast")


def _run_join(spark, pq, radius: float, broadcast: bool) -> tuple[float, int]:
    positions = spark.read.parquet(str(pq / "positions")).filter(F.col("trip_id").isNotNull())
    stops = spark.read.parquet(str(pq / "gtfs" / "stops")).select("stop_id", "stop_lat", "stop_lon")
    stop_times = spark.read.parquet(str(pq / "gtfs" / "stop_times")).select(
        "trip_id", "stop_id", "arrival_time"
    )

    stops_side = F.broadcast(stops) if broadcast else stops
    sched = stop_times.join(stops_side, "stop_id").withColumn(
        "sched_sec", _gtfs_time_to_seconds(F.col("arrival_time"))
    )
    joined = (
        positions.join(sched, "trip_id")
        .withColumn("dist_m", haversine_m(F.col("lat"), F.col("lon"),
                                          F.col("stop_lat"), F.col("stop_lon")))
        .filter(F.col("dist_m") <= radius)
    )

    t0 = time.perf_counter()
    n = joined.count()  # forces full execution
    return round(time.perf_counter() - t0, 2), n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    pq = project_path(cfg["paths"]["parquet"])
    radius = cfg["reliability"]["stop_match_radius_m"]

    results = {}
    for label, flag, threshold in (("no_broadcast", False, "-1"), ("broadcast", True, None)):
        if threshold is not None:  # disable Spark's automatic broadcasting too
            spark = get_spark(f"bench-{label}")
            spark.conf.set("spark.sql.autoBroadcastJoinThreshold", threshold)
        else:
            spark = get_spark(f"bench-{label}")
        try:
            secs, rows = _run_join(spark, pq, radius, broadcast=flag)
            results[label] = {"seconds": secs, "rows_matched": rows}
            log.info("%s: %.2fs (%s candidate rows)", label, secs, f"{rows:,}")
        finally:
            spark.stop()

    out = project_path("docs", "results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "broadcast_bench.json").write_text(json.dumps(results, indent=2))
    if all(k in results for k in ("broadcast", "no_broadcast")):
        speedup = results["no_broadcast"]["seconds"] / results["broadcast"]["seconds"]
        log.info("broadcast speedup: %.2fx", speedup)


if __name__ == "__main__":
    main()
