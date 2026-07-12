"""How much does the on-time window choice matter?

The headline uses the brief's urban bar (+/-2 min). A fair question is whether
the findings are an artefact of that choice, so trip-level on-time flags were
also computed at +/-3 and +/-5 during aggregation. This script rolls each band
up to route x time-band x day compliance and reports the compliant share side
by side.

Output: docs/results/sensitivity.json
"""
from __future__ import annotations

import json
import logging

from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path

log = logging.getLogger("sensitivity")


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    threshold = cfg["reliability"]["compliance_threshold"]
    trips = spark.read.parquet(str(pq / "fact_trip_delay"))

    bands = [c for c in trips.columns if c.startswith("on_time_")]
    rows = []
    for col in sorted(bands):
        label = "+/-" + col.split("_")[-1] + " min"
        rbd = (
            trips.filter(F.col("time_band").isNotNull())
            .groupBy("route_id", "service_date", "time_band")
            .agg(F.avg(col).alias("pct_on_time"))
            .withColumn("compliant", (F.col("pct_on_time") >= threshold).cast("int"))
        )
        agg = rbd.agg(
            F.count("*").alias("band_days"),
            F.avg("compliant").alias("compliant_share"),
            F.avg("pct_on_time").alias("mean_on_time"),
        ).first()
        rows.append(
            {
                "window": label,
                "band_days": agg.band_days,
                "compliant_share": round(float(agg.compliant_share), 4),
                "mean_on_time": round(float(agg.mean_on_time), 4),
            }
        )
        log.info("%s: %.1f%% of %s band-days compliant (mean on-time %.1f%%)",
                 label, 100 * rows[-1]["compliant_share"], f"{agg.band_days:,}",
                 100 * rows[-1]["mean_on_time"])

    out = project_path("docs", "results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("sensitivity")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
