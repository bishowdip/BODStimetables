"""EDA over the static (timetable-side) data -- runs before positions exist.

Computes the brief's named statistics (mean, median, std, skewness, kurtosis)
with Spark functions over the real scheduled data, profiles data quality
(nulls, cardinality), and saves the headway distribution figure. Results land in
docs/results/eda_static.txt and docs/figures/.

    python -m src.viz.eda_static [--hold 120]

--hold keeps the SparkSession (and its UI at localhost:4040) alive that many
seconds after the work finishes, so the Stages tab can be captured for the
report's partition-utilisation evidence.
"""
from __future__ import annotations

import argparse
import logging
import time

from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path

log = logging.getLogger("eda_static")


def run(spark, cfg, out_lines: list[str]) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    st = spark.read.parquet(str(pq / "gtfs" / "stop_times")).cache()
    stops = spark.read.parquet(str(pq / "gtfs" / "stops"))
    trips = spark.read.parquet(str(pq / "gtfs" / "trips"))

    def emit(s: str) -> None:
        print(s)
        out_lines.append(s)

    emit("== Scale ==")
    emit(f"stop_times rows: {st.count():,}")
    emit(f"trips: {trips.count():,}  stops: {stops.count():,}")
    emit(f"stop_times partitions: {st.rdd.getNumPartitions()}")

    emit("\n== Profiling: nulls & cardinality (stop_times) ==")
    nulls = st.select(
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in ("trip_id", "stop_id", "arrival_time", "stop_sequence")]
    ).first()
    emit(f"null counts: {nulls.asDict()}")
    card = st.select(
        F.approx_count_distinct("trip_id").alias("trips"),
        F.approx_count_distinct("stop_id").alias("stops"),
    ).first()
    emit(f"cardinality (approx): {card.asDict()}")

    emit("\n== Brief statistics on scheduled headway (minutes) ==")
    hw = spark.read.parquet(str(pq / "headway_sched"))
    stats = hw.select(
        F.mean("sched_headway_mean_min").alias("mean"),
        F.expr("percentile_approx(sched_headway_mean_min, 0.5)").alias("median"),
        F.stddev("sched_headway_mean_min").alias("std"),
        F.skewness("sched_headway_mean_min").alias("skewness"),
        F.kurtosis("sched_headway_mean_min").alias("kurtosis"),
    ).first()
    emit(", ".join(f"{k}={v:.3f}" for k, v in stats.asDict().items()))

    emit("\n== Service intensity by time band (scheduled) ==")
    for row in hw.groupBy("time_band").agg(
        F.count("*").alias("route_bands"),
        F.avg("sched_headway_mean_min").alias("mean_headway_min"),
    ).orderBy("time_band").collect():
        emit(f"  {row.time_band:12s} route_bands={row.route_bands:5d} mean_headway={row.mean_headway_min:.1f} min")

    # figure: scheduled headway distribution (small aggregate -> pandas)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf = hw.select("sched_headway_mean_min").toPandas()
    figs = project_path("docs", "figures")
    figs.mkdir(parents=True, exist_ok=True)
    ax = pdf["sched_headway_mean_min"].clip(0, 120).hist(bins=48, figsize=(8, 5))
    ax.set_xlabel("scheduled headway (min)")
    ax.set_ylabel("route x time-band count")
    ax.set_title("Scheduled headway distribution, West Yorkshire")
    plt.tight_layout()
    plt.savefig(figs / "headway_distribution.png", dpi=150)
    plt.close()
    emit("\nfigure saved: docs/figures/headway_distribution.png")
    st.unpersist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hold", type=int, default=0, help="keep Spark UI alive N seconds after finishing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    spark = get_spark("eda-static")
    out_lines: list[str] = []
    try:
        run(spark, cfg, out_lines)
        results = project_path("docs", "results")
        results.mkdir(parents=True, exist_ok=True)
        (results / "eda_static.txt").write_text("\n".join(out_lines) + "\n")
        if args.hold:
            log.info("holding SparkSession for %ds -- UI at http://localhost:4040", args.hold)
            time.sleep(args.hold)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
