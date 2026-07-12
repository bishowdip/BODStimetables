# Architecture and performance notes

The pipeline runs in four layers: ingestion (plain Python), processing
(PySpark), storage and models (SQLite, MLlib), and visualisation (pandas +
matplotlib/folium on small aggregates). Parquet files partitioned by
service_date are the hand-off between every stage, so any stage can be re-run
alone. `docs/architecture.png` is the diagram; `tools/make_diagrams.py`
regenerates it.

## Tool choice per stage

| Stage | Tool | Reason |
|---|---|---|
| Collect positions and disruptions | Python (requests) | one HTTP call a minute is I/O-bound; Spark adds nothing |
| Parse GTFS-RT snapshots | Python | each protobuf file is parsed and discarded, so memory stays flat |
| Trip matching | PySpark | 4.9M positions joined to 2.6M stop_times is the distributed step |
| Reliability aggregation | PySpark SQL | groupBy over millions of matched events |
| Stop-to-LSOA join | GeoPandas | 16k stops is small; distributed tooling isn't justified at that size |
| Feature build and models | PySpark MLlib | pipelines and CrossValidator on the shared session |
| Plots and map | pandas + matplotlib/folium | only aggregated tables leave Spark |

## Measured performance (M3 MacBook Air, local[*], 8 shuffle partitions)

- Parsing the full week (10,546 snapshots, 11.4M raw pings) takes about a
  minute and writes 4.9M deduplicated rows.
- The matching join runs in 17 to 28 seconds depending on cache state
  (docs/results/timings.json).
- Broadcasting the stops table cuts the join from 8.98s to 4.37s, a 2.05x
  speedup with identical output (docs/results/broadcast_bench.json).
- `docs/spark_ui_stages.png` shows the stage list for the real join: 8/8 tasks
  per shuffle stage, shuffle reads up to 165 MiB, and nine skipped stages where
  Spark reused earlier shuffle output instead of recomputing it.

Caching: the matched-events table is cached once and reused by the on-time,
headway and variability aggregations. Persistence between stages is the
Parquet write itself.

## Complexity

| Operation | Cost | Note |
|---|---|---|
| Snapshot parsing | O(P) over P pings | linear scan, constant memory per file |
| Matching join | ~O(P log S) | broadcast avoids the naive P x S comparison |
| Aggregations | O(M) over M matched events | one shuffle by key each |
| Logistic regression | O(n d i) | n rows, d features, i iterations |
| Random forest / GBT | O(t n d log n) | t trees; GBT is sequential in t, hence slowest |

Training cost against quality is reported as F1 per training second in the
model comparison (docs/results/metrics.csv).

## Memory

A week of national GTFS-RT does not fit in pandas on an 8 GB laptop. The
pipeline deals with that twice over: snapshots are parsed one file at a time
and only the bbox-filtered rows kept, and everything downstream of Parquet is
Spark, which spills to disk when a partition exceeds memory. The one deliberate
exception is the LSOA spatial join, where the data is small enough that
GeoPandas is the simpler and faster choice.
