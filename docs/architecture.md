# Architecture

Data ingestion -> processing -> storage/ML -> visualisation. Export this to
`architecture.png` for the report (the marker expects an image).

```
 INGESTION (Python, rate-limited, I/O-bound)
   download_archive.py  -> NDL GTFS-RT @60s + daily timetables
   parse_gtfsrt.py      -> filter to WY bbox, write Parquet (parse-and-stream)
   load_static.py       -> GTFS txt, IMD, LSOA -> Parquet
   fetch_weather.py     -> Open-Meteo hourly -> Parquet
        |  Parquet, partitioned by service_date
        v
 PROCESSING (PySpark, the big-data core)
   trip_match.py          -> positions <-> stop_times spatial-temporal join
                             (broadcast small stops/routes; partition positions)
   compute_reliability.py -> per-stop delay -> per-trip on-time -> AVL-confirmed
                             -> route x time-band x day compliant flag
   equity_join.py         -> stop -> LSOA -> IMD (GeoPandas, small) -> route IMD
        |  intermediate Parquet checkpoints (persistence between stages)
        v
 STORAGE (SQLite, parameterised)        ML (PySpark MLlib)
   dim_route / dim_stop / dim_lsoa        features.py  (leakage-safe, time split)
   fact_trip_delay                        train_models.py (LR / RF / GBT + CV)
   fact_route_band_day  (ML input)        evaluate.py  (F1, ROC-AUC, F1/sec)
        |                                       |
        v                                       v
 VISUALISATION (small aggregates -> pandas -> matplotlib / folium)
   reliability map  |  compliance x IMD decile  |  model comparison + ROC/PR
```

## Why the tool changes at each stage

| Stage | Tool | Reason |
|---|---|---|
| Download | Python `requests` | I/O-bound, capped at 1 req/s; Spark adds nothing |
| Parse GTFS-RT | Python (stream) then Spark | parse-and-discard national files to bound memory |
| Trip matching | PySpark | millions x thousands join; broadcast the small side |
| Aggregation | PySpark SQL | groupBy/agg over millions of rows; lazy DAG |
| Spatial join stop->LSOA | GeoPandas | only a few thousand unique stops -> not a big-data job |
| Feature table + ML | PySpark MLlib | pipeline + CrossValidator at scale |
| Plotting | pandas + matplotlib/folium | convert only the small aggregated outputs |

## Spark optimisation evidence (where to capture it)

- `spark.sql.shuffle.partitions = 8` (set in `src/common.py`).
- `broadcast(stops)` / `broadcast(routes)` in the trip-match and feature joins.
- `.cache()` on the trip-level delay table (reused by aggregation + ML).
- Parquet writes between stages = the persistence/checkpoint strategy.
- Spark UI: run a stage, open `localhost:4040`, screenshot the **Stages** tab
  during the big join (partition count + shuffle read/write) into
  `docs/spark_ui_*.png`. Capture it *while the job runs*.
