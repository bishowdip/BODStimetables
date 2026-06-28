# Who Gets the Worst Bus? Bus service reliability and equity in West Yorkshire

Big Data Programming Project (ST5011CEM). The pipeline measures route-level
service reliability against the regulator's 85% on-time bar using real BODS
timetables and archived live vehicle positions, predicts non-compliance from
features known *before* a trip runs, and tests whether failures concentrate on
routes serving the most deprived areas.

## What it does

1. **Ingest** – rate-limited download of archived GTFS-RT positions and daily
   GTFS timetables for a 7-day window, plus hourly weather (Open-Meteo), IMD
   2019 deprivation and LSOA boundaries. Everything is filtered to the West
   Yorkshire bounding box and written to Parquet.
2. **Process (Spark)** – match positions to scheduled stop times, infer per-stop
   delay, roll up to a per-trip on-time rate, then aggregate to
   route × time-band × service-day with a `compliant` flag (>=85% on time, ±2 min).
3. **Store** – load the star schema into SQLite via parameterised queries.
4. **Model (MLlib)** – Logistic Regression, Random Forest and Gradient-Boosted
   Trees on an identical, leakage-safe feature set with a time-aware split.
5. **Visualise** – reliability map, compliance vs IMD decile, model comparison.

## Setup

Tested on macOS (Apple Silicon) with Python 3.13 and a JDK on the path.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Spark 4 runs on Java 17 or 21. Check `java -version`; if you need to pin it:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # or -v 21
```

## Run order

The scripts read everything from `config/settings.yaml` (study dates, bbox,
on-time thresholds, paths). Edit that first, then:

```bash
# 1. ingestion  (writes Parquet under data/parquet/, partitioned by service_date)
python -m src.ingest.download_archive
python -m src.ingest.parse_gtfsrt
python -m src.ingest.load_static
python -m src.ingest.fetch_weather

# 2. processing (the Spark core)
python -m src.process.trip_match
python -m src.process.compute_reliability
python -m src.process.equity_join

# 3. database
python -m src.db.load_db

# 4. machine learning
python -m src.ml.features
python -m src.ml.train_models
python -m src.ml.evaluate

# 5. visualisation
python -m src.viz.plots
python -m src.viz.map
```

A small committed sample lives in `data/sample/` so the processing and ML stages
can be smoke-tested without the full multi-GB download.

## Repository map

```
config/      settings.yaml – bbox, dates, thresholds, paths (no secrets)
data/sample/ small committed sample for testing
src/ingest/  download + parse + static loaders
src/process/ trip matching, reliability, equity join (PySpark)
src/db/      schema.sql, loader, parameterised query examples
src/ml/      feature build, model training, evaluation
src/viz/     matplotlib + folium outputs
notebooks/   01_eda, 02_results
docs/        architecture + schema diagrams, Spark UI screenshots
tests/       unit tests for the matching + feature logic
```

## Notes

- Open-Meteo needs no API key; SQLite is keyless. There are no credentials in
  this repo. Any machine-specific override goes in `config/settings.local.yaml`,
  which is git-ignored.
- Raw downloads and generated Parquet are git-ignored — rerun the ingest scripts
  to regenerate them.
