# Bus Reliability in West Yorkshire

Big Data coursework project for analysing whether West Yorkshire bus services
meet an 85% on-time reliability standard and whether reliability differs across
areas of deprivation.

The project combines scheduled timetables, live bus positions, disruptions,
weather, IMD deprivation data and LSOA boundaries. It cleans and joins the data
with PySpark, stores the final facts in SQLite, trains predictive models, and
presents the results in a Streamlit dashboard.

## What This Does

- Captures real BODS GTFS-RT bus positions over a study window.
- Compares observed bus movement with scheduled GTFS timetable stops.
- Calculates delay, on-time status, route reliability, headway regularity and
  travel-time variability.
- Adds weather, disruption and deprivation features.
- Trains Logistic Regression, Random Forest and GBT models.
- Evaluates models with accuracy, precision, recall, F1, ROC-AUC, PR-AUC and
  confusion-matrix metrics.
- Builds a SQLite warehouse and Streamlit dashboard for reporting.

## Data Sources

- **BODS GTFS timetable**: scheduled routes, trips, stops and stop times.
- **BODS GTFS-RT vehicle positions**: live GPS positions collected locally.
- **BODS SIRI-SX disruptions**: incidents and road/service disruptions.
- **IMD 2019**: deprivation scores and deciles by LSOA.
- **ONS LSOA boundaries**: geographic areas used for deprivation joins.
- **Open-Meteo**: hourly weather features.

No synthetic data is used.

## Setup

Requirements: Python 3.13 and Java 17 or 21.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For live BODS feeds, create a local `.env` file:

```bash
BODS_API_KEY=your_key_here
```

## Run The Dashboard

```bash
source .venv/bin/activate
PYTHONPATH=. streamlit run src/viz/dashboard.py
```

## Rebuild The Project

The project settings are in `config/settings.yaml`.

```bash
python -m src.ingest.download_timetable
python -m src.ingest.download_supplementary
python -m src.ingest.fetch_weather
python -m src.ingest.parse_gtfsrt
python -m src.ingest.load_static
python -m src.process.trip_match
python -m src.process.compute_reliability
python -m src.process.headway
python -m src.process.match_stats
python -m src.process.sensitivity
python -m src.process.equity_join
python -m src.ml.features
python -m src.ml.train_models
python -m src.ml.evaluate
python -m src.ml.interpret
python -m src.db.load_db
python -m src.process.equity_stats
python -m src.viz.eda_static
python -m src.viz.eda_charts
python -m src.viz.plots
python -m src.viz.map
```

Live collection is separate because GTFS-RT and SIRI-SX must be captured while
the buses are running:

```bash
python -m src.ingest.poll_live --hours 168
python -m src.ingest.fetch_disruptions --every-hours 6 --for-hours 168
```

## Outputs

- `data/parquet/`: cleaned Spark datasets and model artefacts.
- `data/reliability.sqlite`: final reporting warehouse.
- `docs/results/`: metrics, confusion matrices, ROC/PR data and summaries.
- `docs/figures/`: generated charts and report figures.
- `src/viz/dashboard.py`: Streamlit dashboard.

## Tests

```bash
python -m pytest
```

## Project Structure

```text
src/ingest/    data collection and source loading
src/process/   Spark matching, cleaning, reliability and equity processing
src/ml/        feature engineering, training, evaluation and interpretation
src/db/        SQLite schema, loading and reporting queries
src/viz/       dashboard, maps and report figures
tests/         unit tests for metrics, features, geo logic and disruptions
```

