# Bus Reliability and Deprivation in West Yorkshire

ST5011CEM Big Data project. Measures how reliably West Yorkshire buses run
against the 85% on-time standard (±2 min), predicts which route/time-band/day
combinations will fail it, and checks whether unreliability falls hardest on
routes serving deprived areas.

## Data

All real, no synthetic rows:

- **BODS timetables** – Yorkshire GTFS download (2.5M+ stop_times after
  filtering to West Yorkshire)
- **BODS vehicle positions** – GTFS-RT live feed, self-archived at 60s
  intervals over a 7-day window (the feed is live-only, so it has to be
  captured as it happens)
- **BODS disruptions** – SIRI-SX feed, snapshotted 6-hourly over the same window
- **IMD 2019** (gov.uk), **LSOA 2011 boundaries** (ONS), **hourly weather**
  (Open-Meteo) as supplementary joins

A free BODS API key is needed for the two live feeds. Put it in a `.env` file
as `BODS_API_KEY=...` — it is git-ignored and nothing else needs a key.

## Setup

Needs Python 3.13 and Java 17 or 21 (Spark 4 does not run on Java 24+; the
code picks a compatible JDK automatically if you have several).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Dates, bounding box and thresholds live in `config/settings.yaml`.

```bash
# collect (positions need to run across the study window)
python -m src.ingest.download_timetable
python -m src.ingest.download_supplementary
python -m src.ingest.fetch_weather
python -m src.ingest.poll_live --hours 168
python -m src.ingest.fetch_disruptions --every-hours 6 --for-hours 168

# process (Spark)
python -m src.ingest.parse_gtfsrt
python -m src.ingest.load_static
python -m src.process.trip_match          # --hold N keeps the Spark UI up
python -m src.process.compute_reliability
python -m src.process.headway
python -m src.process.match_stats
python -m src.process.sensitivity
python -m src.process.equity_join

# store and analyse
python -m src.db.load_db && python -m src.db.dump
python -m src.process.equity_stats

# model
python -m src.ml.features
python -m src.ml.train_models
python -m src.ml.evaluate
python -m src.ml.interpret

# figures and map
python -m src.viz.eda_static
python -m src.viz.plots && python -m src.viz.map
```

`python -m pytest` runs the tests. Raw downloads and generated outputs are
git-ignored; every artefact is reproducible from the commands above.
