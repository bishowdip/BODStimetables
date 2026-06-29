"""Build the SQLite warehouse from the processed Parquet tables.

Everything is inserted with parameterised executemany (?-placeholders) -- never
string-formatted SQL -- which is the SQL-injection-safe pattern the brief asks us
to demonstrate. The DB is rebuilt from scratch each run from schema.sql.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.common import load_config, project_path

log = logging.getLogger("load_db")


def _read_parquet_dir(path: Path) -> pd.DataFrame:
    """Read a Spark parquet directory (or single file) into pandas."""
    if path.is_dir():
        return pd.read_parquet(path)
    return pd.read_parquet(path)


def _insert(conn, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    cols = [c for c in columns if c in df.columns]
    df = df[cols].where(pd.notnull(df[cols]), None)
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, df.itertuples(index=False, name=None))
    log.info("%s: %d rows", table, len(df))


def build(cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    db_path = project_path(cfg["paths"]["db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    schema_sql = (Path(__file__).parent / "schema.sql").read_text()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)

        # dim_lsoa + stop->lsoa from the equity stage
        stop_imd = _read_parquet_dir(pq / "stop_imd" / "stop_imd.parquet")
        lsoa = stop_imd.dropna(subset=["lsoa_code"]).drop_duplicates("lsoa_code")
        _insert(conn, "dim_lsoa", lsoa, ["lsoa_code", "imd_decile", "imd_score"])

        stops = _read_parquet_dir(pq / "gtfs" / "stops").drop_duplicates("stop_id")
        stops = stops.rename(columns={"stop_name": "name", "stop_lat": "lat", "stop_lon": "lon"})
        stops = stops.merge(stop_imd[["stop_id", "lsoa_code"]], on="stop_id", how="left")
        _insert(conn, "dim_stop", stops, ["stop_id", "name", "lat", "lon", "lsoa_code"])

        # dim_route (+ deprivation overlay)
        routes = _read_parquet_dir(pq / "gtfs" / "routes").drop_duplicates("route_id")
        routes = routes.rename(columns={"route_short_name": "route_short_name", "agency_id": "operator"})
        route_imd = _read_parquet_dir(pq / "route_imd")
        routes = routes.merge(route_imd[["route_id", "route_imd_decile"]], on="route_id", how="left")
        _insert(conn, "dim_route", routes, ["route_id", "operator", "route_short_name", "route_imd_decile"])

        # facts
        trip = _read_parquet_dir(pq / "fact_trip_delay")
        on_time_col = next((c for c in trip.columns if c.startswith("on_time_")), None)
        if on_time_col:
            trip = trip.rename(columns={on_time_col: "on_time"})
        _insert(conn, "fact_trip_delay", trip,
                ["trip_id", "route_id", "service_date", "time_band", "day_of_week",
                 "stops_observed", "median_delay", "on_time"])

        rbd = _read_parquet_dir(pq / "fact_route_band_day")
        _insert(conn, "fact_route_band_day", rbd,
                ["route_id", "service_date", "time_band", "day_of_week",
                 "n_trips", "pct_on_time", "sd_trip_delay", "compliant"])

        conn.commit()
        log.info("warehouse built at %s", db_path)
    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(load_config())


if __name__ == "__main__":
    main()
