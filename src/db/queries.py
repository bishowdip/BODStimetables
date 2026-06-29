"""Sample analytical queries over the warehouse.

Every query uses ?-placeholders and passes user-controlled values as parameters,
never via string formatting -- this is the parameterised / injection-safe pattern
required by the brief. Run directly to print a few results:

    python -m src.db.queries
"""
from __future__ import annotations

import sqlite3

from src.common import load_config, project_path


def connect() -> sqlite3.Connection:
    cfg = load_config()
    conn = sqlite3.connect(project_path(cfg["paths"]["db"]))
    conn.row_factory = sqlite3.Row
    return conn


def worst_routes(conn, limit: int = 10) -> list[sqlite3.Row]:
    """Routes with the lowest compliance rate across the window."""
    sql = """
        SELECT r.route_id, r.route_short_name,
               AVG(f.compliant)  AS compliance_rate,
               COUNT(*)          AS band_days
        FROM fact_route_band_day f
        JOIN dim_route r ON r.route_id = f.route_id
        GROUP BY r.route_id
        HAVING band_days >= ?
        ORDER BY compliance_rate ASC
        LIMIT ?
    """
    return conn.execute(sql, (3, limit)).fetchall()


def compliance_by_imd_decile(conn) -> list[sqlite3.Row]:
    """The equity headline: compliance against deprivation decile."""
    sql = """
        SELECT CAST(r.route_imd_decile AS INT) AS imd_decile,
               AVG(f.compliant)                AS compliance_rate,
               COUNT(*)                        AS band_days
        FROM fact_route_band_day f
        JOIN dim_route r ON r.route_id = f.route_id
        WHERE r.route_imd_decile IS NOT NULL
        GROUP BY imd_decile
        ORDER BY imd_decile
    """
    return conn.execute(sql).fetchall()


def compliance_for_band(conn, time_band: str) -> list[sqlite3.Row]:
    """Compliance for one time band -- shows the parameterised filter."""
    sql = """
        SELECT day_of_week, AVG(pct_on_time) AS mean_on_time, COUNT(*) AS n
        FROM fact_route_band_day
        WHERE time_band = ?
        GROUP BY day_of_week
        ORDER BY mean_on_time
    """
    return conn.execute(sql, (time_band,)).fetchall()


def _demo() -> None:
    conn = connect()
    try:
        print("== worst routes ==")
        for row in worst_routes(conn, limit=5):
            print(dict(row))
        print("\n== compliance by IMD decile ==")
        for row in compliance_by_imd_decile(conn):
            print(dict(row))
        print("\n== AM-peak compliance by day ==")
        for row in compliance_for_band(conn, "am_peak"):
            print(dict(row))
    finally:
        conn.close()


if __name__ == "__main__":
    _demo()
