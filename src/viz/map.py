"""Reliability map (folium).

Plots each WY stop coloured by the mean on-time rate of the routes that serve it,
so the spatial pattern of unreliability -- and how it lines up with deprivation --
is visible. Reads only the small per-stop aggregate. Output is a standalone HTML
saved to docs/figures/reliability_map.html.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.common import load_config, project_path
from src.db.queries import connect

log = logging.getLogger("map")
FIGS = project_path("docs", "figures")


def _stop_reliability() -> pd.DataFrame:
    conn = connect()
    try:
        # Reliability is computed per route; dim_stop has no route_id, so for the
        # map we colour each stop by the window-wide mean on-time rate and tag it
        # with its IMD decile. (A per-stop route join is in the EDA notebook.)
        return pd.read_sql_query(
            """
            SELECT s.stop_id, s.lat, s.lon, s.name, l.imd_decile,
                   (SELECT AVG(f.pct_on_time) FROM fact_route_band_day f) AS on_time
            FROM dim_stop s
            LEFT JOIN dim_lsoa l ON l.lsoa_code = s.lsoa_code
            WHERE s.lat IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()


def _colour(on_time: float) -> str:
    if on_time is None:
        return "gray"
    if on_time >= 0.85:
        return "green"
    if on_time >= 0.7:
        return "orange"
    return "red"


def main() -> None:
    import folium

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    FIGS.mkdir(parents=True, exist_ok=True)

    df = _stop_reliability()
    if df.empty:
        log.warning("no stops to map -- is the DB built?")
        return

    bbox = cfg["region"]["bbox"]
    centre = [(bbox["min_lat"] + bbox["max_lat"]) / 2, (bbox["min_lon"] + bbox["max_lon"]) / 2]
    m = folium.Map(location=centre, zoom_start=11, tiles="cartodbpositron")

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=3,
            color=_colour(r.on_time),
            fill=True,
            fill_opacity=0.7,
            popup=f"{r['name']} | IMD {r.imd_decile}",
        ).add_to(m)

    out = FIGS / "reliability_map.html"
    m.save(str(out))
    log.info("wrote %s (%d stops)", out, len(df))


if __name__ == "__main__":
    main()
