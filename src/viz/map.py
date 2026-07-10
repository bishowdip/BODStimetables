"""Reliability map (folium): each stop coloured by its measured on-time rate.

Stop-level delays come straight from the matched trip_stop_delay table, so the
map shows where the network is actually late, not a route average smeared over
stops. Stops with too few observations are dropped rather than shown with a
noisy colour. Output: docs/figures/reliability_map.html (standalone).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.common import load_config, project_path

log = logging.getLogger("map")
FIGS = project_path("docs", "figures")

MIN_EVENTS = 20  # a stop needs this many matched passes for a stable rate


def _stop_reliability(cfg) -> pd.DataFrame:
    pq = project_path(cfg["paths"]["parquet"])
    lo = cfg["reliability"]["ontime_lower_min"]
    hi = cfg["reliability"]["ontime_upper_min"]

    events = pd.read_parquet(pq / "trip_stop_delay", columns=["stop_id", "delay_min"])
    events["on_time"] = events.delay_min.between(lo, hi)
    per_stop = (
        events.groupby("stop_id")
        .agg(on_time=("on_time", "mean"), n_events=("on_time", "size"))
        .query("n_events >= @MIN_EVENTS")
        .reset_index()
    )

    stops = pd.read_parquet(pq / "gtfs" / "stops")[["stop_id", "stop_name", "stop_lat", "stop_lon"]]
    imd = pd.read_parquet(pq / "stop_imd" / "stop_imd.parquet")[["stop_id", "imd_decile"]]
    df = per_stop.merge(stops, on="stop_id").merge(imd, on="stop_id", how="left")
    log.info("mappable stops: %d (>= %d matched passes each)", len(df), MIN_EVENTS)
    return df


def _colour(on_time: float) -> str:
    if on_time >= 0.85:
        return "green"
    if on_time >= 0.60:
        return "orange"
    return "red"


def main() -> None:
    import folium

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    FIGS.mkdir(parents=True, exist_ok=True)

    df = _stop_reliability(cfg)
    if df.empty:
        log.warning("no stops to map -- has trip matching run?")
        return

    bbox = cfg["region"]["bbox"]
    centre = [(bbox["min_lat"] + bbox["max_lat"]) / 2, (bbox["min_lon"] + bbox["max_lon"]) / 2]
    m = folium.Map(location=centre, zoom_start=11, tiles="cartodbpositron")

    for r in df.itertuples():
        folium.CircleMarker(
            location=[r.stop_lat, r.stop_lon],
            radius=3,
            color=_colour(r.on_time),
            fill=True,
            fill_opacity=0.7,
            popup=(f"{r.stop_name}<br>on-time {r.on_time:.0%} "
                   f"({r.n_events} passes)<br>IMD decile {r.imd_decile}"),
        ).add_to(m)

    out = FIGS / "reliability_map.html"
    m.save(str(out))
    log.info("wrote %s (%d stops)", out, len(df))


if __name__ == "__main__":
    main()
