"""Generate a small, internally consistent sample dataset.

This produces the *output of the ingestion stage* (the Parquet layout the
processing stage reads), so the whole pipeline from trip matching onwards can be
run without the multi-GB archive download. The positions are deliberately placed
near the scheduled stops at scheduled time + a route-specific delay, so trip
matching has something real to match and the resulting compliance mix is varied.

    python -m tools.make_sample                 # -> data/parquet (run the pipeline)
    python -m tools.make_sample --out data/sample --scale 0.3   # committed sample
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load_config

random.seed(7)
np.random.seed(7)

# A handful of stops inside the WY bbox (Leeds-ish coordinates).
BASE_LAT, BASE_LON = 53.80, -1.55


def _offset(lat, lon, dn_m, de_m):
    dlat = dn_m / 111_320.0
    dlon = de_m / (111_320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def build(out: Path, cfg: dict, scale: float) -> None:
    dates = cfg["window"]["dates"]
    n_routes = max(2, int(6 * scale) + 2)
    n_stops_per_route = max(4, int(8 * scale) + 3)

    # --- stops & routes & trips & stop_times ---
    stops, routes, trips, stop_times = [], [], [], []
    stop_imd_rows = []
    for r in range(n_routes):
        route_id = f"R{r:02d}"
        # route-level reliability tendency: lower routes run worse
        route_ontime_bias = 0.2 + 0.6 * (r / max(n_routes - 1, 1))
        routes.append(
            {"route_id": route_id, "agency_id": f"OP{r % 3}",
             "route_short_name": f"{10 + r}"}
        )
        route_stop_ids = []
        for s in range(n_stops_per_route):
            stop_id = f"{route_id}_S{s:02d}"
            lat, lon = _offset(BASE_LAT, BASE_LON, 400 * s + 800 * r, 300 * s - 500 * r)
            stops.append({"stop_id": stop_id, "stop_name": f"Stop {route_id}-{s}",
                          "stop_lat": lat, "stop_lon": lon})
            # deprivation: more deprived (low decile) on poorer-served routes
            decile = int(np.clip(round(route_ontime_bias * 10), 1, 10))
            stop_imd_rows.append(
                {"stop_id": stop_id, "lsoa_code": f"E0100{r:02d}{s:02d}",
                 "imd_decile": decile, "imd_score": float(40 - 3 * decile)}
            )
            route_stop_ids.append((stop_id, lat, lon))

        # several trips per day across the time bands
        for d_idx, date in enumerate(dates):
            for dep_hour in (8, 12, 17, 20):  # one per band
                trip_id = f"{route_id}_{date}_{dep_hour}"
                trips.append({"trip_id": trip_id, "route_id": route_id})
                t = dep_hour * 3600
                for seq, (stop_id, _, _) in enumerate(route_stop_ids):
                    t += 180  # ~3 min between stops
                    hh, rem = divmod(t, 3600)
                    mm, ss = divmod(rem, 60)
                    stop_times.append(
                        {"trip_id": trip_id, "stop_id": stop_id, "stop_sequence": seq,
                         "arrival_time": f"{hh:02d}:{mm:02d}:{ss:02d}"}
                    )

    # --- positions: pings near each stop at scheduled time + route delay ---
    positions = []
    st_df = pd.DataFrame(stop_times)
    stop_xy = {s["stop_id"]: (s["stop_lat"], s["stop_lon"]) for s in stops}
    for trip in trips:
        route_idx = int(trip["route_id"][1:])
        ontime_bias = 0.2 + 0.6 * (route_idx / max(n_routes - 1, 1))
        date = trip["trip_id"].split("_")[1]
        # ~10% of trips leave no AVL trace (the no-show / unconfirmed proxy)
        if random.random() > (0.5 + 0.5 * ontime_bias):
            continue
        rows = st_df[st_df.trip_id == trip["trip_id"]]
        for _, sr in rows.iterrows():
            lat, lon = stop_xy[sr.stop_id]
            hh, mm, ss = (int(x) for x in sr.arrival_time.split(":"))
            sched_sec = hh * 3600 + mm * 60 + ss
            # delay: better routes ~ on time, worse routes skew late
            delay_min = np.random.normal(loc=(1 - ontime_bias) * 6, scale=2.0)
            pass_sec = sched_sec + delay_min * 60
            day0 = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
            ts = day0 + timedelta(seconds=pass_sec)
            jlat, jlon = _offset(lat, lon, np.random.normal(0, 15), np.random.normal(0, 15))
            positions.append(
                {"vehicle_id": f"V{route_idx}", "trip_id": trip["trip_id"],
                 "route_id": trip["route_id"], "lat": jlat, "lon": jlon,
                 "bearing": None, "ts": int(ts.timestamp()), "ts_utc": ts,
                 "service_date": date}
            )

    # --- weather (hourly, 7 days) ---
    weather = []
    for date in dates:
        for hour in range(24):
            weather.append(
                {"ts_local": pd.Timestamp(f"{date} {hour:02d}:00"),
                 "precip_mm": max(0.0, np.random.normal(0.3, 0.6)),
                 "temp_c": float(np.random.normal(15, 4)),
                 "wind_kph": max(0.0, np.random.normal(14, 5)),
                 "service_date": date, "hour": hour}
            )

    imd = (
        pd.DataFrame(stop_imd_rows)[["lsoa_code", "imd_decile", "imd_score"]]
        .drop_duplicates("lsoa_code")
        .rename(columns={"lsoa_code": "LSOA code (2011)",
                         "imd_decile": "IMD Decile", "imd_score": "IMD Score"})
    )

    # --- write everything in the ingest-stage layout ---
    _write_positions(out, positions)
    g = out / "gtfs"
    _write(g / "stops", pd.DataFrame(stops))
    _write(g / "routes", pd.DataFrame(routes))
    _write(g / "trips", pd.DataFrame(trips))
    _write(g / "stop_times", st_df)
    _write(out / "weather", pd.DataFrame(weather))
    _write(out / "imd", imd, single="imd.parquet")
    _write(out / "stop_imd", pd.DataFrame(stop_imd_rows), single="stop_imd.parquet")

    print(f"sample written to {out}: {len(positions)} positions, "
          f"{len(trips)} trips, {len(stops)} stops, {n_routes} routes")


def _write(path: Path, df: pd.DataFrame, single: str | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / (single or "part-0.parquet"), index=False)


def _write_positions(out: Path, positions: list[dict]) -> None:
    df = pd.DataFrame(positions)
    for date, grp in df.groupby("service_date"):
        d = out / "positions" / f"service_date={date}"
        d.mkdir(parents=True, exist_ok=True)
        grp.drop(columns="service_date").to_parquet(d / "part-0.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/parquet")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    build(Path(args.out), load_config(), args.scale)


if __name__ == "__main__":
    main()
