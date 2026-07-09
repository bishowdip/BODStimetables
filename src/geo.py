"""Pure-Python geo / time helpers.

These mirror the Spark column expressions used in trip matching, kept as plain
functions so the matching logic can be unit-tested without spinning up Spark.
"""
from __future__ import annotations

import math

EARTH_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return EARTH_M * 2 * math.asin(math.sqrt(a))


def gtfs_time_to_seconds(value: str) -> int:
    """Parse a GTFS HH:MM:SS time to seconds, allowing hours >= 24."""
    h, m, s = (int(x) for x in value.split(":"))
    return h * 3600 + m * 60 + s


def time_band(hour: int, bands: dict[str, tuple[int, int]]) -> str | None:
    """Return the name of the band [lo, hi) containing `hour`, or None."""
    for name, (lo, hi) in bands.items():
        if lo <= hour < hi:
            return name
    return None


DAY_S = 86_400


def wrap_delay_s(inferred_s: float, sched_s: float) -> float:
    """Delay in seconds, wrapped to the nearest day boundary.

    GTFS schedules run past midnight (25:10:00 is valid), while an observed
    clock time restarts at 00:00. Comparing them naively makes a bus that is
    5 min late on a 24:05 arrival look ~24 h early. Wrapping the difference to
    the interval (-12 h, +12 h] recovers the real delay; no genuine bus delay
    approaches 12 h, so the wrap is unambiguous.
    """
    delay = inferred_s - sched_s
    while delay <= -DAY_S / 2:
        delay += DAY_S
    while delay > DAY_S / 2:
        delay -= DAY_S
    return delay
