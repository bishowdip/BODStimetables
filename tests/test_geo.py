import math

from src.geo import gtfs_time_to_seconds, haversine_m, time_band

BANDS = {"am_peak": (7, 9), "inter_peak": (9, 16), "pm_peak": (16, 19), "evening": (19, 23)}


def test_haversine_zero_distance():
    assert haversine_m(53.8, -1.55, 53.8, -1.55) == 0.0


def test_haversine_known_distance():
    # ~1 degree of latitude is ~111 km
    d = haversine_m(53.0, -1.55, 54.0, -1.55)
    assert math.isclose(d, 111_195, rel_tol=0.01)


def test_haversine_small_offset_under_match_radius():
    # ~15 m north should be well inside a 50 m match radius
    d = haversine_m(53.8, -1.55, 53.8 + 15 / 111_320, -1.55)
    assert d < 50


def test_gtfs_time_after_midnight():
    assert gtfs_time_to_seconds("25:30:00") == 25 * 3600 + 30 * 60


def test_gtfs_time_basic():
    assert gtfs_time_to_seconds("08:15:30") == 8 * 3600 + 15 * 60 + 30


def test_time_band_assignment():
    assert time_band(8, BANDS) == "am_peak"
    assert time_band(13, BANDS) == "inter_peak"
    assert time_band(17, BANDS) == "pm_peak"
    assert time_band(21, BANDS) == "evening"
    assert time_band(3, BANDS) is None  # outside all bands
