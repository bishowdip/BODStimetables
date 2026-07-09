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


def test_wrap_delay_plain_case():
    from src.geo import wrap_delay_s

    # bus observed 3 min after schedule, same day: no wrap needed
    assert wrap_delay_s(10 * 3600 + 180, 10 * 3600) == 180


def test_wrap_delay_after_midnight_schedule():
    from src.geo import wrap_delay_s

    # scheduled 24:05 (86700s), observed 00:10 next clock day (600s):
    # naive diff is -86100s; the real delay is +5 min
    assert wrap_delay_s(600, 86_700) == 300


def test_wrap_delay_early_before_midnight():
    from src.geo import wrap_delay_s

    # scheduled 00:02 (i.e. 24:02 as 120s next day), bus passes 23:58 -> 4 min early
    assert wrap_delay_s(86_280, 120) == -240
