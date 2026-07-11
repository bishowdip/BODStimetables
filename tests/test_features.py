"""Leakage checks on the written feature table.

These run against the real Parquet outputs when they exist (integration-style)
and are skipped otherwise, so `pytest` stays green on a fresh clone.
"""
import glob

import pandas as pd
import pytest

from src.common import load_config, project_path

FEATURES = glob.glob(str(project_path("data/parquet/ml_features/*.parquet")))
RBD = glob.glob(str(project_path("data/parquet/fact_route_band_day/*/*.parquet")))

pytestmark = pytest.mark.skipif(
    not (FEATURES and RBD), reason="feature table not built on this machine"
)


@pytest.fixture(scope="module")
def tables():
    feats = pd.concat(pd.read_parquet(f) for f in FEATURES)
    rbd = pd.concat(
        pd.read_parquet(f).assign(service_date=f.split("service_date=")[1].split("/")[0])
        for f in RBD
    )
    # Spark reads the partition column as a date type; compare as ISO strings
    feats["service_date"] = feats.service_date.astype(str)
    rbd["service_date"] = rbd.service_date.astype(str)
    return feats, rbd


def test_split_is_time_ordered(tables):
    feats, _ = tables
    train_dates = set(feats.loc[feats.is_train == 1, "service_date"])
    test_dates = set(feats.loc[feats.is_train == 0, "service_date"])
    assert train_dates and test_dates
    # every training day strictly precedes every test day
    assert max(train_dates) < min(test_dates)


def test_historical_rate_uses_train_days_only(tables):
    feats, rbd = tables
    train_dates = set(feats.loc[feats.is_train == 1, "service_date"])

    # recompute the historical rate independently, from train days only,
    # and compare with what the feature builder wrote
    expected = (
        rbd[rbd.service_date.isin(train_dates)]
        .groupby("route_id")["pct_on_time"].mean()
    )
    sample = (
        feats[feats.route_hist_ontime_rate > 0][["route_id", "route_hist_ontime_rate"]]
        .drop_duplicates("route_id").head(50)
    )
    joined = sample.join(expected.rename("expected"), on="route_id").dropna()
    assert len(joined) >= 10
    assert (joined.route_hist_ontime_rate - joined.expected).abs().max() < 1e-6


def test_label_not_among_model_features(tables):
    from src.ml.train_models import CATEGORICAL, LABEL, NUMERIC

    # the label and its direct precursors must never enter the feature vector
    assert LABEL not in NUMERIC + CATEGORICAL
    assert "pct_on_time" not in NUMERIC
    assert "mean_trip_delay" not in NUMERIC
    assert "sd_trip_delay" not in NUMERIC
