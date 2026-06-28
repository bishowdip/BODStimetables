"""Shared helpers: config loading and a single place to build the SparkSession.

Keeping the Spark config in one function means the optimisation settings
(shuffle partitions, etc.) are consistent across every stage and easy to point
at in the report.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    """Load settings.yaml, then overlay settings.local.yaml if it exists.

    The local file is git-ignored, so machine-specific paths or optional keys
    stay off GitHub.
    """
    cfg_path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    local = cfg_path.with_name("settings.local.yaml")
    if local.exists():
        with open(local) as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    return cfg


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the repo root."""
    return ROOT.joinpath(*parts)


def get_spark(app_suffix: str = ""):
    """Build (or fetch) the SparkSession used across the pipeline.

    The settings here are the ones cited in the report's optimisation section:
    a fixed shuffle-partition count and Arrow enabled for the small
    Spark -> pandas conversions at the visualisation stage.
    """
    from pyspark.sql import SparkSession

    cfg = load_config()
    s = cfg["spark"]
    name = s["app_name"] + (f"-{app_suffix}" if app_suffix else "")

    builder = (
        SparkSession.builder.appName(name)
        .master(os.environ.get("SPARK_MASTER", s["master"]))
        .config("spark.sql.shuffle.partitions", s["shuffle_partitions"])
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "Europe/London")
    )
    return builder.getOrCreate()
