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


def _ensure_compatible_java() -> None:
    """Point JAVA_HOME at a Spark-supported JDK (8/11/17/21) if the default is newer.

    Spark 4 doesn't run on Java 24+ (the Security Manager was removed and Hadoop's
    UserGroupInformation breaks). On macOS we can ask java_home for a 17 build; if
    JAVA_HOME is already a supported version we leave it alone.
    """
    import re
    import subprocess

    def major(java_home: str) -> int | None:
        try:
            out = subprocess.run(
                [str(Path(java_home) / "bin" / "java"), "-version"],
                capture_output=True, text=True,
            ).stderr
            m = re.search(r'version "(\d+)', out)
            return int(m.group(1)) if m else None
        except OSError:
            return None

    current = os.environ.get("JAVA_HOME")
    if current and (major(current) or 99) <= 21:
        return

    for ver in ("17", "21", "11"):
        try:
            home = subprocess.run(
                ["/usr/libexec/java_home", "-v", ver],
                capture_output=True, text=True,
            ).stdout.strip()
        except OSError:
            break
        if home:
            os.environ["JAVA_HOME"] = home
            return


def get_spark(app_suffix: str = ""):
    """Build (or fetch) the SparkSession used across the pipeline.

    The settings here are the ones cited in the report's optimisation section:
    a fixed shuffle-partition count and Arrow enabled for the small
    Spark -> pandas conversions at the visualisation stage.
    """
    _ensure_compatible_java()

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
