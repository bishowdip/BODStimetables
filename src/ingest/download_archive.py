"""Rate-limited download of the archived BODS feeds for the study window.

Downloading is I/O-bound and capped by the archive's 1 request/second limit, so
Spark would add nothing here -- this stays plain sequential Python. The script
is resumable: anything already on disk is skipped, so an interrupted run can be
restarted without re-fetching.

Two feed kinds per service date:
  * timetables  -> daily GTFS zip (the scheduled truth)
  * gtfsrt      -> archived protobuf snapshots at ~60s cadence (the actual truth)

The exact archive path scheme depends on which NDL/BODS endpoint you have access
to, so the URL templates live in config (sources.*_url_template) and are filled
with the service date. Set them once, then run.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import requests

from src.common import load_config, project_path

log = logging.getLogger("download_archive")


class RateLimiter:
    """Spaces requests so we never exceed the archive's limit."""

    def __init__(self, per_second: float):
        self._min_gap = 1.0 / per_second if per_second > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last = time.monotonic()


def _download(url: str, dest: Path, limiter: RateLimiter, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("skip (exists): %s", dest.name)
        return True
    limiter.wait()
    try:
        with session.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            tmp.rename(dest)
        log.info("got %s (%d bytes)", dest.name, dest.stat().st_size)
        return True
    except requests.RequestException as exc:
        log.warning("failed %s: %s", url, exc)
        return False


def build_targets(cfg: dict, date: str) -> list[tuple[str, Path]]:
    """Return (url, destination) pairs for one service date."""
    src = cfg["sources"]
    raw = project_path(cfg["paths"]["raw"])
    targets: list[tuple[str, Path]] = []

    tt_tpl = src.get("timetable_url_template")
    if tt_tpl:
        targets.append((tt_tpl.format(date=date), raw / date / "timetable.zip"))

    rt_tpl = src.get("gtfsrt_url_template")
    if rt_tpl:
        # A single archived snapshot index per date; the real archive may expose
        # one file or a tar of 60s snapshots -- adjust the template accordingly.
        targets.append((rt_tpl.format(date=date), raw / date / "gtfsrt.dat"))

    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="*", help="override the dates in settings.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    dates = args.dates or cfg["window"]["dates"]

    limiter = RateLimiter(cfg["sources"].get("rate_limit_per_sec", 1))
    session = requests.Session()
    session.headers.update({"User-Agent": "st5011cem-research-project"})

    ok = fail = 0
    for date in dates:
        targets = build_targets(cfg, date)
        if not targets:
            log.error("no URL templates configured in sources.* -- nothing to fetch for %s", date)
            continue
        for url, dest in targets:
            if _download(url, dest, limiter, session):
                ok += 1
            else:
                fail += 1

    log.info("done: %d downloaded/cached, %d failed", ok, fail)


if __name__ == "__main__":
    main()
