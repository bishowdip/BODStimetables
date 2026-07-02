"""Snapshot the BODS disruptions feed (SIRI-SX) and parse it to Parquet.

Third BODS catalogue after timetables and vehicle locations. Each fetch saves
the raw XML (reproducibility) and appends parsed situations to Parquet. A
situation carries a validity window, a reason, and the operators / lines / stops
it affects; we keep the affected stop ATCO codes, which are the same identifiers
as GTFS stop_ids, so disruptions join back to routes through stop_times.

West Yorkshire filter: ATCO codes for WY start with "450". A situation is kept
if any affected stop is in WY, or if it names no stops at all (kept, flagged
wy_specific=False) so regional/blanket disruptions are not lost.

One-off snapshot:            python -m src.ingest.fetch_disruptions
Repeat during study window:  python -m src.ingest.fetch_disruptions --every-hours 6 --for-hours 168
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.common import load_config, project_path

log = logging.getLogger("fetch_disruptions")
NS = {"s": "http://www.siri.org.uk/siri"}
WY_ATCO_PREFIX = "450"


def _text(el, path: str) -> str | None:
    node = el.find(path, NS)
    return node.text.strip() if node is not None and node.text else None


def parse_situations(xml_bytes: bytes, fetched_at: str) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows = []
    for sit in root.findall(".//s:PtSituationElement", NS):
        stops = [n.text for n in sit.findall(".//s:AffectedStopPoint/s:StopPointRef", NS) if n.text]
        lines = [n.text for n in sit.findall(".//s:AffectedLine/s:LineRef", NS) if n.text]
        ops = [n.text for n in sit.findall(".//s:AffectedOperator/s:OperatorRef", NS) if n.text]
        wy_stops = [s for s in stops if s.startswith(WY_ATCO_PREFIX)]
        if stops and not wy_stops:
            continue  # names stops, none in WY -> not our region
        rows.append(
            {
                "situation_id": _text(sit, "s:SituationNumber"),
                "participant": _text(sit, "s:ParticipantRef"),
                "progress": _text(sit, "s:Progress"),
                "reason": _text(sit, "s:MiscellaneousReason") or _text(sit, "s:EquipmentReason"),
                "planned": _text(sit, "s:Planned") == "true",
                "summary": _text(sit, "s:Summary"),
                "validity_start": _text(sit, "s:ValidityPeriod/s:StartTime"),
                "validity_end": _text(sit, "s:ValidityPeriod/s:EndTime"),
                "n_affected_stops": len(stops),
                "wy_stop_refs": ",".join(wy_stops),
                "line_refs": ",".join(lines),
                "operator_refs": ",".join(ops),
                "wy_specific": bool(wy_stops),
                "fetched_at": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def fetch_once(cfg, session) -> int:
    key = os.environ.get(cfg["sources"]["bods"]["api_key_env"])
    if not key:
        raise SystemExit("set BODS_API_KEY before fetching disruptions")
    r = session.get(
        "https://data.bus-data.dft.gov.uk/api/v1/siri-sx/",
        params={"api_key": key}, timeout=60,
    )
    r.raise_for_status()

    now = datetime.now()
    raw_dir = project_path(cfg["paths"]["raw"]) / "disruptions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{int(now.timestamp())}.xml").write_bytes(r.content)

    df = parse_situations(r.content, now.isoformat(timespec="seconds"))
    out_dir = project_path(cfg["paths"]["parquet"]) / "disruptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"snapshot_{int(now.timestamp())}.parquet", index=False)
    log.info("saved %d situations (%d WY-specific)", len(df), int(df["wy_specific"].sum()) if len(df) else 0)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--every-hours", type=float, help="repeat at this interval")
    ap.add_argument("--for-hours", type=float, default=168.0, help="total duration when repeating")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    session = requests.Session()
    session.headers.update({"User-Agent": "st5011cem-research-project"})

    if not args.every_hours:
        fetch_once(cfg, session)
        return

    deadline = time.monotonic() + args.for_hours * 3600
    while time.monotonic() < deadline:
        try:
            fetch_once(cfg, session)
        except requests.RequestException as exc:
            log.warning("fetch failed: %s", exc)
        time.sleep(args.every_hours * 3600)


if __name__ == "__main__":
    main()
