"""Export a portable SQL dump of the warehouse (required submission artefact).

Uses sqlite3's iterdump(), so the output is plain CREATE/INSERT statements that
rebuild the whole database on any SQLite. Run after load_db.py:

    python -m src.db.dump            # -> data/reliability_dump.sql
"""
from __future__ import annotations

import argparse
import logging
import sqlite3

from src.common import load_config, project_path

log = logging.getLogger("dump")


def export(db_path, out_path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        lines = list(conn.iterdump())
    finally:
        conn.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/reliability_dump.sql")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    db_path = project_path(cfg["paths"]["db"])
    if not db_path.exists():
        raise SystemExit(f"no database at {db_path} -- run load_db first")
    out = project_path(args.out)
    n = export(db_path, out)
    log.info("wrote %s (%d statements)", out, n)


if __name__ == "__main__":
    main()
