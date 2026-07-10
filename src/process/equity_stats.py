"""The equity question, tested: does non-compliance track deprivation?

Route-level analysis over the study week. Each route carries the median IMD
decile of the stops it serves (equity_join) and a compliance rate across its
route-band-days (warehouse). Two outputs:

  * compliance by IMD decile (band-day weighted) -- the chart the report leads
    with;
  * Spearman rank correlation between a route's IMD decile and its compliance
    rate, with a permutation p-value (10,000 shuffles) so the claim is a tested
    association, not a bar chart eyeballed into a finding.

Claims stay at LSOA/route level -- no inference about individuals (ecological
fallacy noted in the report).

Output: docs/results/equity_stats.json
"""
from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np
import pandas as pd

from src.common import load_config, project_path

log = logging.getLogger("equity_stats")

N_PERMUTATIONS = 10_000
MIN_BAND_DAYS = 5  # routes with fewer observations carry no stable rate


def route_table(conn) -> pd.DataFrame:
    sql = """
        SELECT r.route_id,
               r.route_imd_decile          AS imd_decile,
               AVG(f.compliant)            AS compliance_rate,
               AVG(f.pct_on_time)          AS mean_on_time,
               COUNT(*)                    AS band_days
        FROM fact_route_band_day f
        JOIN dim_route r ON r.route_id = f.route_id
        WHERE r.route_imd_decile IS NOT NULL
        GROUP BY r.route_id
        HAVING band_days >= ?
    """
    return pd.read_sql_query(sql, conn, params=(MIN_BAND_DAYS,))


def decile_table(conn) -> pd.DataFrame:
    sql = """
        SELECT CAST(r.route_imd_decile AS INT) AS imd_decile,
               AVG(f.compliant)                AS compliance_rate,
               AVG(f.pct_on_time)              AS mean_on_time,
               COUNT(*)                        AS band_days
        FROM fact_route_band_day f
        JOIN dim_route r ON r.route_id = f.route_id
        WHERE r.route_imd_decile IS NOT NULL
        GROUP BY imd_decile ORDER BY imd_decile
    """
    return pd.read_sql_query(sql, conn)


def spearman_with_permutation(x: pd.Series, y: pd.Series, rng: np.random.Generator) -> dict:
    """Spearman rho plus a two-sided permutation p-value.

    Spearman is Pearson on ranks, computed directly with numpy so we do not
    pull in scipy for one statistic.
    """
    xr = x.rank().to_numpy()
    yr = y.rank().to_numpy()
    observed = np.corrcoef(xr, yr)[0, 1]
    rho = observed
    hits = 0
    for _ in range(N_PERMUTATIONS):
        perm = np.corrcoef(rng.permutation(xr), yr)[0, 1]
        if abs(perm) >= abs(observed):
            hits += 1
    return {"spearman_rho": round(float(rho), 4),
            "p_value_permutation": round(hits / N_PERMUTATIONS, 5),
            "n_routes": int(len(x))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    conn = sqlite3.connect(project_path(cfg["paths"]["db"]))
    try:
        routes = route_table(conn)
        deciles = decile_table(conn)
    finally:
        conn.close()

    rng = np.random.default_rng(11)
    stats_compliance = spearman_with_permutation(routes.imd_decile, routes.compliance_rate, rng)
    stats_on_time = spearman_with_permutation(routes.imd_decile, routes.mean_on_time, rng)

    most = deciles[deciles.imd_decile <= 3]
    least = deciles[deciles.imd_decile >= 8]
    gap = {
        "deciles_1_3_compliance": round(float(np.average(most.compliance_rate, weights=most.band_days)), 4),
        "deciles_8_10_compliance": round(float(np.average(least.compliance_rate, weights=least.band_days)), 4),
    }
    gap["gap_percentage_points"] = round(100 * (gap["deciles_8_10_compliance"] - gap["deciles_1_3_compliance"]), 2)

    out = {
        "min_band_days_per_route": MIN_BAND_DAYS,
        "correlation_imd_vs_compliance": stats_compliance,
        "correlation_imd_vs_on_time": stats_on_time,
        "deprivation_gap": gap,
        "by_decile": deciles.round(4).to_dict(orient="records"),
    }
    results = project_path("docs", "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "equity_stats.json").write_text(json.dumps(out, indent=2))

    log.info("IMD decile vs compliance: rho=%.3f (p=%.5f, n=%d routes)",
             stats_compliance["spearman_rho"], stats_compliance["p_value_permutation"],
             stats_compliance["n_routes"])
    log.info("deciles 1-3 compliance %.1f%% vs deciles 8-10 %.1f%% (gap %.1f pp)",
             100 * gap["deciles_1_3_compliance"], 100 * gap["deciles_8_10_compliance"],
             gap["gap_percentage_points"])
    for r in out["by_decile"]:
        log.info("  decile %2d: compliance %.1f%%  on-time %.1f%%  (n=%d)",
                 r["imd_decile"], 100 * r["compliance_rate"], 100 * r["mean_on_time"], r["band_days"])


if __name__ == "__main__":
    main()
