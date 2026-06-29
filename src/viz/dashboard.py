"""Interactive reliability dashboard (Streamlit).

A thin read-only view over the warehouse and the saved model metrics -- it runs
no Spark, only the small aggregates, which is the same discipline as the static
plots. Build the DB and run the ML stage first, then:

    streamlit run src/viz/dashboard.py
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from src.common import load_config, project_path

cfg = load_config()
DB = project_path(cfg["paths"]["db"])
RESULTS = project_path("docs", "results")
FIGS = project_path("docs", "figures")


@st.cache_data
def query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def main() -> None:
    st.set_page_config(page_title="WY Bus Reliability", layout="wide")
    st.title("Who Gets the Worst Bus?")
    st.caption("West Yorkshire bus service reliability, compliance and equity")

    if not DB.exists():
        st.error("No warehouse found. Run the pipeline (see README) then reload.")
        return

    overview = query(
        """
        SELECT COUNT(DISTINCT route_id) AS routes,
               COUNT(*)                 AS band_days,
               AVG(compliant)           AS compliance_rate,
               AVG(pct_on_time)         AS mean_on_time
        FROM fact_route_band_day
        """
    ).iloc[0]
    trips = query("SELECT COUNT(*) AS n, AVG(on_time) AS on_time FROM fact_trip_delay").iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routes", int(overview.routes))
    c2.metric("Route-band-days", int(overview.band_days))
    c3.metric("Compliant (>=85% on time)", f"{overview.compliance_rate:.1%}")
    c4.metric("Trips observed", int(trips.n))

    st.subheader("Compliance by deprivation decile")
    eq = query(
        """
        SELECT CAST(r.route_imd_decile AS INT) AS imd_decile,
               AVG(f.compliant) AS compliance_rate
        FROM fact_route_band_day f
        JOIN dim_route r ON r.route_id = f.route_id
        WHERE r.route_imd_decile IS NOT NULL
        GROUP BY imd_decile ORDER BY imd_decile
        """
    )
    if not eq.empty:
        st.bar_chart(eq.set_index("imd_decile")["compliance_rate"])
        st.caption("1 = most deprived. A rising line means worse service in poorer areas.")

    left, right = st.columns(2)
    with left:
        st.subheader("Worst routes")
        worst = query(
            """
            SELECT r.route_short_name AS route, AVG(f.compliant) AS compliance, COUNT(*) AS band_days
            FROM fact_route_band_day f JOIN dim_route r ON r.route_id = f.route_id
            GROUP BY f.route_id HAVING band_days >= 3
            ORDER BY compliance ASC LIMIT 10
            """
        )
        st.dataframe(worst, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Compliance by time band")
        band = query(
            "SELECT time_band, AVG(pct_on_time) AS on_time FROM fact_route_band_day "
            "GROUP BY time_band ORDER BY on_time"
        )
        if not band.empty:
            st.bar_chart(band.set_index("time_band")["on_time"])

    st.subheader("Model comparison")
    metrics_path = RESULTS / "metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        st.dataframe(m, use_container_width=True, hide_index=True)
    else:
        st.info("Run the ML stage to populate model metrics.")

    map_path = FIGS / "reliability_map.html"
    if map_path.exists():
        st.subheader("Reliability map")
        st.components.v1.html(map_path.read_text(), height=500)


if __name__ == "__main__":
    main()
