"""Render architecture.png and schema.png for the report (matplotlib, no extra deps).

    python -m tools.make_diagrams
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.common import project_path

OUT = project_path("docs")


def _box(ax, x, y, w, h, title, lines, fc="#eef3fb", ec="#2b5d9e"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.4, edgecolor=ec, facecolor=fc)
    )
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
            fontsize=11, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.62 - i * 0.30, ln, ha="left", va="top", fontsize=8.5)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.3, color="#555"))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_title("System architecture: ingestion -> processing -> storage/ML -> visualisation",
                 fontsize=12, fontweight="bold", pad=10)

    _box(ax, 1, 10.6, 8, 2.0, "INGESTION  (Python collectors)",
         ["poll_live / fetch_disruptions  -> GTFS-RT + SIRI-SX over the window",
          "download_timetable / download_supplementary  -> GTFS, IMD, LSOA",
          "parse_gtfsrt / load_static / fetch_weather  -> WY-filtered Parquet"],
         fc="#fdf3e7", ec="#c8801f")
    _box(ax, 1, 7.6, 8, 2.3, "PROCESSING  (PySpark core)",
         ["trip_match  -> positions <-> stop_times (broadcast join)",
          "compute_reliability  -> on-time -> route x band x day",
          "equity_join  -> stop -> LSOA -> IMD (GeoPandas, small)"],
         fc="#eaf5ec", ec="#2f8f46")
    _box(ax, 0.6, 4.4, 4.2, 2.2, "STORAGE (SQLite)",
         ["dim_route / dim_stop / dim_lsoa",
          "fact_trip_delay",
          "fact_route_band_day  (ML input)"])
    _box(ax, 5.2, 4.4, 4.2, 2.2, "ML (PySpark MLlib)",
         ["features  (leakage-safe, time split)",
          "LR / RF / GBT + CrossValidator",
          "evaluate  -> F1, ROC-AUC, F1/sec"],
         fc="#f3eafa", ec="#7a3fb0")
    _box(ax, 1, 1.4, 8, 2.0, "VISUALISATION  (small aggregates -> pandas)",
         ["matplotlib: model comparison, ROC/PR, equity",
          "folium: reliability map   |   Streamlit dashboard"],
         fc="#eef3fb", ec="#2b5d9e")

    _arrow(ax, 5, 10.6, 5, 9.9)
    _arrow(ax, 5, 7.6, 2.7, 6.6)
    _arrow(ax, 5, 7.6, 7.3, 6.6)
    _arrow(ax, 2.7, 4.4, 5, 3.4)
    _arrow(ax, 7.3, 4.4, 5, 3.4)
    ax.text(5.15, 10.25, "Parquet, partitioned by service_date", fontsize=8, style="italic", color="#555")

    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def schema() -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Database schema (star schema, SQLite)", fontsize=12, fontweight="bold", pad=8)

    _box(ax, 0.4, 5.4, 3.0, 1.9, "dim_lsoa",
         ["lsoa_code  PK", "imd_decile", "imd_score"], fc="#eaf5ec", ec="#2f8f46")
    _box(ax, 0.4, 2.3, 3.0, 2.2, "dim_stop",
         ["stop_id  PK", "name, lat, lon", "lsoa_code  FK"])
    _box(ax, 4.6, 5.4, 3.2, 2.1, "dim_route",
         ["route_id  PK", "operator", "route_short_name", "route_imd_decile"])
    _box(ax, 8.4, 4.7, 3.2, 2.6, "fact_trip_delay",
         ["trip_id, service_date  PK", "route_id  FK", "time_band, day_of_week",
          "stops_observed, median_delay", "on_time"], fc="#f3eafa", ec="#7a3fb0")
    _box(ax, 8.4, 1.0, 3.2, 2.9, "fact_route_band_day  (ML)",
         ["route_id, service_date,", "  time_band  PK", "route_id  FK",
          "n_trips, pct_on_time", "sd_trip_delay", "compliant"], fc="#f3eafa", ec="#7a3fb0")

    _arrow(ax, 1.9, 5.4, 1.9, 4.5)              # lsoa -> stop
    _arrow(ax, 6.2, 5.4, 9.9, 7.3)              # route -> trip_delay (FK)
    _arrow(ax, 6.2, 5.4, 9.9, 3.9)              # route -> route_band_day (FK)
    ax.text(2.0, 4.95, "FK", fontsize=8, color="#555")

    fig.savefig(OUT / "schema.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    schema()
    print(f"wrote {OUT/'architecture.png'} and {OUT/'schema.png'}")


if __name__ == "__main__":
    main()
