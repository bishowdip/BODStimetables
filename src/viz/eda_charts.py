"""Exploratory data analysis charts on the real study week.

Reads the processed Parquet tables and the warehouse, and writes a set of EDA
figures to docs/figures/eda_*.png. Everything here works on aggregates small
enough for pandas; the heavy work happened upstream in Spark. Run after the
processing and database stages.

    python -m src.viz.eda_charts
"""
from __future__ import annotations

import glob
import logging
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, project_path

log = logging.getLogger("eda_charts")

BLUE, ORANGE, GRAY, GREEN, RED, INK = "#4269d0", "#efb118", "#9498a0", "#3ca951", "#ff725c", "#2b2b33"
FIGS = project_path("docs", "figures")
BAND_ORDER = ["am_peak", "inter_peak", "pm_peak", "evening"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)
    log.info("wrote %s", name)


def _bare(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _trip_delay() -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in glob.glob(
        str(project_path("data/parquet/fact_trip_delay/*/*.parquet"))))


def _stop_delay(cols) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f, columns=cols) for f in glob.glob(
        str(project_path("data/parquet/trip_stop_delay/*/*.parquet"))))


def on_time_by_hour(trip: pd.DataFrame) -> None:
    g = trip.groupby("hour")["on_time_2_2"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(g.index, g.values, color=BLUE, marker="o", linewidth=2)
    ax.axvspan(7, 9, color=ORANGE, alpha=0.12)     # AM peak
    ax.axvspan(16, 19, color=ORANGE, alpha=0.12)   # PM peak
    ax.set_xlabel("hour of day (shaded = peak)")
    ax.set_ylabel("trips on time (%)")
    ax.set_title("Punctuality is worst through the afternoon", loc="left", color=INK)
    ax.set_xticks(range(0, 24, 2))
    _bare(ax)
    _save(fig, "eda_ontime_by_hour.png")


def on_time_by_day(trip: pd.DataFrame) -> None:
    g = (trip.groupby("day_of_week")["on_time_2_2"].mean() * 100).reindex(DAY_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(g)), g.values, color=BLUE, width=0.62)
    ax.set_xticks(range(len(g)), [d[:3] for d in g.index])
    ax.set_ylabel("trips on time (%)")
    ax.set_title("On-time rate by day of week", loc="left", color=INK)
    for b, v in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    _bare(ax)
    _save(fig, "eda_ontime_by_day.png")


def on_time_by_band(trip: pd.DataFrame) -> None:
    g = (trip.groupby("time_band")["on_time_2_2"].mean() * 100).reindex(BAND_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(range(len(g)), g.values, color=BLUE, width=0.55)
    ax.set_xticks(range(len(g)), [b.replace("_", " ") for b in g.index])
    ax.set_ylabel("trips on time (%)")
    ax.set_title("On-time rate by time band", loc="left", color=INK)
    for b, v in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    _bare(ax)
    _save(fig, "eda_ontime_by_band.png")


def delay_by_stop_sequence() -> None:
    df = _stop_delay(["stop_sequence", "delay_min"])
    df = df[df.stop_sequence <= 40]
    g = df.groupby("stop_sequence")["delay_min"].median()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(g.index, g.values, color=BLUE, linewidth=2)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("stop number along the trip")
    ax.set_ylabel("median delay (min)")
    ax.set_title("Delay builds up along a trip", loc="left", color=INK)
    _bare(ax)
    _save(fig, "eda_delay_by_stop_sequence.png")


def match_distance() -> None:
    df = _stop_delay(["dist_m"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df.dist_m, bins=50, color=BLUE)
    ax.set_xlabel("distance from nearest ping to stop (m)")
    ax.set_ylabel("matched stop events")
    ax.set_title("Match quality: most pings land within a few metres of the stop",
                 loc="left", color=INK)
    _bare(ax)
    _save(fig, "eda_match_distance.png")


def top_routes(conn) -> None:
    q = """
        SELECT r.route_short_name AS route, COUNT(*) AS trips,
               AVG(f.on_time) AS on_time
        FROM fact_trip_delay f JOIN dim_route r USING (route_id)
        GROUP BY f.route_id ORDER BY trips DESC LIMIT 10
    """
    df = pd.read_sql_query(q, conn).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df.route.astype(str), df.trips, color=BLUE)
    ax.set_xlabel("trips observed in the week")
    ax.set_title("Ten busiest routes by observed trips", loc="left", color=INK)
    for b, v in zip(bars, df.trips):
        ax.text(v + df.trips.max() * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=8, color=INK)
    _bare(ax)
    _save(fig, "eda_top_routes.png")


def class_balance() -> None:
    df = pd.concat(pd.read_parquet(f) for f in glob.glob(
        str(project_path("data/parquet/fact_route_band_day/*/*.parquet"))))
    counts = df.compliant.value_counts().reindex([0, 1]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(["non-compliant", "compliant"], counts.values, color=[RED, GREEN], width=0.55)
    ax.set_ylabel("route-band-days")
    ax.set_title("The target is imbalanced: most band-days fail the 85% bar",
                 loc="left", color=INK)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{int(v):,}\n{v/counts.sum():.0%}",
                ha="center", fontsize=9, color=INK)
    _bare(ax)
    _save(fig, "eda_class_balance.png")


def weather_vs_ontime() -> None:
    feats = pd.concat(pd.read_parquet(f) for f in glob.glob(
        str(project_path("data/parquet/ml_features/*.parquet"))))
    feats["rain"] = pd.cut(feats.precip_mm, [-0.1, 0.0, 0.5, 2, 100],
                           labels=["dry", "drizzle", "light", "wet"])
    g = feats.groupby("rain", observed=True)["pct_on_time"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(range(len(g)), g.values, color=BLUE, width=0.55)
    ax.set_xticks(range(len(g)), list(g.index))
    ax.set_ylabel("mean on-time rate (%)")
    ax.set_xlabel("precipitation in the band")
    ax.set_title("On-time rate against rainfall", loc="left", color=INK)
    for b, v in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    _bare(ax)
    _save(fig, "eda_weather_vs_ontime.png")


def correlation_heatmap() -> None:
    feats = pd.concat(pd.read_parquet(f) for f in glob.glob(
        str(project_path("data/parquet/ml_features/*.parquet"))))
    cols = ["pct_on_time", "n_trips", "n_stops", "sched_headway_mean_min",
            "route_imd_decile", "route_hist_ontime_rate", "precip_mm",
            "n_active_disruptions", "compliant"]
    corr = feats[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else INK, fontsize=7)
    ax.set_title("Correlation between features and the target", loc="left", color=INK)
    fig.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, "eda_correlation_heatmap.png")


def delay_boxplot_by_band(trip: pd.DataFrame) -> None:
    data = [trip.loc[trip.time_band == b, "median_delay"].clip(-10, 20).dropna()
            for b in BAND_ORDER if (trip.time_band == b).any()]
    labels = [b.replace("_", " ") for b in BAND_ORDER if (trip.time_band == b).any()]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.6)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_ylabel("trip median delay (min)")
    ax.set_title("Spread of trip delay by time band", loc="left", color=INK)
    _bare(ax)
    _save(fig, "eda_delay_boxplot_by_band.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_config()
    FIGS.mkdir(parents=True, exist_ok=True)
    trip = _trip_delay()

    on_time_by_hour(trip)
    on_time_by_day(trip)
    on_time_by_band(trip)
    delay_boxplot_by_band(trip)
    delay_by_stop_sequence()
    match_distance()
    class_balance()
    weather_vs_ontime()
    correlation_heatmap()

    conn = sqlite3.connect(project_path(load_config()["paths"]["db"]))
    try:
        top_routes(conn)
    finally:
        conn.close()
    log.info("EDA charts written to docs/figures/eda_*.png")


if __name__ == "__main__":
    main()
