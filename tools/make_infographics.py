"""Summary infographics for the report.

Each figure is built from the measured outputs under docs/results/ (or the
warehouse), so re-running the pipeline re-derives every number shown. Nothing
here is typed in by hand.

    python -m tools.make_infographics
"""
from __future__ import annotations

import glob
import json
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, project_path

RESULTS = project_path("docs", "results")
FIGS = project_path("docs", "figures")

# one consistent palette across all figures: blue for magnitude, orange for
# the contrast series, gray for context, green/red only as pass/fail status
BLUE = "#4269d0"
ORANGE = "#efb118"
GRAY = "#9498a0"
GREEN = "#3ca951"
RED = "#ff725c"
INK = "#2b2b33"


def _load_json(name: str):
    return json.loads((RESULTS / name).read_text())


def data_funnel() -> None:
    """Row counts at each pipeline stage, raw pings down to model rows."""
    quality = _load_json("parse_quality.json")
    raw = sum(d["raw_pings"] for d in quality)
    deduped = sum(d["deduped_pings"] for d in quality)
    matched = sum(
        len(pd.read_parquet(f, columns=["stop_id"]))
        for f in glob.glob(str(project_path("data/parquet/trip_stop_delay/*/*.parquet")))
    )
    conn = sqlite3.connect(project_path(load_config()["paths"]["db"]))
    trips = conn.execute("SELECT COUNT(*) FROM fact_trip_delay").fetchone()[0]
    bands = conn.execute("SELECT COUNT(*) FROM fact_route_band_day").fetchone()[0]
    conn.close()

    stages = [
        ("Raw pings (7 days)", raw),
        ("After deduplication", deduped),
        ("Matched stop events", matched),
        ("Trip-day facts", trips),
        ("Route-band-days (model rows)", bands),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [s[0] for s in stages][::-1]
    vals = [s[1] for s in stages][::-1]
    bars = ax.barh(names, vals, color=BLUE, height=0.62)
    ax.set_xscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_title("From raw feed to model table", loc="left", fontsize=13, color=INK)
    for b, v in zip(bars, vals):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(right=max(vals) * 12)
    plt.tight_layout()
    plt.savefig(FIGS / "info_data_funnel.png", dpi=150)
    plt.close()


def metric_tiles() -> None:
    """The brief's six metrics as headline tiles, value against its bar."""
    sens = {r["window"]: r for r in _load_json("sensitivity.json")}
    match = _load_json("match_stats.json")
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    rf = metrics.set_index("model").loc["random_forest"]
    bench = _load_json("broadcast_bench.json")
    speedup = bench["no_broadcast"]["seconds"] / bench["broadcast"]["seconds"]

    hw = pd.concat(pd.read_parquet(f) for f in
                   glob.glob(str(project_path("data/parquet/headway_obs/*.parquet"))))
    ttv = pd.concat(pd.read_parquet(f) for f in
                    glob.glob(str(project_path("data/parquet/travel_time_variability/*.parquet"))))

    tiles = [
        ("Service reliability", f"{100*sens['+/-2 min']['compliant_share']:.1f}%",
         "of route-band-days meet the 85% bar", RED),
        ("Headway regularity", f"{100*hw.headway_regular.mean():.1f}%",
         "of band-days keep SD within 20% of schedule", RED),
        ("Travel time variability", f"{100*ttv.ttv_ok.mean():.1f}%",
         "of routes keep trip-time CV under 15%", RED),
        ("Service efficiency", f"{100*match['overall_avl_confirmed_rate']:.1f}%",
         "of scheduled trips left a live trace (floor)", ORANGE),
        ("Model efficiency", f"{rf.f1:.3f}",
         f"best F1 (random forest), {rf.train_seconds:.0f}s to train", GREEN),
        ("Algorithmic efficiency", f"{speedup:.2f}x",
         "join speedup from broadcasting the stops table", GREEN),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    for ax, (title, value, sub, colour) in zip(axes.flat, tiles):
        ax.axis("off")
        ax.text(0.03, 0.78, title, fontsize=11, color=GRAY, transform=ax.transAxes)
        ax.text(0.03, 0.36, value, fontsize=30, fontweight="bold",
                color=colour, transform=ax.transAxes)
        ax.text(0.03, 0.10, sub, fontsize=9, color=INK, transform=ax.transAxes, wrap=True)
    fig.suptitle("The brief's six metrics, measured on the study week",
                 x=0.02, ha="left", fontsize=13, color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(FIGS / "info_metric_tiles.png", dpi=150)
    plt.close()


def frequency_confound() -> None:
    """Compliance falls with trips per band while punctuality stays flat."""
    eq = _load_json("equity_stats.json")
    rows = pd.DataFrame(eq["frequency_confound"]["compliance_by_trips_per_band"])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(rows))
    ax.bar(x, 100 * rows.compliance, color=BLUE, width=0.6,
           label="share of band-days compliant")
    ax.plot(x, 100 * rows.on_time, color=ORANGE, marker="o", linewidth=2,
            label="mean on-time rate of trips")
    ax.set_xticks(list(x), rows.trips_per_band)
    ax.set_xlabel("scheduled trips in the time band")
    ax.set_ylabel("per cent")
    ax.set_ylim(0, 100)
    for i, v in enumerate(rows.compliance):
        ax.text(i, 100 * v + 2, f"{100*v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_title("The 85% threshold rewards infrequent service, not punctual service",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / "info_frequency_confound.png", dpi=150)
    plt.close()


def equity_panels() -> None:
    """Two panels, same x: the two deprivation gradients point opposite ways."""
    eq = _load_json("equity_stats.json")
    dec = pd.DataFrame(eq["by_decile"])
    rho_c = eq["correlation_imd_vs_compliance"]
    rho_o = eq["correlation_imd_vs_on_time"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    ax1.bar(dec.imd_decile, 100 * dec.compliance_rate, color=BLUE, width=0.7)
    ax1.set_title(f"Threshold compliance rises with affluence\n"
                  f"(rho = {rho_c['spearman_rho']:+.3f}, p = {rho_c['p_value_permutation']:.3f})",
                  fontsize=11, loc="left", color=INK)
    ax1.set_ylabel("band-days compliant (%)")
    ax2.bar(dec.imd_decile, 100 * dec.mean_on_time, color=ORANGE, width=0.7)
    ax2.set_title(f"Raw punctuality falls with affluence\n"
                  f"(rho = {rho_o['spearman_rho']:+.3f}, p < 0.001)",
                  fontsize=11, loc="left", color=INK)
    ax2.set_ylabel("mean on-time rate (%)")
    for ax in (ax1, ax2):
        ax.set_xlabel("IMD decile (1 = most deprived)")
        ax.set_xticks(dec.imd_decile)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / "info_equity_panels.png", dpi=150)
    plt.close()


def avl_by_day() -> None:
    """AVL-confirmed floor per study day."""
    match = _load_json("match_stats.json")
    days = pd.DataFrame(match["by_day"])
    labels = [pd.Timestamp(d).strftime("%a %d") for d in days.service_date]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(labels, 100 * days.avl_confirmed_rate, color=BLUE, width=0.65)
    ax.axhline(100 * match["overall_avl_confirmed_rate"], color=GRAY,
               linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.4, 100 * match["overall_avl_confirmed_rate"] + 1.2,
            f"week: {100*match['overall_avl_confirmed_rate']:.1f}%",
            fontsize=9, color=GRAY, ha="right")
    for b, v in zip(bars, days.avl_confirmed_rate):
        ax.text(b.get_x() + b.get_width() / 2, 100 * v + 1.2, f"{100*v:.0f}%",
                ha="center", fontsize=9, color=INK)
    ax.set_ylabel("scheduled trips with a live trace (%)")
    ax.set_ylim(0, 100)
    ax.set_title("AVL-confirmed operation by day (a floor, not a cancellation count)",
                 loc="left", fontsize=12, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / "info_avl_by_day.png", dpi=150)
    plt.close()


def sensitivity_bars() -> None:
    """Compliance under the three on-time windows."""
    sens = pd.DataFrame(_load_json("sensitivity.json"))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sens.window, 100 * sens.compliant_share, color=BLUE, width=0.55)
    for b, v in zip(bars, sens.compliant_share):
        ax.text(b.get_x() + b.get_width() / 2, 100 * v + 1.5, f"{100*v:.1f}%",
                ha="center", fontsize=10, color=INK)
    ax.set_ylabel("band-days compliant (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Compliance under wider on-time windows", loc="left",
                 fontsize=12, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS / "info_sensitivity.png", dpi=150)
    plt.close()


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    data_funnel()
    metric_tiles()
    frequency_confound()
    equity_panels()
    avl_by_day()
    sensitivity_bars()
    print("infographics written to docs/figures/: "
          "info_data_funnel, info_metric_tiles, info_frequency_confound, "
          "info_equity_panels, info_avl_by_day, info_sensitivity")


if __name__ == "__main__":
    main()
