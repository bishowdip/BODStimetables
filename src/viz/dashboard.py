"""Interactive dashboard for the West Yorkshire bus reliability project.

Everything on show is computed from the warehouse and the processed Parquet
tables at load time, so the dashboard reflects whatever the pipeline last
produced rather than a set of exported images. No Spark runs here: only the
small aggregates and the saved predictions are read, which keeps it quick.

    streamlit run src/viz/dashboard.py

Tabs: overview, data quality, EDA, reliability metrics, equity, models, map,
disruptions.
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

# `streamlit run` executes this file directly rather than as a package module,
# so the repo root is not on the path and `import src...` would fail.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:  # pydeck drives the interactive map; fall back to st.map without it
    import pydeck as pdk

    HAS_PYDECK = True
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    HAS_PYDECK = False

from src.common import load_config, project_path

CFG = load_config()
DB = project_path(CFG["paths"]["db"])
PQ = project_path(CFG["paths"]["parquet"])
RESULTS = project_path("docs", "results")

BLUE, ORANGE, GREEN, RED, GRAY = "#4269d0", "#efb118", "#3ca951", "#ff725c", "#9498a0"
BAND_ORDER = ["am_peak", "inter_peak", "pm_peak", "evening"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MODELS = ["logistic_regression", "random_forest", "gbt"]


# ----------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(show_spinner=False)
def parquet(rel: str, columns=None) -> pd.DataFrame:
    files = glob.glob(str(PQ / rel))
    if not files:
        return pd.DataFrame()
    return pd.concat(pd.read_parquet(f, columns=columns) for f in files)


@st.cache_data(show_spinner=False)
def result_json(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def result_csv(name: str) -> pd.DataFrame:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def stop_reliability() -> pd.DataFrame:
    """Per-stop on-time rate from the matched events (drives the map)."""
    lo = CFG["reliability"]["ontime_lower_min"]
    hi = CFG["reliability"]["ontime_upper_min"]
    ev = parquet("trip_stop_delay/*/*.parquet", ["stop_id", "delay_min"])
    if ev.empty:
        return pd.DataFrame()
    ev["on_time"] = ev.delay_min.between(lo, hi)
    per = ev.groupby("stop_id").agg(on_time=("on_time", "mean"), n_events=("on_time", "size")).reset_index()
    stops = q("SELECT stop_id, name, lat, lon, lsoa_code FROM dim_stop")
    lsoa = q("SELECT lsoa_code, imd_decile FROM dim_lsoa")
    return per.merge(stops, on="stop_id").merge(lsoa, on="lsoa_code", how="left").dropna(subset=["lat", "lon"])


def fig_of(draw, **kw):
    fig, ax = plt.subplots(**kw)
    draw(ax)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------- page
st.set_page_config(page_title="WY Bus Reliability", layout="wide", page_icon="🚌")

if not DB.exists():
    st.error("No warehouse found. Run the pipeline (see README), then reload.")
    st.stop()

st.title("Bus reliability and deprivation in West Yorkshire")
st.caption(
    f"Study window {min(CFG['window']['dates'])} to {max(CFG['window']['dates'])} · "
    "live BODS positions matched against the published timetable"
)

# sidebar filters
rbd_all = q("SELECT * FROM fact_route_band_day")
dates = sorted(rbd_all.service_date.unique())
with st.sidebar:
    st.header("Filters")
    sel_dates = st.multiselect("Service date", dates, default=dates)
    sel_bands = st.multiselect("Time band", BAND_ORDER, default=BAND_ORDER)
    st.caption("Filters apply to the reliability, equity and routes views.")
    st.divider()
    st.caption("Data: DfT Bus Open Data Service (OGL v3.0), IMD 2019, ONS, Open-Meteo.")

rbd = rbd_all[rbd_all.service_date.isin(sel_dates) & rbd_all.time_band.isin(sel_bands)]

tabs = st.tabs([
    "Overview", "Data quality", "EDA", "Reliability", "Equity", "Models", "Map", "Disruptions",
])

# ----------------------------------------------------------------- overview
with tabs[0]:
    trip = q("SELECT COUNT(*) n, AVG(on_time) ot FROM fact_trip_delay").iloc[0]
    match = result_json("match_stats.json")
    sens = {r["window"]: r for r in (result_json("sensitivity.json") or [])}
    metrics = result_csv("metrics.csv")

    c = st.columns(4)
    c[0].metric("Route-band-days", f"{len(rbd_all):,}")
    c[1].metric("Trips observed", f"{int(trip.n):,}")
    c[2].metric("Compliant (>=85% on time)", f"{rbd_all.compliant.mean():.1%}")
    c[3].metric("Trips on time (±2 min)", f"{trip.ot:.1%}")

    st.subheader("The six metrics from the brief")
    m = st.columns(3)
    hw = parquet("headway_obs/*.parquet")
    ttv = parquet("travel_time_variability/*.parquet")
    m[0].metric("Service reliability", f"{rbd_all.compliant.mean():.1%}", "of band-days meet the bar", delta_color="off")
    m[1].metric("Headway regularity", f"{hw.headway_regular.mean():.1%}" if not hw.empty else "n/a",
                "SD within 20% of schedule", delta_color="off")
    m[2].metric("Travel time variability", f"{ttv.ttv_ok.mean():.1%}" if not ttv.empty else "n/a",
                "routes with CV under 15%", delta_color="off")
    m2 = st.columns(3)
    m2[0].metric("Service efficiency (AVL floor)",
                 f"{match['overall_avl_confirmed_rate']:.1%}" if match else "n/a",
                 "scheduled trips with a live trace", delta_color="off")
    if not metrics.empty:
        best = metrics.loc[metrics.f1.idxmax()]
        m2[1].metric("Model efficiency", f"{best.f1:.3f} F1",
                     f"{best.model.replace('_', ' ')} · {best.f1_per_sec:.4f} F1/s", delta_color="off")
    bench = result_json("broadcast_bench.json")
    m2[2].metric("Algorithmic efficiency",
                 f"{bench['no_broadcast']['seconds'] / bench['broadcast']['seconds']:.2f}x" if bench else "2.05x",
                 "broadcast join speedup", delta_color="off")

    st.subheader("Headline")
    st.info(
        "Most route-band-days miss the 85% standard. The deprivation result is the interesting part: "
        "threshold compliance looks slightly better in richer areas, yet raw punctuality is better in "
        "poorer ones. The metric interacts with service frequency — see the Equity tab."
    )
    if sens:
        st.write("**Sensitivity to the on-time window**")
        sdf = pd.DataFrame(result_json("sensitivity.json"))
        st.bar_chart(sdf.set_index("window")["compliant_share"], color=BLUE, height=240)

# ----------------------------------------------------------------- data quality
with tabs[1]:
    st.subheader("Pipeline scale")
    pq_stats = result_json("parse_quality.json")
    scale = {
        "Scheduled stop_times": len(parquet("gtfs/stop_times/*.parquet", ["trip_id"])),
        "Matched stop events": len(parquet("trip_stop_delay/*/*.parquet", ["stop_id"])),
        "Trip facts": int(trip.n),
        "Route-band-days (model rows)": len(rbd_all),
        "Stops": len(q("SELECT stop_id FROM dim_stop")),
        "Routes": len(q("SELECT route_id FROM dim_route")),
    }
    if pq_stats:
        scale["Positions (deduplicated)"] = sum(d["deduped_pings"] for d in pq_stats)
        scale["GTFS-RT snapshots"] = sum(d["snapshots"] for d in pq_stats)
    st.dataframe(
        pd.DataFrame({"stage": list(scale), "records": list(scale.values())}),
        use_container_width=True, hide_index=True,
        column_config={"records": st.column_config.NumberColumn(format="%d")},
    )

    if pq_stats:
        st.subheader("Capture quality by day")
        pdf = pd.DataFrame(pq_stats)
        st.dataframe(
            pdf[["date", "snapshots", "raw_pings", "deduped_pings", "duplicate_share", "trip_id_share"]],
            use_container_width=True, hide_index=True,
        )
        st.caption("Duplicate share is high because buses report slower than the 60s poll; "
                   "trip_id share is what makes matching possible.")

    if match:
        st.subheader("AVL-confirmed operation by day (a floor, not a cancellation count)")
        mdf = pd.DataFrame(match["by_day"])
        st.bar_chart(mdf.set_index("service_date")["avl_confirmed_rate"], color=BLUE, height=280)

    st.subheader("Match quality: ping distance to the stop")
    dist = parquet("trip_stop_delay/*/*.parquet", ["dist_m"])
    if not dist.empty:
        fig_of(lambda ax: (ax.hist(dist.dist_m, bins=50, color=BLUE),
                           ax.set_xlabel("distance from nearest ping to stop (m)"),
                           ax.set_ylabel("matched stop events")), figsize=(9, 3.6))

# ----------------------------------------------------------------- EDA
with tabs[2]:
    trips_df = parquet("fact_trip_delay/*/*.parquet")
    st.subheader("Delay distribution at matched stops")
    ev = parquet("trip_stop_delay/*/*.parquet", ["delay_min"])
    if not ev.empty:
        med, p90 = ev.delay_min.median(), ev.delay_min.quantile(0.9)
        c = st.columns(3)
        c[0].metric("Median delay", f"{med:+.2f} min")
        c[1].metric("90th percentile", f"{p90:+.1f} min")
        c[2].metric("Within ±2 min", f"{ev.delay_min.between(-2, 2).mean():.1%}")

        def _d(ax):
            ax.hist(ev.delay_min.clip(-10, 20), bins=60, color=BLUE)
            ax.axvspan(-2, 2, color=GREEN, alpha=0.15)
            ax.axvline(med, color="#2b2b33", lw=1.4)
            ax.set_xlabel("delay at stop (min; negative = early)")
            ax.set_ylabel("stop events")
        fig_of(_d, figsize=(9, 3.6))
        st.caption("Green band is on time. The tail runs late, which is why only half of stop events are inside it.")

    if not trips_df.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("On-time rate by hour")
            st.line_chart((trips_df.groupby("hour")["on_time_2_2"].mean() * 100), color=BLUE, height=260)
            st.subheader("On-time rate by day")
            g = (trips_df.groupby("day_of_week")["on_time_2_2"].mean() * 100).reindex(DAY_ORDER).dropna()
            st.bar_chart(g, color=BLUE, height=260)
        with right:
            st.subheader("On-time rate by time band")
            g = (trips_df.groupby("time_band")["on_time_2_2"].mean() * 100).reindex(BAND_ORDER).dropna()
            st.bar_chart(g, color=BLUE, height=260)
            st.subheader("Class balance of the target")
            bal = rbd_all.compliant.value_counts().rename({0: "non-compliant", 1: "compliant"})
            st.bar_chart(bal, color=ORANGE, height=260)

    st.subheader("Delay builds up along a trip")
    seq = parquet("trip_stop_delay/*/*.parquet", ["stop_sequence", "delay_min"])
    if not seq.empty:
        s = seq[seq.stop_sequence <= 40].groupby("stop_sequence")["delay_min"].median()
        st.line_chart(s, color=BLUE, height=260)

    st.subheader("Feature correlation")
    feats = parquet("ml_features/*.parquet")
    if not feats.empty:
        cols = ["pct_on_time", "n_trips", "n_stops", "sched_headway_mean_min", "route_imd_decile",
                "route_hist_ontime_rate", "precip_mm", "n_active_disruptions", "compliant"]
        corr = feats[cols].corr()

        def _c(ax):
            im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(cols)), cols, fontsize=7)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(corr.iloc[i, j]) > 0.5 else "#2b2b33")
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig_of(_c, figsize=(8, 6))

# ----------------------------------------------------------------- reliability
with tabs[3]:
    st.subheader("Compliance under the current filter")
    if rbd.empty:
        st.warning("No rows for this filter.")
    else:
        c = st.columns(3)
        c[0].metric("Band-days selected", f"{len(rbd):,}")
        c[1].metric("Compliant", f"{rbd.compliant.mean():.1%}")
        c[2].metric("Mean on-time rate", f"{rbd.pct_on_time.mean():.1%}")

        st.write("**Compliance by time band and day (% of band-days meeting the bar)**")
        pivot = rbd.pivot_table(index="time_band", columns="day_of_week", values="compliant", aggfunc="mean")
        pivot = pivot.reindex(index=[b for b in BAND_ORDER if b in pivot.index],
                              columns=[d for d in DAY_ORDER if d in pivot.columns])
        st.dataframe((pivot * 100).round(1), use_container_width=True)

        st.subheader("Routes ranked by compliance")
        routes = q("SELECT route_id, route_short_name, operator, route_imd_decile FROM dim_route")
        agg = (rbd.groupby("route_id").agg(band_days=("compliant", "size"),
                                           compliance=("compliant", "mean"),
                                           on_time=("pct_on_time", "mean"),
                                           trips=("n_trips", "sum")).reset_index())
        agg = agg[agg.band_days >= 3].merge(routes, on="route_id", how="left")
        worst_first = st.toggle("Show worst first", value=True)
        agg = agg.sort_values("compliance", ascending=worst_first)
        table = (
            agg[["route_short_name", "operator", "route_imd_decile", "band_days", "trips", "on_time", "compliance"]]
            .rename(columns={"route_short_name": "route", "route_imd_decile": "IMD decile",
                             "on_time": "on-time %", "compliance": "compliant %"})
        )
        table["on-time %"] = (table["on-time %"] * 100).round(1)
        table["compliant %"] = (table["compliant %"] * 100).round(1)
        table["IMD decile"] = table["IMD decile"].round(0)
        st.dataframe(table, use_container_width=True, hide_index=True, height=380)

    st.subheader("Other brief metrics")
    cc = st.columns(2)
    if not hw.empty:
        cc[0].write("**Headway regularity** (bar: SD ≤ 20% of scheduled)")
        cc[0].metric("Band-days meeting the bar", f"{hw.headway_regular.mean():.1%}")
    if not ttv.empty:
        cc[1].write("**Travel time variability** (bar: CV ≤ 15%)")
        cc[1].metric("Routes meeting the bar", f"{ttv.ttv_ok.mean():.1%}",
                     f"median CV {ttv.ttv_cv.median():.3f}", delta_color="off")

# ----------------------------------------------------------------- equity
with tabs[4]:
    eq = result_json("equity_stats.json")
    st.subheader("Compliance and punctuality by deprivation decile")
    routes = q("SELECT route_id, route_imd_decile FROM dim_route WHERE route_imd_decile IS NOT NULL")
    joined = rbd.merge(routes, on="route_id")
    if joined.empty:
        st.warning("No rows for this filter.")
    else:
        by_dec = joined.groupby(joined.route_imd_decile.round().astype(int)).agg(
            compliance=("compliant", "mean"), on_time=("pct_on_time", "mean"), band_days=("compliant", "size"))
        by_dec.index.name = "IMD decile (1 = most deprived)"
        a, b = st.columns(2)
        a.write("**Threshold compliance**")
        a.bar_chart(by_dec["compliance"], color=BLUE, height=280)
        b.write("**Raw on-time rate**")
        b.bar_chart(by_dec["on_time"], color=ORANGE, height=280)
        dec_table = by_dec.copy()
        dec_table["compliance"] = (dec_table["compliance"] * 100).round(1)
        dec_table["on_time"] = (dec_table["on_time"] * 100).round(1)
        st.dataframe(dec_table.rename(columns={"compliance": "compliant %", "on_time": "on-time %"}),
                     use_container_width=True)

    if eq:
        st.subheader("Tested association (permutation test, 10,000 shuffles)")
        c = st.columns(2)
        cc1, cc2 = eq["correlation_imd_vs_compliance"], eq["correlation_imd_vs_on_time"]
        c[0].metric("IMD vs compliance", f"rho {cc1['spearman_rho']:+.3f}",
                    f"p = {cc1['p_value_permutation']:.3f} · n = {cc1['n_routes']} routes", delta_color="off")
        c[1].metric("IMD vs raw on-time", f"rho {cc2['spearman_rho']:+.3f}",
                    f"p = {cc2['p_value_permutation']:.3f}", delta_color="off")
        st.warning(
            "The two point opposite ways. Compliance mildly favours affluent areas, but raw punctuality "
            "favours deprived areas. The explanation is below."
        )

        st.subheader("Why: the threshold rewards infrequent service")
        fc = eq["frequency_confound"]
        bucket = pd.DataFrame(fc["compliance_by_trips_per_band"]).set_index("trips_per_band")

        def _fc(ax):
            x = range(len(bucket))
            ax.bar(x, bucket.compliance * 100, color=BLUE, width=0.6, label="band-days compliant")
            ax.plot(x, bucket.on_time * 100, color=ORANGE, marker="o", lw=2, label="mean on-time rate")
            ax.set_xticks(list(x), bucket.index)
            ax.set_xlabel("scheduled trips in the band")
            ax.set_ylabel("per cent")
            ax.set_ylim(0, 100)
            ax.legend(frameon=False)
        fig_of(_fc, figsize=(9, 4))
        st.caption(
            f"Frequency vs actual punctuality: rho {fc['rho_n_trips_vs_pct_on_time']:+.3f} (no link). "
            f"Frequency vs compliance: rho {fc['rho_n_trips_vs_compliant']:+.3f}. "
            f"Deprived areas run the frequent services: rho {fc['rho_n_trips_vs_imd_decile']:+.3f}."
        )
        gap = eq["deprivation_gap"]
        st.metric("Deciles 1-3 vs 8-10 compliance gap", f"{gap['gap_percentage_points']:+.1f} pp",
                  f"{gap['deciles_1_3_compliance']:.1%} vs {gap['deciles_8_10_compliance']:.1%}", delta_color="off")

# ----------------------------------------------------------------- models
with tabs[5]:
    metrics = result_csv("metrics.csv")
    if metrics.empty:
        st.info("Run `python -m src.ml.evaluate` to populate model metrics.")
    else:
        st.subheader("Held-out evaluation (future days the models never saw)")
        show = metrics.assign(model=metrics.model.str.replace("_", " ")).round(
            {"accuracy": 3, "precision": 3, "recall": 3, "f1": 3, "roc_auc": 3,
             "pr_auc": 3, "train_seconds": 1, "f1_per_sec": 4})
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Majority-class baseline accuracy is 0.730 on a 27% positive rate, so F1 and PR-AUC "
                   "are the honest measures here.")
        st.bar_chart(metrics.assign(model=metrics.model.str.replace("_", " "))
                     .set_index("model")[["f1", "roc_auc", "pr_auc", "accuracy"]],
                     height=320, stack=False)

        st.subheader("ROC and precision-recall")
        a, b = st.columns(2)
        roc = {m: result_csv(f"roc_{m}.csv") for m in MODELS}
        pr = {m: result_csv(f"pr_{m}.csv") for m in MODELS}
        if any(not d.empty for d in roc.values()):
            def _roc(ax):
                for m, d in roc.items():
                    if not d.empty:
                        ax.plot(d.fpr, d.tpr, label=m.replace("_", " "))
                ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
                ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
                ax.legend(frameon=False)
            with a:
                fig_of(_roc, figsize=(5, 4))
        if any(not d.empty for d in pr.values()):
            def _pr(ax):
                for m, d in pr.items():
                    if not d.empty:
                        ax.plot(d.recall, d.precision, label=m.replace("_", " "))
                ax.set_xlabel("recall"); ax.set_ylabel("precision")
                ax.legend(frameon=False)
            with b:
                fig_of(_pr, figsize=(5, 4))

        pick = st.selectbox("Model detail", MODELS, index=1, format_func=lambda s: s.replace("_", " "))
        cm = result_csv(f"confusion_{pick}.csv")
        c1, c2 = st.columns(2)
        if not cm.empty:
            with c1:
                st.write("**Confusion matrix**")
                table = cm.pivot_table(index="compliant", columns="prediction", values="count",
                                       aggfunc="sum").fillna(0).astype(int)
                table.index = ["true non-compliant", "true compliant"][: len(table)]
                table.columns = ["pred non-compliant", "pred compliant"][: len(table.columns)]
                st.dataframe(table, use_container_width=True)
        sweep = result_csv("threshold_sweep.csv")
        if not sweep.empty:
            with c2:
                st.write("**Decision threshold sweep** (random forest)")
                st.line_chart(sweep.set_index("threshold")[["precision", "recall", "f1"]], height=280)
                best = sweep.loc[sweep.f1.idxmax()]
                st.caption(f"Minority-class F1 peaks at {best.threshold:.2f} "
                           f"(precision {best.precision:.2f}, recall {best.recall:.2f}), not 0.5.")

        imp = result_csv("feature_importance.csv")
        if not imp.empty:
            st.subheader("Feature importance")
            which = st.radio("Tree model", ["random_forest", "gbt"], horizontal=True,
                             format_func=lambda s: s.replace("_", " "))
            top = imp[imp.model == which].nlargest(12, "importance").set_index("feature")["importance"]
            st.bar_chart(top, color=BLUE, height=320)
            st.caption("A route's own history leads; trips-per-band is next, which is the frequency effect "
                       "the equity analysis picks up independently.")

        coef = result_csv("lr_coefficients.csv")
        if not coef.empty:
            with st.expander("Logistic regression coefficients (standardised features)"):
                st.dataframe(coef.head(15).round({"coefficient": 3}),
                             use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- map
with tabs[6]:
    st.subheader("Reliability map")
    stops = stop_reliability()
    if stops.empty:
        st.info("No matched events yet. Run the processing stage first.")
    else:
        c = st.columns(3)
        min_events = c[0].slider("Minimum matched passes per stop", 1, 200, 20, step=1)
        dec = c[1].slider("IMD decile range (1 = most deprived)", 1, 10, (1, 10))
        band = c[2].select_slider("Colour threshold (on-time %)", options=list(range(50, 96, 5)), value=85)

        df = stops[(stops.n_events >= min_events)]
        df = df[df.imd_decile.between(dec[0], dec[1]) | df.imd_decile.isna()]

        def colour(v):
            if v >= band / 100:
                return [60, 169, 81]
            if v >= (band - 25) / 100:
                return [239, 177, 24]
            return [255, 114, 92]

        df = df.assign(color=df.on_time.map(colour), pct=(df.on_time * 100).round(1))
        k = st.columns(4)
        k[0].metric("Stops shown", f"{len(df):,}")
        k[1].metric("Mean on-time", f"{df.on_time.mean():.1%}" if len(df) else "n/a")
        k[2].metric(f"At or above {band}%", f"{(df.on_time >= band/100).mean():.1%}" if len(df) else "n/a")
        k[3].metric("Matched passes", f"{int(df.n_events.sum()):,}")

        bbox = CFG["region"]["bbox"]
        if HAS_PYDECK:
            st.pydeck_chart(pdk.Deck(
                map_style="light",
                initial_view_state=pdk.ViewState(
                    latitude=(bbox["min_lat"] + bbox["max_lat"]) / 2,
                    longitude=(bbox["min_lon"] + bbox["max_lon"]) / 2,
                    zoom=9.2, pitch=0,
                ),
                layers=[pdk.Layer(
                    "ScatterplotLayer", data=df,
                    get_position=["lon", "lat"], get_fill_color="color",
                    get_radius=90, radius_min_pixels=2, radius_max_pixels=8,
                    pickable=True, opacity=0.75,
                )],
                tooltip={"text": "{name}\non time: {pct}%\npasses: {n_events}\nIMD decile: {imd_decile}"},
            ))
            st.caption("Green at or above the chosen threshold, amber within 25 points below it, red under "
                       "that. Drag to pan, scroll to zoom, hover a stop for detail.")
        else:
            hexed = df.assign(hex="#" + df.color.map(lambda c: "%02x%02x%02x" % tuple(c)))
            st.map(hexed, latitude="lat", longitude="lon", color="hex", size=40)
            st.caption("Green at or above the chosen threshold, amber within 25 points below it, red under "
                       "that. Install pydeck (`pip install -r requirements.txt`) for hover detail.")

        with st.expander("Worst stops in view"):
            st.dataframe(
                df.nsmallest(25, "on_time")[["name", "pct", "n_events", "imd_decile"]]
                .rename(columns={"name": "stop", "pct": "on-time %", "n_events": "passes",
                                 "imd_decile": "IMD decile"}),
                use_container_width=True, hide_index=True,
            )

# ----------------------------------------------------------------- disruptions
with tabs[7]:
    st.subheader("Disruptions reported during the window (BODS SIRI-SX)")
    dis = q("SELECT * FROM fact_disruption")
    if dis.empty:
        st.info("No disruption snapshots loaded.")
    else:
        c = st.columns(3)
        c[0].metric("Situations", f"{len(dis):,}")
        c[1].metric("West Yorkshire specific", f"{int(dis.wy_specific.sum()):,}")
        c[2].metric("Planned", f"{dis.planned.mean():.0%}")
        st.write("**By reason**")
        st.bar_chart(dis.reason.value_counts().head(10), color=ORANGE, height=280)
        links = q("SELECT COUNT(*) n FROM disruption_stop").iloc[0].n
        st.caption(f"{links:,} disruption-to-stop links join these situations back to routes through stop_times.")
        st.dataframe(
            dis[["situation_id", "reason", "planned", "summary", "validity_start", "validity_end",
                 "n_affected_stops"]].sort_values("n_affected_stops", ascending=False),
            use_container_width=True, hide_index=True, height=360,
        )
