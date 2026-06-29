"""Static figures for the report (matplotlib).

Everything here works off the *small* aggregated outputs -- the metrics CSVs from
the evaluation stage and a couple of grouped queries against SQLite. We never
plot straight from Spark; the whole point is to convert only the tiny result set.
Figures are saved to docs/figures/ for the report.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, project_path
from src.db.queries import compliance_by_imd_decile, connect

log = logging.getLogger("plots")

RESULTS = project_path("docs", "results")
FIGS = project_path("docs", "figures")


def model_comparison() -> None:
    path = RESULTS / "metrics.csv"
    if not path.exists():
        log.warning("no metrics.csv -- run evaluate first")
        return
    df = pd.read_csv(path)
    metrics = ["f1", "roc_auc", "pr_auc", "accuracy"]
    ax = df.set_index("model")[metrics].plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("score")
    ax.set_title("Model comparison on held-out future block")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGS / "model_comparison.png", dpi=150)
    plt.close()
    log.info("wrote model_comparison.png")


def curves() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in ("logistic_regression", "random_forest", "gbt"):
        roc = RESULTS / f"roc_{name}.csv"
        pr = RESULTS / f"pr_{name}.csv"
        if roc.exists():
            d = pd.read_csv(roc)
            ax1.plot(d.fpr, d.tpr, label=name)
        if pr.exists():
            d = pd.read_csv(pr)
            ax2.plot(d.recall, d.precision, label=name)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "roc_pr_curves.png", dpi=150)
    plt.close()
    log.info("wrote roc_pr_curves.png")


def equity_chart() -> None:
    conn = connect()
    try:
        rows = compliance_by_imd_decile(conn)
    finally:
        conn.close()
    if not rows:
        log.warning("no equity rows -- is the DB built?")
        return
    df = pd.DataFrame([dict(r) for r in rows])
    ax = df.plot(x="imd_decile", y="compliance_rate", kind="bar", legend=False, figsize=(8, 5))
    ax.set_xlabel("IMD decile (1 = most deprived)")
    ax.set_ylabel("Mean compliance rate")
    ax.set_title("Service compliance by deprivation decile")
    plt.tight_layout()
    plt.savefig(FIGS / "compliance_by_imd.png", dpi=150)
    plt.close()
    log.info("wrote compliance_by_imd.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_config()
    FIGS.mkdir(parents=True, exist_ok=True)
    model_comparison()
    curves()
    equity_chart()


if __name__ == "__main__":
    main()
