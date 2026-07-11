"""Model interpretation: what drives predicted non-compliance.

Three artefacts, all from the fitted pipelines and the held-out predictions:

  * tree-model feature importances (RF and GBT), with one-hot columns mapped
    back to readable names via the vector metadata;
  * logistic-regression coefficients (standardised features, so magnitudes
    compare);
  * a decision-threshold sweep on the best model -- with a ~27% positive class
    the default 0.5 cut is not obviously right, and the sweep shows the
    precision/recall trade a regulator could actually choose from.

Outputs: docs/results/feature_importance.csv, lr_coefficients.csv,
threshold_sweep.csv and matching figures under docs/figures/.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from src.common import get_spark, load_config, project_path

log = logging.getLogger("interpret")

RESULTS = project_path("docs", "results")
FIGS = project_path("docs", "figures")
LABEL = "compliant"


def feature_names(transformed_df, col: str = "features") -> list[str]:
    """Readable names for every slot of the assembled vector, via ML metadata."""
    meta = transformed_df.schema[col].metadata["ml_attr"]["attrs"]
    slots: dict[int, str] = {}
    for group in meta.values():
        for attr in group:
            slots[attr["idx"]] = attr["name"]
    return [slots[i] for i in sorted(slots)]


def tree_importances(spark, feats, models_dir) -> pd.DataFrame:
    frames = []
    for name in ("random_forest", "gbt"):
        model = PipelineModel.load(str(models_dir / name))
        transformed = model.transform(feats.limit(1))
        names = feature_names(transformed)
        imps = model.stages[-1].featureImportances.toArray()
        frames.append(pd.DataFrame({"feature": names, "importance": imps, "model": name}))
    return pd.concat(frames, ignore_index=True)


def lr_coefficients(spark, feats, models_dir) -> pd.DataFrame:
    model = PipelineModel.load(str(models_dir / "logistic_regression"))
    transformed = model.transform(feats.limit(1))
    # the scaler strips vector metadata, so read names from its input column;
    # scaling preserves slot order, so the mapping still holds
    names = feature_names(transformed, "features_raw")
    coefs = model.stages[-1].coefficients.toArray()
    return pd.DataFrame({"feature": names, "coefficient": coefs}).sort_values(
        "coefficient", key=abs, ascending=False
    )


def threshold_sweep(spark, pred_dir, model: str = "random_forest") -> pd.DataFrame:
    preds = spark.read.parquet(str(pred_dir / model))
    pdf = (
        preds.withColumn("score", vector_to_array("probability")[1])
        .select(F.col(LABEL).alias("label"), "score")
        .toPandas()
    )
    rows = []
    for t in [round(0.05 * i, 2) for i in range(2, 19)]:
        pred = (pdf.score >= t).astype(int)
        tp = int(((pred == 1) & (pdf.label == 1)).sum())
        fp = int(((pred == 1) & (pdf.label == 0)).sum())
        fn = int(((pred == 0) & (pdf.label == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def plot_importances(imp: pd.DataFrame) -> None:
    top = (
        imp[imp.model == "random_forest"].nlargest(12, "importance")
        .sort_values("importance")
    )
    ax = top.plot.barh(x="feature", y="importance", legend=False, figsize=(8, 6))
    ax.set_title("Random Forest feature importance (top 12)")
    ax.set_xlabel("importance")
    plt.tight_layout()
    plt.savefig(FIGS / "feature_importance.png", dpi=150)
    plt.close()


def plot_threshold(sweep: pd.DataFrame) -> None:
    ax = sweep.plot(x="threshold", y=["precision", "recall", "f1"], figsize=(8, 5))
    ax.set_title("Decision-threshold trade-off (Random Forest, held-out days)")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGS / "threshold_sweep.png", dpi=150)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    pq = project_path(cfg["paths"]["parquet"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    spark = get_spark("interpret")
    try:
        feats = spark.read.parquet(str(pq / "ml_features"))
        imp = tree_importances(spark, feats, pq / "models")
        coef = lr_coefficients(spark, feats, pq / "models")
        sweep = threshold_sweep(spark, pq / "predictions")
    finally:
        spark.stop()

    imp.to_csv(RESULTS / "feature_importance.csv", index=False)
    coef.to_csv(RESULTS / "lr_coefficients.csv", index=False)
    sweep.to_csv(RESULTS / "threshold_sweep.csv", index=False)
    plot_importances(imp)
    plot_threshold(sweep)

    top5 = imp[imp.model == "random_forest"].nlargest(5, "importance")
    log.info("RF top features: %s", ", ".join(f"{r.feature}={r.importance:.3f}" for r in top5.itertuples()))
    best = sweep.loc[sweep.f1.idxmax()]
    log.info("best F1 threshold: %.2f (F1=%.3f, precision=%.3f, recall=%.3f)",
             best.threshold, best.f1, best.precision, best.recall)


if __name__ == "__main__":
    main()
