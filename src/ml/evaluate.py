"""Evaluate the three models on the held-out future block and compare them.

Reports the full set the brief asks for: Accuracy, Precision, Recall, F1 and
ROC-AUC, plus PR-AUC and a confusion matrix. Because most route-band-days are
non-compliant (buses run late), accuracy is misleading, so we lead with F1 / PR-
AUC and print a majority-class baseline for context. The Model Efficiency metric
(F1 per training-second) combines these with the times saved during training.

ROC and PR curve points are written out for the plotting stage.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

from src.common import get_spark, load_config, project_path

log = logging.getLogger("evaluate")
LABEL = "compliant"
MODELS = ["logistic_regression", "random_forest", "gbt"]


def _curve_points(pdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ROC and PR points from (score, label), no sklearn dependency."""
    pdf = pdf.sort_values("score", ascending=False).reset_index(drop=True)
    P = (pdf.label == 1).sum()
    N = (pdf.label == 0).sum()
    tp = fp = 0
    roc, pr = [(0.0, 0.0)], []
    for lbl in pdf.label:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        roc.append((fpr, tpr))
        pr.append((tpr, prec))  # (recall, precision)
    return (
        pd.DataFrame(roc, columns=["fpr", "tpr"]),
        pd.DataFrame(pr, columns=["recall", "precision"]),
    )


def evaluate_model(spark, name: str, pred_dir, results_dir) -> dict:
    preds = spark.read.parquet(str(pred_dir / name))
    preds = preds.withColumn("score", vector_to_array("probability")[1])

    acc = MulticlassClassificationEvaluator(labelCol=LABEL, metricName="accuracy")
    f1 = MulticlassClassificationEvaluator(labelCol=LABEL, metricName="f1")
    prec = MulticlassClassificationEvaluator(labelCol=LABEL, metricName="weightedPrecision")
    rec = MulticlassClassificationEvaluator(labelCol=LABEL, metricName="weightedRecall")
    roc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="probability", metricName="areaUnderROC")
    prc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="probability", metricName="areaUnderPR")

    metrics = {
        "model": name,
        "accuracy": acc.evaluate(preds),
        "precision": prec.evaluate(preds),
        "recall": rec.evaluate(preds),
        "f1": f1.evaluate(preds),
        "roc_auc": roc.evaluate(preds),
        "pr_auc": prc.evaluate(preds),
    }

    cm = preds.groupBy(LABEL, "prediction").count().toPandas()
    cm.to_csv(results_dir / f"confusion_{name}.csv", index=False)

    pdf = preds.select(F.col(LABEL).alias("label"), "score").toPandas()
    roc_pts, pr_pts = _curve_points(pdf)
    roc_pts.to_csv(results_dir / f"roc_{name}.csv", index=False)
    pr_pts.to_csv(results_dir / f"pr_{name}.csv", index=False)
    return metrics


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    pred_dir = pq / "predictions"
    results_dir = project_path("docs", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    times_path = results_dir / "train_times.json"
    train_times = json.loads(times_path.read_text()) if times_path.exists() else {}

    rows = []
    for name in MODELS:
        if not (pred_dir / name).exists():
            log.warning("no predictions for %s -- skipping", name)
            continue
        m = evaluate_model(spark, name, pred_dir, results_dir)
        secs = train_times.get(name)
        m["train_seconds"] = secs
        m["f1_per_sec"] = round(m["f1"] / secs, 5) if secs else None
        rows.append(m)

    # majority-class baseline for context
    feats = spark.read.parquet(str(pq / "ml_features")).filter("is_train = 0")
    total = feats.count()
    maj = feats.groupBy(LABEL).count().orderBy(F.desc("count")).first()
    baseline_acc = (maj["count"] / total) if total else 0.0
    log.info("majority-class baseline accuracy on test: %.3f", baseline_acc)

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "metrics.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== Model comparison (held-out future block) ===")
    print(table.to_string(index=False))
    print(f"\nMajority-class baseline accuracy: {baseline_acc:.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("evaluate")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
