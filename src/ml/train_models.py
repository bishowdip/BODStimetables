"""Train and compare three MLlib classifiers on an identical, leakage-safe split.

Same features, same time-aware split for all three:
  * LogisticRegression  -- interpretable baseline (+ StandardScaler)
  * RandomForestClassifier
  * GBTClassifier        -- expected top performer

Tuning uses CrossValidator *inside the training block only* (days 1-5); the held-
out future block (days 6-7) is untouched until evaluation. A random k-fold would
leak future information across the date boundary, which is why we split by time.

For each model we save: the fitted pipeline, its test-set predictions, and the
wall-clock training time (the Model Efficiency metric, F1/sec, needs it).
"""
from __future__ import annotations

import json
import logging
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from src.common import get_spark, load_config, project_path

log = logging.getLogger("train_models")

LABEL = "compliant"
CATEGORICAL = ["day_of_week", "time_band"]
NUMERIC = [
    "n_stops", "n_trips", "precip_mm", "temp_c", "wind_kph",
    "route_imd_decile", "route_hist_ontime_rate", "is_weekend",
]


def base_stages(scale: bool):
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    ohe = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh") for c in CATEGORICAL]
    assembler = VectorAssembler(
        inputCols=NUMERIC + [f"{c}_oh" for c in CATEGORICAL],
        outputCol="features_raw" if scale else "features",
        handleInvalid="keep",
    )
    stages = idx + ohe + [assembler]
    if scale:
        stages.append(StandardScaler(inputCol="features_raw", outputCol="features"))
    return stages


def model_specs():
    """Return (name, estimator, param grid) for each classifier."""
    lr = LogisticRegression(featuresCol="features", labelCol=LABEL)
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )

    rf = RandomForestClassifier(featuresCol="features", labelCol=LABEL)
    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [5, 10])
        .build()
    )

    gbt = GBTClassifier(featuresCol="features", labelCol=LABEL)
    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [30, 60])
        .addGrid(gbt.maxDepth, [3, 5])
        .build()
    )
    return [
        ("logistic_regression", lr, lr_grid, True),
        ("random_forest", rf, rf_grid, False),
        ("gbt", gbt, gbt_grid, False),
    ]


def run(spark, cfg) -> None:
    pq = project_path(cfg["paths"]["parquet"])
    feats = spark.read.parquet(str(pq / "ml_features")).cache()
    train = feats.filter("is_train = 1")
    test = feats.filter("is_train = 0")
    log.info("train rows=%d, test rows=%d", train.count(), test.count())

    evaluator = BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC")
    models_dir = pq / "models"
    pred_dir = pq / "predictions"
    train_times = {}

    for name, estimator, grid, scale in model_specs():
        pipeline = Pipeline(stages=base_stages(scale) + [estimator])
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=2,
        )
        log.info("training %s ...", name)
        t0 = time.perf_counter()
        cv_model = cv.fit(train)
        elapsed = time.perf_counter() - t0
        train_times[name] = round(elapsed, 2)
        log.info("%s trained in %.1fs (best CV AUC=%.4f)", name, elapsed, max(cv_model.avgMetrics))

        best = cv_model.bestModel
        best.write().overwrite().save(str(models_dir / name))
        best.transform(test).select(LABEL, "prediction", "probability") \
            .write.mode("overwrite").parquet(str(pred_dir / name))

    results_dir = project_path("docs", "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "train_times.json").write_text(json.dumps(train_times, indent=2))
    log.info("training times: %s", train_times)
    feats.unpersist()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    spark = get_spark("train_models")
    try:
        run(spark, cfg)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
