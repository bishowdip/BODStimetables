# Figure index

Captions for the report and appendix. Committed images live in docs/; the
rest are regenerated into docs/figures/ by the run order in the README.

| # | File | Caption |
|---|---|---|
| 1 | architecture.png | System architecture: collectors, Spark processing, storage and models, visualisation. Parquet partitioned by service date is the hand-off between stages. |
| 2 | spark_ui_stages.png | Spark UI during the matching join: 8/8 tasks per shuffle stage, shuffle read up to 165 MiB, nine skipped stages where earlier shuffle output was reused. |
| 3 | spark_ui_executors.png | Executor and task metrics for the same job. |
| 4 | schema.png | Star schema: route, stop and LSOA dimensions; trip-level and route-band-day fact tables; disruption bridge. |
| 5 | figures/headway_distribution.png | Scheduled headway distribution across 1,480 route x time-band combinations (right-skewed: skewness 1.63). |
| 6 | figures/compliance_by_imd.png | Share of route-band-days meeting the 85% bar by IMD decile (1 = most deprived). |
| 7 | figures/model_comparison.png | Held-out comparison of the three classifiers: F1, ROC-AUC, PR-AUC, accuracy. |
| 8 | figures/roc_pr_curves.png | ROC and precision-recall curves on the future test days. |
| 9 | figures/feature_importance.png | Random-forest feature importance; route history and trips-per-band dominate. |
| 10 | figures/threshold_sweep.png | Precision, recall and F1 for the compliant class across decision thresholds; F1 peaks at 0.30. |
| 11 | figures/reliability_map.html | Interactive map of 12,393 stops coloured by measured on-time rate (open in a browser; screenshot for the appendix). |
