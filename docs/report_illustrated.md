# Bus reliability and deprivation in West Yorkshire

**Module** ST5011CEM Big Data Programming Project · **Author** Bishow Dip ·
**Supervisor** Mr. Siddhartha Neupane · **Code**
https://github.com/bishowdip/BODStimetables

This is the illustrated version of the report: the same text as
`report_draft.md` with every figure placed inline. Use it to see the whole
argument with its evidence in one read. The figures are regenerated from the
measured results by the pipeline, so every number in them is reproducible.

---

## 1. Executive summary

Bus regulators in England enforce a simple standard: at least 85% of trips
should run within two minutes of the timetable. This project measured a real
week of West Yorkshire buses against that standard, predicted where it would
fail, and asked who the failures land on. I captured 4,933,858 live vehicle
positions from the Bus Open Data Service over seven days in July 2026 and
matched them to 2,565,042 scheduled stop times with PySpark. Only 26.8% of the
9,533 route, time-band and day combinations met the bar. A random forest
predicted non-compliance on two unseen future days with F1 0.746 and ROC-AUC
0.763. The most useful finding was not the prediction but a flaw in the metric
itself: the threshold rewards infrequent service. Deprived areas run the
frequent services, so they fail the standard even though their buses are, on
average, more punctual.

![The six metrics from the brief, measured on the study week](figures/info_metric_tiles.png)

*Figure 1. The brief's six reliability and efficiency metrics, measured on the
study week. Red marks a value below the target, green above it.*

## 2. Introduction

Average punctuality figures hide two things: when the failures happen, and who
they happen to. A Local Transport Authority deciding where to direct
enforcement needs both. That is the stakeholder and the decision this project
serves.

The pipeline ingests four kinds of open data (timetables, live vehicle
positions, service disruptions and deprivation statistics), computes the
brief's reliability metrics at route level, and trains three classifiers to
predict which route and time-band combinations will fall below the 85%
standard the following day. Everything runs on commodity hardware with PySpark
doing the heavy work.

All six metric definitions from the brief are used. Service Reliability (85%
of trips within plus or minus two minutes) is the prediction target. Headway
Regularity (standard deviation of the gap between buses, against a
20%-of-schedule bar) and Travel Time Variability (coefficient of variation of
trip duration, against 15%) are computed per route. Service Efficiency is
reported as the share of scheduled trips confirmed by live data. Algorithmic
Efficiency is documented with stage timings and Spark UI evidence, and Model
Efficiency as F1 per second of training.

The work addresses learning outcomes B1 (complexity analysis of the matching
join), B2 (a working multi-stage system), B4 (large-scale data and machine
learning), B6 (version control, parameterised SQL, ethics), B7 (this report
and the viva) and B8 (the equity analysis).

## 3. Related work

Furth and Muller (2006) showed that automatic vehicle location data changes
how reliability should be measured, because unreliable service costs
passengers budgeted waiting time rather than average waiting time. Daganzo
(2009) treats bus bunching as a control problem: once headways destabilise
they collapse, which motivates measuring headway regularity separately from
lateness. Lucas (2012) reviews the evidence that transport disadvantage
concentrates on low-income communities. Robinson (1950) is the standard
caution for area-level analysis, and this project keeps every equity claim at
LSOA level for that reason.

## 4. Data collection and preprocessing

Six sources were used, all real and all open. BODS supplies three: the
Yorkshire GTFS timetable (a 109 MB zip), the GTFS-RT vehicle position feed,
and the SIRI-SX disruptions feed. The English Indices of Deprivation 2019
give an IMD decile per LSOA, ONS boundaries locate the 1,767 West Yorkshire
LSOAs, and Open-Meteo provides hourly weather without a key.

![How each dataset is collected](figures/info_data_sources.png)

*Figure 2. The six sources and how each was collected. The two live feeds
(blue rows) are polled over the window; the rest are single downloads.*

Positions were the hard part. BODS serves them live only; there is no
historical archive to download. So the feed was polled every 60 seconds for
seven days (2 to 8 July 2026), server-filtered to the West Yorkshire bounding
box, saving 10,546 protobuf snapshots. Coverage was 97.5% of the theoretical
maximum, with one gap of about three hours on the Sunday.

Parsing produced 11.4 million raw pings. Buses report on their own cadence,
slower than the poll rate, so 57% of pings were repeats of the same vehicle
message; deduplicating on vehicle and timestamp left 4,933,858 genuine
positions, 97.5% of which carry a trip identifier. Cleaning also wrapped GTFS
times past midnight (a 24:05 arrival observed at 00:10 is five minutes late,
not 24 hours early) and dropped the 1.17% of matched events with delays over
an hour, which inspection showed to be GPS artefacts.

Combined, the datasets total over seven million records, roughly seventy times
the 100,000-record requirement, met with real multi-catalogue, multi-day and
supplementary ingestion. The brief's three-month extended window is an
augmentation option for datasets below the threshold, which this one exceeds
seventy-fold; it is also impractical here, since the position feed is live-only
and three months of it can only exist after three months of capture. No
synthetic data was used.

![Row counts at each stage of the pipeline](figures/info_data_funnel.png)

*Figure 3. Volumes through the pipeline, from 11.4 million raw pings down to
the 9,533 route-band-day rows the models use (log scale).*

## 5. Methodology

Matching joins each position to its trip's scheduled stops (trip identifier
plus service day), computes the haversine distance to every stop, and keeps the
nearest ping within 50 metres per trip, day and stop. Its timestamp is the
inferred passing time and the difference from schedule is the delay. This
yielded 1,698,693 matched stop events across 34,077 distinct trips. With the
small stops table broadcast, the join costs roughly O(P log S) rather than the
naive P times S comparison; the full complexity table is in
`docs/architecture.md`.

A trip counts as on time when its median stop delay sits within plus or minus
two minutes (the median resists GPS jumps). Trips aggregate to route by
time-band by day rows, compliant when at least 85% of trips were on time:
9,533 rows, 26.8% compliant, which is the class imbalance the models face.

Every feature is known before the trip runs: day of week, time band, weekend
flag, stop count, scheduled trips in the band, scheduled headway mean and
deviation, weather, route IMD decile, active disruptions, and the route's
historical on-time rate computed from training days only. A unit test
recomputes that historical feature independently and requires agreement to
within 1e-6, and asserts that every training date precedes every test date.

The split is time-ordered because the task is forecasting: train on 2 to 6
July (6,592 rows), test on 7 and 8 July (2,941 rows). A random k-fold would let
the model see the future. CrossValidator tunes hyperparameters with 3-fold
validation inside the training block for logistic regression (the interpretable
baseline), random forest and gradient-boosted trees.

## 6. System design and implementation

The system is four layers with Parquet as the hand-off between them, so any
stage re-runs alone. Collectors in plain Python poll the live feeds; Spark owns
everything from parsing output to features; SQLite stores the star schema;
pandas and matplotlib touch only final aggregates. The one deliberate exception
is the stop-to-LSOA spatial join, which runs in GeoPandas because 16,139 stops
is not a big-data problem.

![System architecture](architecture.png)

*Figure 4. The four-layer architecture. Parquet partitioned by service date is
the hand-off between every stage.*

Spark evidence: eight shuffle partitions throughout; broadcasting the stops
table cut the matching join from 8.98 to 4.37 seconds (2.05 times faster,
identical 2,602,507 output rows); the cached matched-events table feeds three
aggregations.

![Spark UI stages for the matching join](spark_ui_stages.png)

*Figure 5. The Spark UI during the real matching join: 8 of 8 tasks per shuffle
stage, shuffle reads up to 165 MiB, and nine skipped stages where Spark reused
earlier shuffle output instead of recomputing it (lazy evaluation in practice).*

Storage is a star schema. Every SQL statement uses ?-placeholders, and the only
credential (the BODS key) lives in a git-ignored .env file read at runtime.

![Database schema](schema.png)

*Figure 6. The SQLite star schema: route, stop and LSOA dimensions around
trip-level and route-band-day fact tables, with a disruption bridge.*

On an 8 GB laptop the memory question is not academic. A national week of
positions cannot sit in pandas; the parser handles one snapshot at a time and
Spark spills partitions to disk beyond that.

## 7. Results and evaluation

Half of matched stop events (50.3%) were within plus or minus two minutes; the
median trip delay was 1.05 minutes late with a long tail (90th percentile 7.3
minutes late). At the standard's grain, 26.8% of route-band-days were
compliant. Widening the window raises the share smoothly, so the headline is
not an artefact of the two-minute choice.

![Distribution of delays at stops](figures/info_delay_distribution.png)

*Figure 7. Delay at every matched stop event. The distribution leans late: the
median is about a minute behind schedule and the tail runs well past the
on-time band, which is why only half of stop events are inside it.*

![Compliance under three on-time windows](figures/info_sensitivity.png)

*Figure 8. Compliance under the plus or minus 2, 3 and 5 minute windows. The
gradient is smooth, so the finding does not depend on one threshold.*

Headway regularity was met on 7.8% of band-days and the travel-time bar on
13.6% of routes. The AVL-confirmed floor was 79.1% of the week's 100,100
scheduled trips, lower on the gapped Sunday. A trip without a trace is
unconfirmed, not proven cancelled.

![AVL-confirmed operation by day](figures/info_avl_by_day.png)

*Figure 9. Share of scheduled trips that left a live trace, by day, against the
weekly floor of 79.1%. Sunday's dip includes the three-hour capture gap.*

On the unseen future days the random forest performed best (accuracy 0.769, F1
0.746, ROC-AUC 0.763, PR-AUC 0.534, against a 0.730 majority baseline and 0.27
prevalence). Gradient boosting won cross-validation (AUC 0.816) but generalised
slightly worse (0.753), a small real example of overfitting. Logistic
regression trained fastest and wins F1 per second.

![Model comparison](figures/model_comparison.png)

*Figure 10. The three classifiers on the held-out future days.*

![ROC and precision-recall curves](figures/roc_pr_curves.png)

*Figure 11. ROC and precision-recall curves. PR-AUC of 0.53 is about twice the
0.27 base rate, so the models carry real signal on the minority class.*

The random forest ranks a route's history first and the number of trips in the
band second. The decision threshold that maximises minority-class F1 is 0.30,
not 0.5, a knob a regulator would tune toward recall.

![Random forest feature importance](figures/feature_importance.png)

*Figure 12. Feature importance. Route history leads; trips-per-band is second,
which matters for the equity finding below.*

![Threshold sweep](figures/threshold_sweep.png)

*Figure 13. Precision, recall and F1 for the compliant class across decision
thresholds. F1 peaks at 0.30.*

The two deprivation gradients point in opposite directions. Threshold
compliance is slightly better in affluent areas (Spearman rho +0.093,
permutation p = 0.033, n = 522 routes). Raw punctuality is better in deprived
areas (rho -0.227, p < 0.001).

![The two deprivation gradients](figures/info_equity_panels.png)

*Figure 14. The same x-axis, opposite stories: compliance rises with affluence
(left), while raw on-time rate falls with it (right).*

The mechanism is the metric. Bands with one or two trips clear the 85% bar 46%
of the time; bands with twenty or more clear it 7%, while frequency is
uncorrelated with actual punctuality (rho -0.007). Deprived areas run the
frequent services (rho -0.216). The random forest finding trips-per-band as its
second feature confirms the same effect independently. The recommendation a
regulator can act on: stratify the compliance standard by service frequency
before using it for enforcement.

![Compliance falls with frequency while punctuality stays flat](figures/info_frequency_confound.png)

*Figure 15. The core finding. As scheduled trips per band rise, the share of
"compliant" band-days collapses from 46% to 7% (bars), yet the mean on-time
rate of the trips barely moves (line). The threshold measures frequency, not
punctuality.*

![Compliance by deprivation decile](figures/compliance_by_imd.png)

*Figure 16. Share of route-band-days meeting the 85% bar by IMD decile
(1 = most deprived).*

![Reliability map of West Yorkshire](figures/reliability_map.png)

*Figure 17. 12,393 stops coloured by measured on-time rate (green at or above
85%, orange 60 to 85%, red below 60%). Red dominates, matching the 26.8%
headline; the interactive version is `figures/reliability_map.html`.*

## 8. Critical reflection

The AVL floor is the largest caveat: 20.9% of scheduled trips left no trace,
and I cannot separate cancellations from silent vehicles. The 60-second poll
cadence blurs passing times by up to a minute, and 2.5% of pings carry no trip
identifier and are unmatched. Equity claims stay at area level; nothing here
says anything about individuals (Robinson 1950). The data is operational rather
than personal under the Open Government Licence, but vehicle traces could in
principle profile drivers, so all analysis stays at trip level or above. There
is also an ethics point in the main finding: a frequency-biased metric,
enforced naively, would penalise the operators serving deprived areas. Working
on 8 GB of memory forced the distributed design rather than decorating it.

## 9. Conclusion

The project delivers a reproducible big-data pipeline from live national feeds
to a tested equity finding, three compared models with honest evaluation, and
evidence for every optimisation claim. The result that matters most is
methodological. West Yorkshire's buses broadly fail the 85% standard, but the
standard itself measures frequency as much as punctuality. Future work: a
longer window, calendar exception handling, and a spatial fallback for
unidentified pings.

## References

Daganzo, C.F. (2009) 'A headway-based approach to eliminate bus bunching:
systematic analysis and comparisons', *Transportation Research Part B:
Methodological*, 43(10), pp. 913-921. doi:10.1016/j.trb.2009.04.002.

Furth, P.G. and Muller, T.H.J. (2006) 'Service reliability and hidden waiting
time: insights from automatic vehicle location data', *Transportation Research
Record*, 1955(1), pp. 79-87. doi:10.1177/0361198106195500110.

Lucas, K. (2012) 'Transport and social exclusion: where are we now?',
*Transport Policy*, 20, pp. 105-113. doi:10.1016/j.tranpol.2012.01.013.

Robinson, W.S. (1950) 'Ecological correlations and the behavior of
individuals', *American Sociological Review*, 15(3), pp. 351-357.
doi:10.2307/2087176.

Data sources: Department for Transport, Bus Open Data Service, Open Government
Licence v3.0; Ministry of Housing, Communities and Local Government, English
Indices of Deprivation 2019; Office for National Statistics, LSOA 2011
boundaries; Open-Meteo historical weather API. All accessed July 2026.
