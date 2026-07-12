# Bus Reliability and Deprivation in West Yorkshire

**Module:** ST5011CEM Big Data Programming Project
**Author:** Bishow Dip · **Supervisor:** Mr. Siddhartha Neupane
**Code:** https://github.com/bishowdip/BODStimetables

> Draft for the 2000-word report. Figures referenced by file live in `docs/`
> and `docs/figures/`. Word counts per section are noted so trimming is easy.

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
average, more punctual. *(~150 words)*

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
of trips within ±2 minutes) is the prediction target. Headway Regularity
(standard deviation of the gap between buses, against a 20%-of-schedule bar)
and Travel Time Variability (coefficient of variation of trip duration,
against 15%) are computed per route. Service Efficiency is reported as the
share of scheduled trips confirmed by live data. Algorithmic Efficiency is
documented with stage timings and Spark UI evidence, and Model Efficiency as
F1 per second of training. Figure `info_metric_tiles.png` summarises all six
on the study week.

The work addresses learning outcomes B1 (complexity analysis of the matching
join), B2 (a working multi-stage system), B4 (large-scale data and machine
learning), B6 (version control, parameterised SQL, ethics), B7 (this report
and the viva) and B8 (the equity analysis). *(~300 words)*

## 3. Related work

Furth and Muller (2006) showed that automatic vehicle location data changes
how reliability should be measured, because unreliable service costs
passengers budgeted waiting time rather than average waiting time. Daganzo
(2009) treats bus bunching as a control problem: once headways destabilise
they collapse, which motivates measuring headway regularity separately from
lateness. Lucas (2012) reviews the evidence that transport disadvantage
concentrates on low-income communities. Robinson (1950) is the standard
caution for area-level analysis, and this project keeps every equity claim at
LSOA level for that reason. *(~100 words)*

## 4. Data collection and preprocessing

Six sources were used, all real and all open. BODS supplies three: the
Yorkshire GTFS timetable (a 109 MB zip), the GTFS-RT vehicle position feed,
and the SIRI-SX disruptions feed. The English Indices of Deprivation 2019
give an IMD decile per LSOA, ONS boundaries locate the 1,767 West Yorkshire
LSOAs, and Open-Meteo provides hourly weather without a key.

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

Combined, the datasets total over seven million records, roughly seventy
times the 100,000-record requirement, met entirely with real multi-catalogue,
multi-day and supplementary ingestion. No synthetic data was used. Figure
`info_data_funnel.png` traces the volumes through the pipeline. *(~300 words)*

## 5. Methodology

Matching joins each position to its trip's scheduled stops (trip identifier
plus service day), computes the haversine distance to every stop, and keeps
the nearest ping within 50 metres per trip, day and stop. Its timestamp is the
inferred passing time and the difference from schedule is the delay. This
yielded 1,698,693 matched stop events across 34,077 distinct trips. With the
small stops table broadcast, the join costs roughly O(P log S) rather than the
naive P × S comparison; the full complexity table is in
`docs/architecture.md`.

A trip counts as on time when its median stop delay sits within ±2 minutes
(the median resists GPS jumps). Trips aggregate to route × time-band × day
rows, compliant when at least 85% of trips were on time: 9,533 rows, 26.8%
compliant, which is the class imbalance the models face.

Every feature is known before the trip runs: day of week, time band, weekend
flag, stop count, scheduled trips in the band, scheduled headway mean and
deviation, weather, route IMD decile, active disruptions, and the route's
historical on-time rate computed from training days only. A unit test
recomputes that historical feature independently and requires agreement to
within 1e-6, and asserts that every training date precedes every test date.

The split is time-ordered because the task is forecasting: train on 2 to 6
July (6,592 rows), test on 7 and 8 July (2,941 rows). A random k-fold would
let the model see the future. CrossValidator tunes hyperparameters with
3-fold validation inside the training block for logistic regression (the
interpretable baseline), random forest and gradient-boosted trees, sharing
one feature pipeline. *(~300 words)*

## 6. System design and implementation

The system is four layers with Parquet as the hand-off between them, so any
stage re-runs alone (Figure `architecture.png`). Collectors in plain Python
poll the live feeds; Spark owns everything from parsing output to features;
SQLite stores the star schema (Figure `schema.png`); pandas and matplotlib
touch only final aggregates. Tool choice is justified per stage in
`docs/architecture.md`, including the one deliberate exception: the
stop-to-LSOA spatial join runs in GeoPandas because 16,139 stops is not a
big-data problem.

Spark evidence: eight shuffle partitions throughout; broadcasting the stops
table cut the matching join from 8.98 to 4.37 seconds (2.05×, identical
2,602,507 output rows); the cached matched-events table feeds three
aggregations. Figure `spark_ui_stages.png` shows the real join with 8/8 tasks
per stage, shuffle reads up to 165 MiB and nine skipped stages where Spark
reused earlier shuffle output rather than recomputing, which is lazy
evaluation visible in practice.

Security follows the brief: every SQL statement uses ?-placeholders, and the
only credential (the BODS key) lives in a git-ignored .env file read at
runtime. The repository carries pinned dependencies, tests and a README that
reproduces the pipeline from a fresh machine.

On an 8 GB laptop the memory question is not academic. A national week of
positions cannot sit in pandas; the parser handles one snapshot at a time and
Spark spills partitions to disk beyond that. *(~250 words)*

## 7. Results and evaluation

**Reliability.** Half of matched stop events (50.3%) were within ±2 minutes;
the median trip delay was +1.05 minutes with a long late tail (90th
percentile +7.3). At the standard's grain, 26.8% of route-band-days were
compliant. Widening the window raises the share smoothly (44.2% at ±3, 68.3%
at ±5, Figure `info_sensitivity.png`), so the headline is not an artefact of
the two-minute choice. Headway regularity was met on 7.8% of band-days and
the travel-time bar on 13.6% of routes (median CV 0.277). The AVL-confirmed
floor was 79.1% of the week's 100,100 scheduled trips, lower on the gapped
Sunday (Figure `info_avl_by_day.png`); a trip without a trace is unconfirmed,
not proven cancelled.

**Models.** On the unseen future days the random forest performed best
(accuracy 0.769, F1 0.746, ROC-AUC 0.763, PR-AUC 0.534, against a 0.730
majority baseline and 0.27 prevalence). Gradient boosting won cross-validation
(AUC 0.816) but generalised slightly worse (0.753), a small, real example of
overfitting. Logistic regression trained fastest and wins F1 per second
(0.0496 against the forest's 0.0370 and GBT's 0.0136). Figures
`model_comparison.png` and `roc_pr_curves.png` show the comparison; the
decision threshold that maximises minority-class F1 is 0.30, not 0.5
(Figure `threshold_sweep.png`), a knob a regulator would tune toward recall.

**Equity.** The two deprivation gradients point in opposite directions
(Figure `info_equity_panels.png`). Threshold compliance is slightly better in
affluent areas (Spearman rho +0.093, permutation p = 0.033, n = 522 routes;
deciles 1–3 average 25.9% against 30.6% for deciles 8–10). Raw punctuality is
better in deprived areas (rho −0.227, p < 0.001). The mechanism is the
metric: bands with one or two trips clear the 85% bar 46% of the time, bands
with twenty or more clear it 7%, while frequency is uncorrelated with actual
punctuality (rho −0.007, Figure `info_frequency_confound.png`). Deprived
areas run the frequent services (rho −0.216). The random forest ranked trips
per band its second feature (0.171) behind route history (0.299), confirming
the same effect from a different direction. The practical recommendation:
stratify the compliance standard by service frequency before using it for
enforcement. *(~350 words)*

## 8. Critical reflection

The AVL floor is the largest caveat: 20.9% of scheduled trips left no trace,
and I cannot separate cancellations from silent vehicles. The 60-second poll
cadence blurs passing times by up to a minute, and 2.5% of pings carry no
trip identifier and are unmatched. Equity claims stay at area level; nothing
here says anything about individuals (Robinson 1950). The data is
operational rather than personal under the Open Government Licence, but
vehicle traces could in principle profile drivers, so all analysis stays at
trip level or above. There is also an ethics point in the main finding: a
frequency-biased metric, enforced naively, would penalise exactly the
operators serving deprived areas. Working on 8 GB of memory forced the
distributed design rather than decorating it. *(~150 words)*

## 9. Conclusion

The project delivers what it set out to: a reproducible big-data pipeline
from live national feeds to a tested equity finding, three compared models
with honest evaluation, and evidence for every optimisation claim. The result
that matters most is methodological. West Yorkshire's buses broadly fail the
85% standard, but the standard itself measures frequency as much as
punctuality. Future work: a longer window, calendar exception handling, and
a spatial fallback for unidentified pings. *(~100 words)*

## References

Daganzo, C.F. (2009) 'A headway-based approach to eliminate bus bunching:
systematic analysis and comparisons', *Transportation Research Part B*,
43(10), pp. 913–921. doi:10.1016/j.trb.2009.04.002.

Furth, P.G. and Muller, T.H.J. (2006) 'Service reliability and hidden waiting
time: insights from automatic vehicle location data', *Transportation
Research Record*, 1955(1), pp. 79–87. doi:10.1177/0361198106195500110.

Lucas, K. (2012) 'Transport and social exclusion: where are we now?',
*Transport Policy*, 20, pp. 105–113. doi:10.1016/j.tranpol.2012.01.013.

Robinson, W.S. (1950) 'Ecological correlations and the behavior of
individuals', *American Sociological Review*, 15(3), pp. 351–357.
doi:10.2307/2087176.

Data: Department for Transport, Bus Open Data Service (OGL v3.0); MHCLG,
English Indices of Deprivation 2019; ONS, LSOA 2011 boundaries; Open-Meteo
historical weather API. Accessed July 2026.
