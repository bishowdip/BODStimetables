# Database schema

Star schema in SQLite. Export this to `schema.png` for the report. Full DDL is in
`src/db/schema.sql`.

```
        dim_lsoa                         dim_route
   +----------------+              +----------------------+
   | lsoa_code  PK  |              | route_id        PK   |
   | imd_decile     |              | operator             |
   | imd_score      |              | route_short_name     |
   +-------+--------+              | n_stops              |
           |                       | route_imd_decile     |
           | 1                     +----------+-----------+
           |                                  | 1
           | *                                |
   +-------v--------+                         | *
   |   dim_stop     |        +----------------v-----------------+
   | stop_id    PK  |        |        fact_trip_delay           |
   | name           |        | trip_id, service_date     PK     |
   | lat, lon       |        | route_id                  FK     |
   | lsoa_code  FK  |        | time_band, day_of_week           |
   +----------------+        | stops_observed, median_delay     |
                             | on_time                          |
                             +----------------------------------+

                             +----------------------------------+
                             |     fact_route_band_day  (ML)    |
                             | route_id, service_date,          |
                             |   time_band               PK     |
                             | route_id                  FK     |
                             | day_of_week, n_trips             |
                             | pct_on_time, sd_trip_delay       |
                             | compliant   (>=85% on time)      |
                             +----------------------------------+
```

`fact_route_band_day` is the modelling grain (route x time-band x service-day);
`compliant` is the target. All inserts and queries use `?`-parameterised SQL
(`src/db/load_db.py`, `src/db/queries.py`).
