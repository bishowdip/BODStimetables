-- Star schema for the reliability warehouse (SQLite).
-- SQLite is keyless, so there are no credentials anywhere in the project.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fact_route_band_day;
DROP TABLE IF EXISTS fact_trip_delay;
DROP TABLE IF EXISTS dim_stop;
DROP TABLE IF EXISTS dim_route;
DROP TABLE IF EXISTS dim_lsoa;

CREATE TABLE dim_lsoa (
    lsoa_code   TEXT PRIMARY KEY,
    imd_decile  INTEGER,
    imd_score   REAL
);

CREATE TABLE dim_route (
    route_id          TEXT PRIMARY KEY,
    operator          TEXT,
    route_short_name  TEXT,
    n_stops           INTEGER,
    route_imd_decile  REAL
);

CREATE TABLE dim_stop (
    stop_id    TEXT PRIMARY KEY,
    name       TEXT,
    lat        REAL,
    lon        REAL,
    lsoa_code  TEXT,
    FOREIGN KEY (lsoa_code) REFERENCES dim_lsoa (lsoa_code)
);

CREATE TABLE fact_trip_delay (
    trip_id         TEXT,
    route_id        TEXT,
    service_date    TEXT,
    time_band       TEXT,
    day_of_week     TEXT,
    stops_observed  INTEGER,
    median_delay    REAL,
    on_time         INTEGER,
    PRIMARY KEY (trip_id, service_date),
    FOREIGN KEY (route_id) REFERENCES dim_route (route_id)
);

CREATE TABLE fact_route_band_day (
    route_id      TEXT,
    service_date  TEXT,
    time_band     TEXT,
    day_of_week   TEXT,
    n_trips       INTEGER,
    pct_on_time   REAL,
    sd_trip_delay REAL,
    compliant     INTEGER,
    PRIMARY KEY (route_id, service_date, time_band),
    FOREIGN KEY (route_id) REFERENCES dim_route (route_id)
);

-- Disruptions (SIRI-SX situations); linked to routes through affected stops
CREATE TABLE fact_disruption (
    situation_id     TEXT PRIMARY KEY,
    reason           TEXT,
    planned          INTEGER,
    progress         TEXT,
    summary          TEXT,
    validity_start   TEXT,
    validity_end     TEXT,
    n_affected_stops INTEGER,
    wy_specific      INTEGER
);

CREATE TABLE disruption_stop (
    situation_id TEXT,
    stop_id      TEXT,
    PRIMARY KEY (situation_id, stop_id),
    FOREIGN KEY (situation_id) REFERENCES fact_disruption (situation_id),
    FOREIGN KEY (stop_id) REFERENCES dim_stop (stop_id)
);

CREATE INDEX idx_rbd_route ON fact_route_band_day (route_id);
CREATE INDEX idx_trip_route ON fact_trip_delay (route_id);
CREATE INDEX idx_stop_lsoa ON dim_stop (lsoa_code);
CREATE INDEX idx_disruption_stop ON disruption_stop (stop_id);
