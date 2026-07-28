#!/usr/bin/env bash
# Pre-recording smoke check. Run this once before you hit record; it confirms
# every part you will demonstrate works, without running any heavy Spark job.
# Usage:  bash tools/demo_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || { echo "!! could not activate .venv"; exit 1; }

pass() { printf "  [ok]  %s\n" "$1"; }
fail() { printf "  [!!]  %s\n" "$1"; FAILED=1; }
FAILED=0

echo "1. Environment"
python -c "import sys; assert '.venv' in sys.executable" 2>/dev/null && pass "using project .venv" || fail "NOT the project .venv"
python -c "import pyspark, pandas, streamlit, pydeck, folium" 2>/dev/null && pass "core libraries import" || fail "a library is missing (pip install -r requirements.txt)"

echo "2. Data present"
[ -f data/reliability.sqlite ] && pass "SQLite warehouse" || fail "warehouse missing (run src.db.load_db)"
[ "$(ls data/parquet 2>/dev/null | wc -l)" -gt 10 ] && pass "processed Parquet tables" || fail "parquet tables missing"
[ -d data/parquet/models/random_forest ] && pass "trained models" || fail "models missing"

echo "3. Fast pipeline parts (these you can run live)"
python -m pytest -q >/tmp/demo_pytest.log 2>&1 && pass "unit tests pass ($(grep -oE '[0-9]+ passed' /tmp/demo_pytest.log | head -1))" || fail "tests fail (see /tmp/demo_pytest.log)"
python -m src.db.queries >/tmp/demo_queries.log 2>&1 && pass "parameterised SQL queries run" || fail "queries fail (see /tmp/demo_queries.log)"
python -c "import importlib; importlib.import_module('src.viz.dashboard')" >/tmp/demo_dash.log 2>&1 && pass "dashboard imports cleanly" || fail "dashboard import error (see /tmp/demo_dash.log)"

echo "4. Report artefacts"
[ "$(ls docs/results/*.csv 2>/dev/null | wc -l)" -gt 3 ] && pass "model result files" || fail "docs/results missing (run src.ml.evaluate)"
[ "$(ls docs/figures/*.png 2>/dev/null | wc -l)" -gt 10 ] && pass "figures present" || fail "figures missing (run src.viz.eda_charts and src.viz.plots)"

echo
if [ "$FAILED" -eq 0 ]; then
  echo "ALL GOOD — you are ready to record."
else
  echo "SOME CHECKS FAILED — fix the [!!] lines above before recording."
fi