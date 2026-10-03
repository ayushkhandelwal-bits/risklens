"""
Build the whole RiskLens backend end-to-end:

    python -m scripts.build_all            # ETL + analytics + models + EWS + anomaly + monitoring
    python -m scripts.build_all --skip-etl # re-run everything after the raw load

Steps
  1. ETL           raw CSVs -> data quality -> PostgreSQL raw layer -> Customer 360
  2. Analytics     portfolio views, behaviour trend
  3. Risk models   train champion/challenger, register, score PD / EL, SHAP drivers
  4. Early warning rule-based EWS score + triggers (sql/early_warning.sql)
  5. Anomaly       Isolation Forest + rules -> investigation queue
  6. Monitoring    PSI / drift / performance snapshot
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings

warnings.filterwarnings("ignore")
log = logging.getLogger("risklens.build")


def slim_for_hosting() -> None:
    """Drop raw tables the running app never queries (they are only needed to BUILD
    Customer 360), so a public demo database fits small free tiers (e.g. Neon 1 GB)."""
    from common.db import execute
    execute("DROP TABLE IF EXISTS raw_bureau_balance CASCADE")
    log.info("slim mode: dropped raw_bureau_balance (aggregated into customer_360 already)")


def main(skip_etl: bool = False, portfolio: int | None = None, intake: int | None = None,
         slim: bool = False) -> None:
    from common.config import ROOT
    from common.db import run_sql_file

    t0 = time.time()
    if not skip_etl:
        from etl.run_pipeline import main as etl
        from common.config import SAMPLE_INTAKE, SAMPLE_PORTFOLIO
        log.info("== 1/6 ETL")
        etl(None, portfolio or SAMPLE_PORTFOLIO, intake or SAMPLE_INTAKE, True)
    else:
        from etl.run_pipeline import build_analytics
        run_sql_file(ROOT / "sql" / "schema.sql")
        build_analytics()
    log.info("== 2/6 analytics views")
    run_sql_file(ROOT / "sql" / "portfolio_metrics.sql")
    log.info("== 3/6 risk models + scoring")
    from ml.train import main as train
    from ml.scoring import main as score
    train()
    score()
    log.info("== 4/6 early warning")
    run_sql_file(ROOT / "sql" / "early_warning.sql")
    log.info("== 5/6 behavioural anomaly")
    from ml.anomaly import main as anomaly
    anomaly()
    log.info("== 6/6 model monitoring")
    try:
        from ml.monitoring import main as monitor
        monitor()
    except ModuleNotFoundError:
        log.info("monitoring module not present yet — skipped")
    try:   # least-privilege AI role loses grants when tables are recreated
        run_sql_file(ROOT / "sql" / "ai_readonly_grants.sql")
    except Exception as exc:
        log.warning("could not refresh AI read-only grants: %s", exc)
    if slim:
        slim_for_hosting()
    from backend.services.insight_service import clear_cache
    clear_cache()
    log.info("RiskLens build complete in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-etl", action="store_true")
    ap.add_argument("--portfolio", type=int, default=None, help="booked applicants to sample")
    ap.add_argument("--intake", type=int, default=None, help="recent-intake applicants to sample")
    ap.add_argument("--slim", action="store_true", help="drop build-only raw tables (free-tier hosting)")
    a = ap.parse_args()
    main(a.skip_etl, a.portfolio, a.intake, a.slim)
