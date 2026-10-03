"""
RiskLens ETL orchestrator.

    python -m etl.run_pipeline [--raw-dir PATH] [--portfolio N] [--intake N] [--skip-source-dq]

Raw CSVs -> ingestion (sampling) -> data quality -> cleaning -> transform
-> PostgreSQL raw layer -> Customer 360 + analytical views.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from common.config import ROOT, SAMPLE_INTAKE, SAMPLE_PORTFOLIO
from common.db import execute, run_sql_file
from etl.cleaning import clean
from etl.data_quality import run_checks
from etl.ingestion import ingest
from etl.load_postgres import load_frame, load_raw
from etl.transformations import transform

log = logging.getLogger("risklens.etl")


def build_analytics() -> None:
    """(Re)build SQL-layer analytical objects that only depend on raw tables."""
    t = time.time()
    run_sql_file(ROOT / "sql" / "customer_360.sql")
    log.info("customer_360 built in %.1fs", time.time() - t)


def main(raw_dir: Path | None, n_portfolio: int, n_intake: int, source_dq: bool) -> str:
    run_id = datetime.now(timezone.utc).strftime("etl_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    started = datetime.now(timezone.utc)
    t0 = time.time()

    run_sql_file(ROOT / "sql" / "schema.sql")

    log.info("[1/5] Ingesting source systems (sample: %s portfolio + %s intake applicants)", n_portfolio, n_intake)
    frames = ingest(raw_dir, n_portfolio, n_intake)

    log.info("[2/5] Running data quality checks")
    dq = run_checks(frames, raw_dir, full_source=source_dq)
    dq.insert(0, "run_id", run_id)

    log.info("[3/5] Cleaning + transforming")
    frames, cleaning_log = clean(frames)
    tables = transform(frames)
    del frames

    log.info("[4/5] Loading PostgreSQL raw layer")
    load_raw(tables)
    load_frame("data_quality_checks", dq, if_exists="append")

    log.info("[5/5] Building Customer 360")
    build_analytics()

    counts = {k: int(len(v)) for k, v in tables.items()}
    execute(
        """INSERT INTO etl_runs (run_id, started_at, finished_at, status, sample_portfolio, sample_intake, row_counts, cleaning_log)
           VALUES (:r, :s, now(), 'success', :p, :i, CAST(:rc AS JSONB), CAST(:cl AS JSONB))""",
        {"r": run_id, "s": started, "p": n_portfolio, "i": n_intake,
         "rc": json.dumps(counts), "cl": json.dumps(cleaning_log)},
    )
    log.info("ETL run %s finished in %.1f min", run_id, (time.time() - t0) / 60)
    return run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--portfolio", type=int, default=SAMPLE_PORTFOLIO)
    ap.add_argument("--intake", type=int, default=SAMPLE_INTAKE)
    ap.add_argument("--skip-source-dq", action="store_true")
    a = ap.parse_args()
    main(a.raw_dir, a.portfolio, a.intake, not a.skip_source_dq)
