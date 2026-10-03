"""
Build the free-tier PUBLIC DEMO database (e.g. Neon) from your local Kaggle files.

    python -m scripts.build_public

* asks for the cloud Postgres connection string (paste it - no shell escaping needed)
* smaller sample (30,000 booked + 8,000 intake applicants) so it fits a 1 GB free tier
* drops the build-only raw_bureau_balance table afterwards (--slim)
* writes the models to ml/artifacts_public/ so your LOCAL app's models are untouched
* prints the final database size
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> None:
    url = os.getenv("PUBLIC_DATABASE_URL") or input("Paste your Neon / cloud Postgres connection string: ").strip()
    if not url or "://" not in url:
        sys.exit("That does not look like a connection string (expected postgresql://user:pass@host/db?sslmode=require).")
    os.environ["DATABASE_URL"] = url
    os.environ["MODEL_PATH"] = "ml/artifacts_public"
    os.environ.setdefault("AI_DATABASE_URL", url)   # no separate AI role on the free tier; READ ONLY txn still enforced

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from common.config import DATABASE_URL, MODEL_PATH
    from common.db import read_sql
    host = DATABASE_URL.split("@")[-1].split("/")[0]
    print(f"\nBuilding the public demo database on {host}\nModels -> {MODEL_PATH}\n")
    read_sql("SELECT 1")                       # fail fast on a bad URL / password
    from scripts.build_all import main as build
    build(skip_etl=False, portfolio=int(os.getenv("PUBLIC_PORTFOLIO", 30000)),
          intake=int(os.getenv("PUBLIC_INTAKE", 8000)), slim=True)
    size = read_sql("SELECT pg_size_pretty(pg_database_size(current_database())) AS s").s.iloc[0]
    print(f"\nPublic demo database ready - size {size}. Commit ml/artifacts_public and deploy.")


if __name__ == "__main__":
    main()
