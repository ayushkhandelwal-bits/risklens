import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def _db_ok() -> bool:
    from common.db import ping
    return ping()


requires_db = pytest.mark.skipif(not _db_ok(), reason="PostgreSQL with loaded RiskLens data not available")
