"""
Central configuration for RiskLens.

Everything that is a *business assumption* or a *threshold* lives here so it is
explicit, documented and easy to change. Nothing in this file is observed data.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://risklens:risklens@localhost:5432/risklens")
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", ROOT / "data"))   # Kaggle CSVs (also searched in data/raw, data/fresh)
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "ml" / "artifacts"))
API_URL = os.getenv("API_URL", "http://localhost:8000")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# LLM (AI Risk Analyst). Provider-agnostic: "anthropic" or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty -> provider default (see ai/llm.py)

# --------------------------------------------------------------------------
# ETL / sampling
# The full Home Credit dataset is ~2.7 GB / 58M rows. RiskLens works on a
# reproducible random sample of applicants and keeps *all* of their related
# records so referential integrity is preserved.
# --------------------------------------------------------------------------
SAMPLE_PORTFOLIO = int(os.getenv("SAMPLE_PORTFOLIO", 50000))   # from application_train (has outcomes)
SAMPLE_INTAKE = int(os.getenv("SAMPLE_INTAKE", 12000))         # from application_test (recent intake, no outcomes)
RANDOM_SEED = 42

# Simulated booking vintages (ASSUMPTION — Home Credit has no calendar dates).
# Applicants are ordered by SK_ID_CURR and split into equal-sized quarters.
VINTAGES = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"]
INTAKE_VINTAGE = "2026-Q1"  # application_test applicants = most recent intake

# --------------------------------------------------------------------------
# Expected Loss assumptions (NOT observed in Home Credit)
# EL = PD x LGD x EAD
# --------------------------------------------------------------------------
LGD_ASSUMPTIONS = {
    "Cash loans": 0.45,       # Basel foundation-IRB reference for senior unsecured exposures
    "Revolving loans": 0.65,  # unsecured revolving lines typically recover less
}
CCF_REVOLVING = 0.75          # credit conversion factor applied to revolving limits to estimate EAD

# --------------------------------------------------------------------------
# Risk tiers (on model PD)
# --------------------------------------------------------------------------
PD_TIERS = [  # (upper bound exclusive, label) — portfolio base default rate is ~8%
    (0.05, "Low"),        # well below base rate
    (0.10, "Medium"),     # around base rate
    (0.20, "High"),       # >1.25x base rate
    (1.01, "Very High"),  # >2.5x base rate
]
HIGH_RISK_TIERS = ("High", "Very High")

# --------------------------------------------------------------------------
# Early Warning System
# --------------------------------------------------------------------------
EWS_BANDS = [(30, "Low"), (60, "Medium"), (80, "High"), (100, "Critical")]
EWS_SEVERITY_POINTS = {"CRITICAL": 35, "HIGH": 20, "MEDIUM": 10}
EWS_ALERT_THRESHOLD = 31  # Medium and above counts as an active early warning

# --------------------------------------------------------------------------
# Behavioural anomaly detection
# --------------------------------------------------------------------------
ANOMALY_WEIGHT_MODEL = 0.6   # Isolation Forest percentile
ANOMALY_WEIGHT_RULES = 0.4   # rule-trigger score
ANOMALY_BANDS = [(60, "Normal"), (80, "Watch"), (100, "Suspicious")]
ISOLATION_CONTAMINATION = 0.03

# --------------------------------------------------------------------------
# Investigation queue priority weights
# --------------------------------------------------------------------------
PRIORITY_WEIGHTS = {"pd": 0.30, "expected_loss": 0.25, "ews": 0.25, "anomaly": 0.20}

# --------------------------------------------------------------------------
# Model monitoring (Population Stability Index)
# --------------------------------------------------------------------------
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25
MODEL_VERSION = "1.0"


def pd_tier(p: float) -> str:
    for bound, label in PD_TIERS:
        if p < bound:
            return label
    return PD_TIERS[-1][1]


def band(score: float, bands) -> str:
    for upper, label in bands:
        if score <= upper:
            return label
    return bands[-1][1]


# Display order for ordinal dimensions (used by API sorting)
DIMENSION_ORDER = {
    "income_band": ["<100K", "100K–150K", "150K–225K", "225K–300K", "300K+"],
    "risk_tier": ["Low", "Medium", "High", "Very High"],
    "ews_band": ["Low", "Medium", "High", "Critical"],
    "anomaly_status": ["Normal", "Watch", "Suspicious"],
}
