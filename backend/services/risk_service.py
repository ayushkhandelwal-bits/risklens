"""Risk Engine service — customer risk, SHAP explanations, Expected Loss, model registry."""
from __future__ import annotations

from backend.schemas.filters import FILTER_COLUMNS, PortfolioFilters
from backend.services import portfolio_service as ps
from backend.services.customer_service import CustomerNotFound, normalise_id
from common.config import CCF_REVOLVING, LGD_ASSUMPTIONS, PD_TIERS
from common.db import read_sql


def _rec(df):
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def customer_risk(customer_id: str) -> dict:
    cid, sk = normalise_id(customer_id)
    df = read_sql("""SELECT r.customer_id, r.pd, r.pd_challenger, r.risk_score, r.risk_tier, r.lgd, r.ead,
                            r.expected_loss, r.dataset_split, r.model_name, r.model_version, r.scored_at::text AS scored_at,
                            c.product, c.amt_credit, c.current_exposure, c.default_flag
                     FROM risk_scores r JOIN customer_360 c USING (sk_id_curr) WHERE sk_id_curr = :sk""", {"sk": sk})
    if df.empty:
        if read_sql("SELECT 1 FROM customer_360 WHERE sk_id_curr = :sk", {"sk": sk}).empty:
            raise CustomerNotFound(f"Customer {cid} not found")
        raise FileNotFoundError("customer has not been scored — run `python -m ml.train`")
    r = _rec(df)[0]
    r["el_formula"] = (f"EL = PD {r['pd']:.4f} × LGD {r['lgd']:.2f} × EAD {r['ead']:,.0f} = {r['expected_loss']:,.0f}")
    r["provenance"] = {"pd": "model prediction (champion)", "lgd": "assumption (by product)",
                       "ead": "derived from loan amount" + (f" × CCF {CCF_REVOLVING}" if r["product"] == "Revolving loans" else ""),
                       "default_flag": "observed outcome" if r["default_flag"] is not None else "not yet observed"}
    r["portfolio_pd_percentile"] = float(read_sql(
        "SELECT AVG((pd < :p)::int) AS pr FROM risk_scores", {"p": r["pd"]}).pr.iloc[0])
    return r


def explanation(customer_id: str, top_n: int = 5) -> dict:
    _, sk = normalise_id(customer_id)
    if read_sql("SELECT 1 FROM customer_360 WHERE sk_id_curr = :sk", {"sk": sk}).empty:
        raise CustomerNotFound(f"Customer C{sk} not found")
    from ml.explainability import explain_customer
    res = explain_customer(sk, top_n)
    res.pop("all", None)
    return res


def assumptions() -> dict:
    return {
        "expected_loss_formula": "EL = PD × LGD × EAD",
        "pd": "Champion XGBoost model prediction (12-month default proxy = Home Credit TARGET).",
        "lgd": {k: v for k, v in LGD_ASSUMPTIONS.items()},
        "lgd_note": "ASSUMPTION — Home Credit contains no recovery / collateral data. 45% follows the Basel "
                    "foundation-IRB reference for senior unsecured exposures; revolving set higher (65%).",
        "ead": f"Cash loans: loan amount (AMT_CREDIT). Revolving: limit × CCF {CCF_REVOLVING} (ASSUMPTION).",
        "risk_tiers": [{"tier": lab, "pd_upper_bound": b} for b, lab in PD_TIERS],
        "risk_score": "0–100 percentile of PD within the development population (100 = riskiest).",
    }


def expected_loss_by(dimension: str, f: PortfolioFilters) -> list[dict]:
    if dimension not in FILTER_COLUMNS:
        raise ValueError(f"Unknown dimension '{dimension}'")
    rows = ps.breakdown(dimension, f)
    for r in rows:
        r["el_rate"] = (r["expected_loss"] / r["exposure"]) if r.get("expected_loss") and r["exposure"] else None
    return rows


def model_registry() -> list[dict]:
    df = read_sql("""SELECT model_name, model_version, role, algorithm, training_date::text AS training_date,
                            features, hyperparameters, metrics, curves, train_rows, test_rows, artifact_path, is_active
                     FROM model_registry ORDER BY role, training_date DESC""")
    return _rec(df)


def feature_importance(population: str = "portfolio", limit: int = 20) -> list[dict]:
    from ml.feature_engineering import FEATURES
    df = read_sql("""SELECT feature, mean_abs_shap, mean_shap FROM global_feature_importance
                     WHERE population = :p ORDER BY mean_abs_shap DESC LIMIT :l""", {"p": population, "l": limit})
    df["label"] = df.feature.map(lambda f: FEATURES.get(f, (f,))[0])
    df["group"] = df.feature.map(lambda f: FEATURES.get(f, ("", "Other"))[1])
    return _rec(df)


def driver_analysis(f: PortfolioFilters) -> list[dict]:
    """Group-level SHAP contribution for the filtered population vs the whole book."""
    from ml.feature_engineering import FEATURES
    where, params = ps._where(f, alias="p")
    df = read_sql(f"""
        SELECT e.feature, COUNT(*) FILTER (WHERE e.direction='increases_risk') AS customers_up,
               COUNT(*) FILTER (WHERE e.direction='decreases_risk') AS customers_down,
               AVG(e.shap_value) AS avg_shap
        FROM customer_explanations e JOIN v_portfolio p USING (sk_id_curr) {where}
        GROUP BY e.feature""", params)
    n = ps.summary(f)["customers"] or 1
    df["share_up"] = df.customers_up / n
    df["label"] = df.feature.map(lambda x: FEATURES.get(x, (x,))[0])
    df["group"] = df.feature.map(lambda x: FEATURES.get(x, ("", "Other"))[1])
    return _rec(df.sort_values("customers_up", ascending=False))
