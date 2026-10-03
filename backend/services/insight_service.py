"""
Portfolio insight generator.

Builds a short narrative from live query results: period-over-period change,
the segments contributing most to that change, behavioural movement and the
dominant model drivers. Every number in the text is computed here from the
database; the narrative structure is templated so it never invents figures.
(When an LLM is configured, the AI Risk Analyst can expand on it using tools.)
"""
from __future__ import annotations

from backend.schemas.filters import PortfolioFilters
from backend.services import portfolio_service as ps
from common.config import VINTAGES


MATERIALITY = 0.03  # relative change below 3% is reported as stable


def _seg(s: dict) -> str:
    return f"{s['segment']} ({s['dimension'].replace('_', ' ')}, {_pp(s['change'])})"


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _pp(v):
    return "n/a" if v is None else f"{v * 100:+.2f} pp"


def period_comparison(f: PortfolioFilters, metric: str = "avg_pd") -> dict:
    """Compare the most recent four booked vintages with the four before (2025 vs 2024)."""
    rows = [r for r in ps.trend(f) if r["vintage"] in VINTAGES]
    if not rows:
        return {}
    use = metric if rows[0].get(metric) is not None else "default_rate"
    prior = [r for r in rows if r["vintage"] < "2025"]
    recent = [r for r in rows if r["vintage"] >= "2025"]

    def wavg(rs, m):
        n = sum(r["customers"] for r in rs)
        return sum((r[m] or 0) * r["customers"] for r in rs) / n if n else None

    out = {"metric": use, "prior_label": "2024 vintages", "recent_label": "2025 vintages",
           "prior": wavg(prior, use), "recent": wavg(recent, use),
           "latest_vintage": rows[-1]["vintage"], "latest": rows[-1][use]}
    if out["prior"] is not None and out["recent"] is not None:
        out["change"] = out["recent"] - out["prior"]
        out["relative_change"] = out["change"] / out["prior"] if out["prior"] else None
    for m in ("avg_recent_utilisation", "avg_recent_late_rate", "high_risk_share", "early_warning_rate"):
        p, r = wavg(prior, m) if prior else None, wavg(recent, m) if recent else None
        out[m] = {"prior": p, "recent": r, "change": (r - p) if (p is not None and r is not None) else None}
    return out


def segment_contributions(f: PortfolioFilters, metric: str = "avg_pd") -> list[dict]:
    """Which segments moved most between 2024 and 2025 vintages, weighted by their size
    (contribution = change in segment metric x segment share of the recent book)."""
    results = []
    for dim in ("product", "customer_segment", "income_band", "region", "credit_score_band"):
        if len(getattr(f, dim, []) or []) == 1:   # already pinned to one value: not an explanatory split
            continue
        prior_f = f.model_copy(update={"vintage": [v for v in VINTAGES if v < "2025" and (not f.vintage or v in f.vintage)]})
        recent_f = f.model_copy(update={"vintage": [v for v in VINTAGES if v >= "2025" and (not f.vintage or v in f.vintage)]})
        if not prior_f.vintage or not recent_f.vintage:
            continue
        prior = {r["segment"]: r for r in ps.breakdown(dim, prior_f)}
        recent = {r["segment"]: r for r in ps.breakdown(dim, recent_f)}
        total_recent = sum(r["customers"] for r in recent.values()) or 1
        for seg, r in recent.items():
            p = prior.get(seg)
            m = metric if r.get(metric) is not None else "default_rate"
            if not p or p.get(m) is None or r.get(m) is None or r["customers"] < 150:
                continue
            share = r["customers"] / total_recent
            delta = r[m] - p[m]
            results.append({"dimension": dim, "segment": seg, "metric": m, "prior": p[m], "recent": r[m],
                            "change": delta, "recent_share": share, "contribution": delta * share,
                            "customers": r["customers"]})
    return sorted(results, key=lambda x: x["contribution"], reverse=True)


def behaviour_shift(f: PortfolioFilters) -> dict:
    rows = ps.behaviour_trend(f)
    if not rows:
        return {}

    MIN_N = 200  # do not narrate behaviour measured on tiny samples

    def avg(lo, hi, k):
        n_key = "card_customers_observed" if k == "utilisation" else "customers_observed"
        sel = [r for r in rows if lo <= r["month"] <= hi and r.get(k) is not None]
        if not sel or min(r.get(n_key) or 0 for r in sel) < MIN_N:
            return None
        return sum(r[k] for r in sel) / len(sel)

    out = {}
    for k in ("utilisation", "late_rate", "pos_dpd_rate"):
        early, late = avg(-24, -13, k), avg(-3, -1, k)
        out[k] = {"months_13_24_before": early, "last_3_months": late,
                  "change": (late - early) if (early is not None and late is not None) else None}
    return out


_cache: dict = {}


def portfolio_insight(f: PortfolioFilters) -> dict:
    import time
    key = f.model_dump_json()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    out = _portfolio_insight(f)
    _cache[key] = (time.time(), out)
    return out


def clear_cache() -> None:
    _cache.clear()


def _portfolio_insight(f: PortfolioFilters) -> dict:
    summ = ps.summary(f)
    comp = period_comparison(f)
    segs = segment_contributions(f)
    beh = behaviour_shift(f)
    drivers = ps.top_drivers(f, 3) if summ.get("model_scored") else []

    lines, evidence = [], []
    metric_name = "Average PD" if comp.get("metric") == "avg_pd" else "Observed default rate"
    if comp.get("change") is not None:
        rel = comp.get("relative_change") or 0
        if abs(rel) < MATERIALITY:
            lines.append(f"{metric_name} is stable across vintages: {_pct(comp['prior'])} for 2024 vintages vs "
                         f"{_pct(comp['recent'])} for 2025 ({_pp(comp['change'])}, below the {MATERIALITY:.0%} "
                         f"relative-change materiality threshold).")
            up = [s for s in segs if s["change"] > 0][:1]
            down = sorted([s for s in segs if s["change"] < 0], key=lambda s: s["contribution"])[:1]
            if up and down:
                lines.append(f"Underneath, segments offset each other: {_seg(up[0])} deteriorated while "
                             f"{_seg(down[0])} improved.")
        else:
            direction = "increased" if comp["change"] > 0 else "decreased"
            lines.append(f"{metric_name} {direction} from {_pct(comp['prior'])} in 2024 vintages to "
                         f"{_pct(comp['recent'])} in 2025 vintages ({_pp(comp['change'])}, {rel:+.1%} relative).")
            top = [s for s in segs if (s["contribution"] > 0) == (comp["change"] > 0)][:2]
            if top:
                lines.append(f"The largest size-weighted contributors are {' and '.join(_seg(s) for s in top)}.")
    u = beh.get("utilisation", {})
    lr = beh.get("late_rate", {})
    if u.get("change") is not None:
        verb = "rose" if u["change"] > 0 else "fell"
        lverb = "rose" if (lr.get("change") or 0) > 0 else "fell"
        lines.append(f"Behaviourally, card utilisation {verb} from {_pct(u['months_13_24_before'])} (13–24 months before "
                     f"application) to {_pct(u['last_3_months'])} in the last 3 months, while the late-payment rate {lverb} "
                     f"from {_pct(lr.get('months_13_24_before'))} to {_pct(lr.get('last_3_months'))}.")
    if drivers:
        names = ", ".join(d.get("label") or d["feature"] for d in drivers)
        lines.append(f"Across the selection the model's most frequent risk-increasing drivers are: {names}.")
    if summ.get("model_scored") and summ.get("high_risk_share") is not None:
        evidence.append(f"High-risk customers: {summ['high_risk_customers']:,} ({_pct(summ['high_risk_share'])}), "
                        f"high-risk exposure {ps_money(summ.get('high_risk_exposure'))}.")
    evidence.append(f"Population: {summ['customers']:,} customers · {summ['filters']}.")
    return {"headline": lines[0] if lines else "Not enough data in this selection to compare periods.",
            "lines": lines, "evidence": evidence, "period_comparison": comp,
            "segment_contributions": segs[:5], "behaviour_shift": beh, "drivers": drivers,
            "method": "Generated from live database queries (vintage comparison, size-weighted segment "
                      "contribution, behavioural trend, aggregated SHAP drivers). Vintages are simulated."}


def ps_money(v):
    if v is None:
        return "n/a"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.2f}{suf}"
    return f"{v:,.0f}"
