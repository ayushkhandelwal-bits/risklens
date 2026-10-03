from __future__ import annotations

from fastapi import APIRouter

from backend.services import audit_service, customer_service

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("/{customer_id}")
def profile(customer_id: str, log: bool = False):
    data = customer_service.get_profile(customer_id)
    if log:
        audit_service.log_event("investigation_opened", customer_id=data["identity"]["customer_id"],
                                result_summary=f"risk_tier={(data['risk'] or {}).get('risk_tier')}")
    return data


@router.get("/{customer_id}/history")
def history(customer_id: str):
    return {
        "previous_applications": customer_service.previous_applications(customer_id),
        "bureau": customer_service.bureau_history(customer_id),
        "payments": customer_service.payment_timeline(customer_id),
        "utilisation": customer_service.utilisation_timeline(customer_id),
    }
