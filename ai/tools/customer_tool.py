"""get_customer_profile — consolidated Customer 360 for one customer."""
from __future__ import annotations

from ai.tools.base import compact, tool
from backend.services import customer_service

CID = {"type": "string", "description": "Customer id, e.g. C100002 (the 'C' prefix is optional)."}


@tool("get_customer_profile",
      "Retrieve the Customer 360 profile for one customer: identity/segment, observed outcome, application and "
      "affordability, exposure, payment behaviour, credit-card utilisation, previous applications and bureau "
      "history. Use before explaining any individual customer.",
      {"type": "object", "properties": {"customer_id": CID}, "required": ["customer_id"]},
      "Retrieving customer profile...")
def get_customer_profile(customer_id: str) -> dict:
    p = customer_service.get_profile(customer_id)
    p.pop("risk", None), p.pop("early_warning", None), p.pop("anomaly", None)   # dedicated tools cover these
    return compact(p)
