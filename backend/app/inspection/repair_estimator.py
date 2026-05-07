"""AI Repair Estimator: estimates repair cost and 5-year yield impact for defects.

This is a mocked estimator that can be replaced by a trained model using BIM
and historical cost databases.
"""
from typing import List, Dict


def estimate_repairs(defects: List[Dict], property_value: int) -> Dict:
    total_cost = 0
    details = []
    for d in defects:
        severity = d.get('severity', 'medium')
        if severity == 'low': cost = 200
        elif severity == 'medium': cost = 1500
        elif severity == 'high': cost = 8000
        else: cost = 20000
        total_cost += cost
        details.append({"defect_id": d.get('defect_id'), "estimated_cost_usd": cost})

    # naive yield impact: repair cost amortized over 5 years vs property value
    amortized = total_cost / 5.0
    yield_impact_pct = (amortized / max(property_value, 1)) * 100

    return {
        "total_estimated_cost_usd": total_cost,
        "annual_amortized_cost_usd": round(amortized, 2),
        "five_year_yield_impact_pct": round(yield_impact_pct, 4),
        "details": details,
    }
