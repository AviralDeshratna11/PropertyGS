"""MARLIN Orchestrator: LLM-guided high-level planning for negotiations.

This module provides a lightweight orchestration layer that uses an LLM
to generate a human-readable negotiation plan and transforms that plan
into guidance signals for the MAPPO agents. For now this is a safe
mocked implementation that can be replaced with real LLM calls.
"""
from typing import Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger("propos.marlin")


def generate_plan(buyer_config: Dict[str, Any], seller_config: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    """Return a high-level plan and recommended strategies for each agent.

    The real implementation would call an LLM (Anthropic/OpenAI) to
    produce a structured plan with explanations. Here we return a
    deterministic plan that demonstrates the structure.
    """
    fmv = market.get("property_fair_value_usd")
    plan = {
        "summary": f"Target midpoint strategy around ${fmv}",
        "buyer_strategy": {
            "open_with": int(buyer_config.get("target_price_usd", fmv) * 0.9),
            "concede_profile": "slow",
            "explainable_reasoning": "Preserve surplus while signaling seriousness",
        },
        "seller_strategy": {
            "open_with": int(seller_config.get("target_price_usd", fmv) * 1.05),
            "concede_profile": "moderate",
            "explainable_reasoning": "Anchor high but allow structured concessions",
        },
        "llm_text": (
            "Plan: buyer opens slightly below target, seller anchors above target. "
            "Both parties prioritize quick progress; buyer signals willingness to close if inspection passes."
        ),
    }

    logger.debug("Generated MARLIN plan: %s", plan)
    return plan
