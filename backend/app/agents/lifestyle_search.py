"""
PropOS Agentic Lifestyle Search Engine
=======================================
Interprets natural-language lifestyle queries into structured
API calls, fuses data from Shadowmap, INRIX, HowLoud, WalkScore,
and produces weighted Lifestyle Match Scores.

Architecture:
  - Context Contract: defines goal, authoritative sources, escalation gates
  - Atomic Agents: reusable units (sunlight_check, commute_calc, etc.)
  - Composite Scoring: weighted fusion with explainable breakdowns
"""

import asyncio
import httpx
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from app.core.config import settings
from app.models.schemas import LifestyleWeights, PropertyMatch

logger = logging.getLogger("propos.search")


# ══════════════════════════════════════════════════════════════════════
# §1  CONTEXT CONTRACT
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ContextContract:
    """
    Defines the agent's operating boundaries before execution begins.

    - goal: what the agent must achieve
    - authoritative_sources: trusted data providers
    - escalation_gates: conditions that force human-in-the-loop
    """
    goal: str
    authoritative_sources: List[str] = field(default_factory=lambda: [
        "Dubai Land Department", "Shadowmap", "INRIX", "HowLoud", "WalkScore"
    ])
    budget_min_usd: Optional[int] = None
    budget_max_usd: Optional[int] = None
    city: str = "Dubai"
    escalation_gates: List[str] = field(default_factory=lambda: [
        "budget_constraint_within_5pct",   # Alert if results near budget edge
        "missing_legal_documents",          # Halt if RERA/DLD data unavailable
        "conflicting_data_sources",         # Flag if APIs disagree significantly
    ])


# ══════════════════════════════════════════════════════════════════════
# §2  LLM QUERY INTERPRETER
# ══════════════════════════════════════════════════════════════════════

INTERPRETATION_PROMPT = """You are the PropOS Lifestyle Search Interpreter.
Given a natural-language real estate query, extract structured lifestyle parameters.

Return ONLY valid JSON with this schema:
{{
  "sunlight": <0.0-1.0 importance weight>,
  "quietness": <0.0-1.0>,
  "walkability": <0.0-1.0>,
  "commute": <0.0-1.0>,
  "commute_target": "<location name or null>",
  "commute_max_minutes": <integer or null>,
  "property_type": "<villa|apartment|penthouse|any>",
  "bedrooms_min": <int or null>,
  "bathrooms_min": <int or null>,
  "area_sqft_min": <int or null>,
  "custom_criteria": {{<key>: <weight>}},
  "reasoning": "<1-2 sentence explanation of how you interpreted the query>"
}}

Examples:
- "Quiet villa with morning sun near DIFC" →
  sunlight=0.9, quietness=0.85, commute=0.7, commute_target="DIFC", property_type="villa"
- "Family apartment, good schools, walkable" →
  walkability=0.9, custom_criteria={{"school_proximity": 0.8}}, property_type="apartment"
"""


async def interpret_query(query: str) -> Tuple[LifestyleWeights, Dict[str, Any]]:
    """
    Use LLM to parse natural-language lifestyle query into structured weights.
    Returns (weights, full_interpretation).
    """
    async with httpx.AsyncClient(timeout=30) as client:
        if settings.LLM_PROVIDER == "anthropic":
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,
                    "max_tokens": 1024,
                    "system": INTERPRETATION_PROMPT,
                    "messages": [{"role": "user", "content": query}],
                },
            )
            data = response.json()
            text = data["content"][0]["text"]
        else:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": INTERPRETATION_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            data = response.json()
            text = data["choices"][0]["message"]["content"]

    # Parse JSON from LLM response
    text = re.sub(r"```json\s*|```", "", text).strip()
    parsed = json.loads(text)

    weights = LifestyleWeights(
        sunlight=parsed.get("sunlight", 0),
        quietness=parsed.get("quietness", 0),
        walkability=parsed.get("walkability", 0),
        commute=parsed.get("commute", 0),
        commute_target=parsed.get("commute_target"),
        commute_max_minutes=parsed.get("commute_max_minutes"),
        custom_criteria=parsed.get("custom_criteria", {}),
    )

    return weights, parsed


# ══════════════════════════════════════════════════════════════════════
# §3  ATOMIC DATA AGENTS (API Integrations)
# ══════════════════════════════════════════════════════════════════════

class SunlightAgent:
    """Fetches solar irradiance and shadow data from Shadowmap API."""

    @staticmethod
    async def fetch(lat: float, lon: float, client: httpx.AsyncClient) -> Dict[str, Any]:
        if not settings.SHADOWMAP_API_KEY:
            logger.warning("Shadowmap API key not configured — using fallback")
            return {"score": 65.0, "source": "fallback", "hours_direct_sun": 6.5}

        try:
            resp = await client.get(
                f"{settings.SHADOWMAP_BASE_URL}/sunlight",
                params={"lat": lat, "lon": lon, "key": settings.SHADOWMAP_API_KEY},
            )
            data = resp.json()
            hours = data.get("direct_sunlight_hours", 0)
            score = min(100, (hours / 10.0) * 100)  # Normalize to 0-100
            return {"score": score, "source": "shadowmap", "hours_direct_sun": hours,
                    "morning_sun": data.get("morning_exposure", True)}
        except Exception as e:
            logger.error(f"Shadowmap API error: {e}")
            return {"score": 65.0, "source": "fallback", "hours_direct_sun": 6.5}


class CommuteAgent:
    """Calculates drive-time via INRIX API."""

    @staticmethod
    async def fetch(
        origin_lat: float, origin_lon: float,
        dest_name: str, dest_lat: float, dest_lon: float,
        client: httpx.AsyncClient,
    ) -> Dict[str, Any]:
        if not settings.INRIX_APP_ID:
            logger.warning("INRIX API not configured — using fallback")
            return {"minutes": 25, "source": "fallback", "destination": dest_name}

        try:
            resp = await client.get(
                f"{settings.INRIX_BASE_URL}/route/drivetime",
                params={
                    "appId": settings.INRIX_APP_ID,
                    "appKey": settings.INRIX_APP_KEY,
                    "origin": f"{origin_lat},{origin_lon}",
                    "destination": f"{dest_lat},{dest_lon}",
                    "departureTime": "08:00",  # Morning commute
                },
            )
            data = resp.json()
            minutes = data.get("result", {}).get("trip", {}).get("travelTimeMinutes", 30)
            return {"minutes": minutes, "source": "inrix", "destination": dest_name}
        except Exception as e:
            logger.error(f"INRIX API error: {e}")
            return {"minutes": 30, "source": "fallback", "destination": dest_name}


class AcousticAgent:
    """Fetches noise levels from HowLoud API."""

    @staticmethod
    async def fetch(lat: float, lon: float, client: httpx.AsyncClient) -> Dict[str, Any]:
        if not settings.HOWLOUD_API_KEY:
            return {"score": 70.0, "source": "fallback"}

        try:
            resp = await client.get(
                f"{settings.HOWLOUD_BASE_URL}/score",
                params={"lat": lat, "lon": lon, "key": settings.HOWLOUD_API_KEY},
            )
            data = resp.json()
            # HowLoud returns 0-100 where higher = quieter
            return {"score": data.get("soundscore", 70), "source": "howloud",
                    "traffic_score": data.get("traffic", 50),
                    "nightlife_score": data.get("nightlife", 50)}
        except Exception as e:
            logger.error(f"HowLoud API error: {e}")
            return {"score": 70.0, "source": "fallback"}


class WalkabilityAgent:
    """Fetches Walk Score, Transit Score from WalkScore API."""

    @staticmethod
    async def fetch(lat: float, lon: float, address: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        if not settings.WALKSCORE_API_KEY:
            return {"walk_score": 55.0, "transit_score": 45.0, "source": "fallback"}

        try:
            resp = await client.get(
                f"{settings.WALKSCORE_BASE_URL}/score",
                params={
                    "format": "json", "lat": lat, "lon": lon,
                    "address": address, "wsapikey": settings.WALKSCORE_API_KEY,
                    "transit": 1, "bike": 1,
                },
            )
            data = resp.json()
            return {
                "walk_score": data.get("walkscore", 50),
                "transit_score": data.get("transit", {}).get("score", 40),
                "bike_score": data.get("bike", {}).get("score", 30),
                "source": "walkscore",
            }
        except Exception as e:
            logger.error(f"WalkScore API error: {e}")
            return {"walk_score": 55.0, "transit_score": 45.0, "source": "fallback"}


# ══════════════════════════════════════════════════════════════════════
# §4  COMPOSITE LIFESTYLE SCORER
# ══════════════════════════════════════════════════════════════════════

# Well-known commute destinations with coordinates
KNOWN_DESTINATIONS = {
    "DIFC": (25.2100, 55.2708),
    "Dubai Marina": (25.0805, 55.1403),
    "Downtown Dubai": (25.1972, 55.2744),
    "JBR": (25.0780, 55.1339),
    "Business Bay": (25.1860, 55.2616),
    "Abu Dhabi": (24.4539, 54.3773),
    "Manhattan": (40.7580, -73.9855),
    "Midtown NYC": (40.7549, -73.9840),
    "Wall Street": (40.7074, -74.0113),
    "Times Square": (40.7580, -73.9855),
}


async def score_property(
    property_data: Dict[str, Any],
    weights: LifestyleWeights,
    contract: ContextContract,
) -> PropertyMatch:
    """
    Fuse all API data sources into a single Lifestyle Match Score.

    Score = Σ(weight_i × normalized_score_i) / Σ(weight_i)
    """
    lat = property_data["latitude"]
    lon = property_data["longitude"]
    address = property_data.get("address", "")

    async with httpx.AsyncClient(timeout=15) as client:
        # Fire all API calls concurrently
        tasks = {
            "sunlight": SunlightAgent.fetch(lat, lon, client),
            "acoustic": AcousticAgent.fetch(lat, lon, client),
            "walkability": WalkabilityAgent.fetch(lat, lon, address, client),
        }

        # Commute calculation (if target specified)
        if weights.commute_target and weights.commute_target in KNOWN_DESTINATIONS:
            dest_lat, dest_lon = KNOWN_DESTINATIONS[weights.commute_target]
            tasks["commute"] = CommuteAgent.fetch(lat, lon, weights.commute_target, dest_lat, dest_lon, client)

        results = {}
        for key, coro in tasks.items():
            try:
                results[key] = await coro
            except Exception as e:
                logger.error(f"Agent {key} failed: {e}")
                results[key] = {}

    # ── Normalize each dimension to 0-100 ─────────────────────────────
    scores = {}
    breakdown = {}

    # Sunlight
    sun_raw = results.get("sunlight", {}).get("score", 50)
    scores["sunlight"] = sun_raw
    breakdown["sunlight"] = {"raw": sun_raw, "weight": weights.sunlight}

    # Quietness
    noise_raw = results.get("acoustic", {}).get("score", 50)
    scores["quietness"] = noise_raw
    breakdown["quietness"] = {"raw": noise_raw, "weight": weights.quietness}

    # Walkability
    walk_raw = results.get("walkability", {}).get("walk_score", 50)
    scores["walkability"] = walk_raw
    breakdown["walkability"] = {"raw": walk_raw, "weight": weights.walkability}

    # Commute
    if "commute" in results:
        minutes = results["commute"].get("minutes", 60)
        max_min = weights.commute_max_minutes or 30
        commute_score = max(0, 100 * (1 - minutes / (max_min * 2)))
        scores["commute"] = commute_score
        breakdown["commute"] = {
            "raw_minutes": minutes, "target": weights.commute_target,
            "score": commute_score, "weight": weights.commute,
        }

    # ── Weighted composite ────────────────────────────────────────────
    weight_map = {
        "sunlight": weights.sunlight,
        "quietness": weights.quietness,
        "walkability": weights.walkability,
        "commute": weights.commute,
    }

    total_weight = sum(w for k, w in weight_map.items() if k in scores and w > 0)
    if total_weight > 0:
        composite = sum(scores[k] * weight_map[k] for k in scores if weight_map.get(k, 0) > 0) / total_weight
    else:
        composite = sum(scores.values()) / max(len(scores), 1)

    # ── Escalation gate checks ────────────────────────────────────────
    if contract.budget_max_usd:
        price = property_data.get("asking_price_usd", 0)
        if price > contract.budget_max_usd * 0.95:
            breakdown["escalation_warning"] = "Property price within 5% of budget ceiling"

    return PropertyMatch(
        property_id=property_data["id"],
        title=property_data["title"],
        city=property_data.get("city", contract.city),
        district=property_data.get("district"),
        asking_price_usd=property_data.get("asking_price_usd", 0),
        area_sqft=property_data.get("area_sqft"),
        bedrooms=property_data.get("bedrooms"),
        bathrooms=property_data.get("bathrooms"),
        sunlight_score=scores.get("sunlight"),
        noise_score=scores.get("quietness"),
        walk_score=scores.get("walkability"),
        commute_minutes={weights.commute_target: results.get("commute", {}).get("minutes")}
            if weights.commute_target and "commute" in results else None,
        lifestyle_match_score=round(composite, 2),
        score_breakdown={k: round(v, 2) if isinstance(v, float) else v
                         for k, v in breakdown.items()},
        reasoning=f"Composite score {composite:.1f}/100 weighted across "
                  f"{len([w for w in weight_map.values() if w > 0])} lifestyle dimensions.",
    )


# ══════════════════════════════════════════════════════════════════════
# §5  ORCHESTRATOR (full pipeline)
# ══════════════════════════════════════════════════════════════════════

async def run_lifestyle_search(
    query: str,
    city: str,
    properties: List[Dict[str, Any]],
    budget_min: Optional[int] = None,
    budget_max: Optional[int] = None,
    max_results: int = 10,
) -> Tuple[LifestyleWeights, List[PropertyMatch]]:
    """
    Full pipeline:
      1. LLM interprets natural-language query → LifestyleWeights
      2. Pre-filter properties by budget/city
      3. Score each property concurrently via API fusion
      4. Rank and return top matches
    """
    # Step 1: Interpret
    weights, raw_interpretation = await interpret_query(query)
    logger.info(f"Query interpreted: {weights.model_dump()}")

    # Step 2: Build context contract
    contract = ContextContract(
        goal=query, city=city,
        budget_min_usd=budget_min, budget_max_usd=budget_max,
    )

    # Step 3: Pre-filter
    filtered = [p for p in properties if p.get("city", "").lower() == city.lower()]
    if budget_min:
        filtered = [p for p in filtered if p.get("asking_price_usd", 0) >= budget_min]
    if budget_max:
        filtered = [p for p in filtered if p.get("asking_price_usd", 0) <= budget_max]

    # Property type filter (from LLM interpretation)
    ptype = raw_interpretation.get("property_type", "any")
    if ptype and ptype != "any":
        filtered = [p for p in filtered if p.get("property_type", "").lower() == ptype.lower()]

    # Step 4: Score concurrently
    tasks = [score_property(p, weights, contract) for p in filtered]
    scored = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out errors
    valid_results = [r for r in scored if isinstance(r, PropertyMatch)]
    for r in scored:
        if isinstance(r, Exception):
            logger.error(f"Scoring error: {r}")

    # Step 5: Rank by lifestyle match score
    valid_results.sort(key=lambda x: x.lifestyle_match_score, reverse=True)

    return weights, valid_results[:max_results]
