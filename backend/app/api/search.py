"""Agentic Lifestyle Search API endpoints."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import LifestyleQuery, SearchResponse
from app.models.domain import Property, PropertyStatus
from app.services.db import get_db
from app.agents.lifestyle_search import run_lifestyle_search

import uuid

router = APIRouter()


@router.post("/lifestyle", response_model=SearchResponse)
async def lifestyle_search(
    query: LifestyleQuery,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Interpret a natural-language lifestyle query and return ranked property matches.

    The engine:
      1. Uses LLM to parse lifestyle parameters (sunlight, noise, commute, walkability)
      2. Fetches real-time data from Shadowmap, INRIX, HowLoud, WalkScore
      3. Computes weighted Lifestyle Match Scores
      4. Returns ranked results with full score breakdowns for transparency
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))

    # Fetch candidate properties from DB
    stmt = (
        select(Property)
        .where(Property.status == PropertyStatus.ACTIVE)
        .where(Property.city.ilike(f"%{query.city}%"))
    )
    if query.budget_min_usd:
        stmt = stmt.where(Property.asking_price_usd >= query.budget_min_usd)
    if query.budget_max_usd:
        stmt = stmt.where(Property.asking_price_usd <= query.budget_max_usd)

    result = await db.execute(stmt.limit(100))
    properties = result.scalars().all()

    # Convert ORM objects to dicts for the scoring engine
    prop_dicts = [
        {
            "id": p.id, "title": p.title, "city": p.city, "district": p.district,
            "latitude": p.latitude, "longitude": p.longitude, "address": p.address,
            "property_type": p.property_type, "bedrooms": p.bedrooms,
            "bathrooms": p.bathrooms, "area_sqft": p.area_sqft,
            "asking_price_usd": p.asking_price_usd,
        }
        for p in properties
    ]

    weights, matches = await run_lifestyle_search(
        query=query.query,
        city=query.city,
        properties=prop_dicts,
        budget_min=query.budget_min_usd,
        budget_max=query.budget_max_usd,
        max_results=query.max_results,
    )

    # Log to transparency ledger
    redis = request.app.state.redis
    await redis.log_decision(trace_id, {
        "agent": "lifestyle_search",
        "action": "query_interpreted",
        "input": query.query,
        "weights": weights.model_dump(),
        "results_count": len(matches),
    })

    return SearchResponse(
        query_interpreted=weights,
        results=matches,
        total_found=len(matches),
        trace_id=trace_id,
    )
