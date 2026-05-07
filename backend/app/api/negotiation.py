"""MARL Negotiation API — manages bargaining sessions."""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio, json
from app.models.schemas import NegotiationStart, NegotiationState, NegotiationRound
from app.agents.negotiation_engine import (
    NegotiationEngine, AgentConfig, MarketContext, NegotiationAction, MAPPOAgent,
)
from datetime import datetime, timezone
import uuid

router = APIRouter()

# In production, this would be initialized with trained checkpoints
_engine_cache: dict[str, MAPPOAgent] = {}


def _get_or_create_engine(role: str) -> MAPPOAgent:
    """Lazy-init a reusable MAPPO agent for the given role."""
    cache_key = f"{role}_agent"
    if cache_key not in _engine_cache:
        _engine_cache[cache_key] = MAPPOAgent(role=role)
    return _engine_cache[cache_key]


@router.post("/start")
async def start_negotiation(payload: NegotiationStart, request: Request):
    """
    Initialize a new bilateral bargaining session.

    Creates buyer and seller MARL agents with fiduciary boundaries
    and begins the Stochastic Markov Game.
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))

    buyer_config = AgentConfig(
        agent_id=payload.buyer_id,
        role="buyer",
        reserve_price_usd=payload.buyer_max_budget_usd,
        target_price_usd=int(payload.buyer_max_budget_usd * 0.85),
        urgency=payload.buyer_urgency,
    )

    seller_config = AgentConfig(
        agent_id=payload.seller_id,
        role="seller",
        reserve_price_usd=payload.seller_reserve_price_usd,
        target_price_usd=int(payload.seller_reserve_price_usd * 1.1),
        urgency=payload.seller_urgency,
    )

    # In production, FMV comes from comps analysis
    fmv = (payload.buyer_max_budget_usd + payload.seller_reserve_price_usd) // 2
    market = MarketContext(
        property_fair_value_usd=fmv,
        days_on_market=45,
        market_temperature=0.6,
        comparable_sold_prices=[int(fmv * 0.95), int(fmv * 1.02), int(fmv * 0.98)],
    )

    from app.agents.negotiation_engine import NegotiationEnvironment
    from app.agents.marlin_orchestrator import generate_plan
    session_id = str(uuid.uuid4())[:12]

    env = NegotiationEnvironment(buyer_config, seller_config, market)
    env.reset()

    # Generate an LLM-guided plan (MARLIN)
    plan = generate_plan(buyer_config.__dict__, seller_config.__dict__, market.__dict__)

    # Store session in Redis
    redis = request.app.state.redis
    await redis.set_agent_state(session_id, "env", {
        "buyer_config": {
            "agent_id": buyer_config.agent_id, "role": "buyer",
            "reserve": buyer_config.reserve_price_usd,
            "target": buyer_config.target_price_usd,
            "urgency": buyer_config.urgency,
        },
        "seller_config": {
            "agent_id": seller_config.agent_id, "role": "seller",
            "reserve": seller_config.reserve_price_usd,
            "target": seller_config.target_price_usd,
            "urgency": seller_config.urgency,
        },
        "market": {"fmv": fmv, "dom": 45, "temp": 0.6},
        "property_id": payload.property_id,
        "current_bid": env.current_bid,
        "current_ask": env.current_ask,
        "round": 0,
        "done": False,
    })

    await redis.log_decision(trace_id, {
        "agent": "negotiation",
        "action": "session_started",
        "session_id": session_id,
        "buyer_reserve": buyer_config.reserve_price_usd,
        "seller_reserve": seller_config.reserve_price_usd,
        "fmv": fmv,
    })

    # Persist the plan for transparency and human review
    await redis.cache_set(f"neg_plan:{session_id}", plan, ttl=3600)

    return {
        "session_id": session_id,
        "status": "initiated",
        "initial_bid": env.current_bid,
        "initial_ask": env.current_ask,
        "fair_market_value": fmv,
        "plan": plan,
        "max_rounds": 20,
        "trace_id": trace_id,
    }


@router.post("/{session_id}/round")
async def execute_round(session_id: str, request: Request):
    """
    Execute one round of MARL negotiation.

    Both buyer and seller agents independently select actions
    (hold, concede, counter, accept, walk away) based on their
    trained policies and the current state.
    """
    redis = request.app.state.redis
    state = await redis.get_agent_state(session_id, "env")

    if not state:
        raise HTTPException(404, "Negotiation session not found")
    if state.get("done"):
        raise HTTPException(400, "Negotiation already complete")

    # Reconstruct environment from stored state
    buyer_config = AgentConfig(
        agent_id=state["buyer_config"]["agent_id"], role="buyer",
        reserve_price_usd=state["buyer_config"]["reserve"],
        target_price_usd=state["buyer_config"]["target"],
        urgency=state["buyer_config"]["urgency"],
    )
    seller_config = AgentConfig(
        agent_id=state["seller_config"]["agent_id"], role="seller",
        reserve_price_usd=state["seller_config"]["reserve"],
        target_price_usd=state["seller_config"]["target"],
        urgency=state["seller_config"]["urgency"],
    )
    market = MarketContext(
        property_fair_value_usd=state["market"]["fmv"],
        days_on_market=state["market"]["dom"],
        market_temperature=state["market"]["temp"],
    )

    from app.agents.negotiation_engine import NegotiationEnvironment, MAPPOAgent
    import numpy as np

    env = NegotiationEnvironment(buyer_config, seller_config, market)
    env.reset()
    env.current_bid = state["current_bid"]
    env.current_ask = state["current_ask"]
    env.round = state["round"]

    # Use trained agents or heuristic fallback
    buyer_agent = _get_or_create_engine("buyer")
    seller_agent = _get_or_create_engine("seller")

    buyer_obs = env._get_obs("buyer")
    seller_obs = env._get_obs("seller")

    buyer_action, _ = buyer_agent.select_action(buyer_obs)
    seller_action, _ = seller_agent.select_action(seller_obs)

    _, (br, sr), done, info = env.step(buyer_action, seller_action)

    # Update state in Redis
    state["current_bid"] = env.current_bid
    state["current_ask"] = env.current_ask
    state["round"] = env.round
    state["done"] = done
    await redis.set_agent_state(session_id, "env", state)

    round_data = {
        "round_number": env.round,
        "buyer_action": NegotiationAction(buyer_action).name,
        "buyer_amount_usd": env.current_bid,
        "seller_action": NegotiationAction(seller_action).name,
        "seller_amount_usd": env.current_ask,
        "buyer_reward": round(br, 4),
        "seller_reward": round(sr, 4),
        "cooperative_resilience": round(info["cooperative_resilience"], 4),
        "nash_distance": round(info["nash_distance"], 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await redis.push_round(session_id, round_data)

    return {
        "session_id": session_id,
        "status": "accepted" if done and env.agreed_price else ("failed" if done else "in_progress"),
        "agreed_price_usd": env.agreed_price,
        **round_data,
    }


@router.get("/{session_id}/status")
async def get_status(session_id: str, request: Request):
    """Get current negotiation state and full round history."""
    redis = request.app.state.redis
    state = await redis.get_agent_state(session_id, "env")
    if not state:
        raise HTTPException(404, "Session not found")

    rounds = await redis.get_rounds(session_id)

    return {
        "session_id": session_id,
        "property_id": state["property_id"],
        "current_bid": state["current_bid"],
        "current_ask": state["current_ask"],
        "spread": state["current_ask"] - state["current_bid"],
        "round": state["round"],
        "done": state["done"],
        "rounds": rounds,
    }


@router.get("/{session_id}/stream")
async def stream_session(session_id: str, request: Request):
    """Server-Sent Events stream of negotiation rounds for a session."""

    redis = request.app.state.redis

    async def event_generator():
        last_count = 0
        # Send existing rounds first
        try:
            while not await request.is_disconnected():
                rounds = await redis.get_rounds(session_id)
                if len(rounds) > last_count:
                    for r in rounds[last_count:]:
                        yield f"data: {json.dumps(r)}\n\n"
                    last_count = len(rounds)
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")
