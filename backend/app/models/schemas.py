"""Pydantic schemas for all API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


# ── Lifestyle Search ──────────────────────────────────────────────────

class LifestyleQuery(BaseModel):
    """Natural-language lifestyle search input."""
    query: str = Field(..., example="Quiet villa with morning sun and <20 min commute to DIFC")
    city: str = Field(default="Dubai", example="Dubai")
    budget_min_usd: Optional[int] = None
    budget_max_usd: Optional[int] = None
    currency: str = Field(default="USD")
    max_results: int = Field(default=10, ge=1, le=50)


class LifestyleWeights(BaseModel):
    """Extracted weights from NLP interpretation."""
    sunlight: float = Field(default=0.0, ge=0, le=1)
    quietness: float = Field(default=0.0, ge=0, le=1)
    walkability: float = Field(default=0.0, ge=0, le=1)
    commute: float = Field(default=0.0, ge=0, le=1)
    commute_target: Optional[str] = None
    commute_max_minutes: Optional[int] = None
    custom_criteria: Dict[str, float] = Field(default_factory=dict)


class PropertyMatch(BaseModel):
    """Single property with its lifestyle match scoring."""
    property_id: int
    title: str
    city: str
    district: Optional[str]
    asking_price_usd: int
    area_sqft: Optional[float]
    bedrooms: Optional[int]
    bathrooms: Optional[int]

    # Lifestyle scores
    sunlight_score: Optional[float]
    noise_score: Optional[float]
    walk_score: Optional[float]
    commute_minutes: Optional[Dict[str, int]]
    lifestyle_match_score: float

    # Breakdown
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


class SearchResponse(BaseModel):
    query_interpreted: LifestyleWeights
    results: List[PropertyMatch]
    total_found: int
    trace_id: str


# ── Negotiation ───────────────────────────────────────────────────────

class NegotiationAction(str, Enum):
    BID = "bid"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    CONCEDE = "concede"
    WALK_AWAY = "walk_away"


class NegotiationStart(BaseModel):
    property_id: int
    buyer_id: str
    seller_id: str
    buyer_max_budget_usd: int
    seller_reserve_price_usd: int
    buyer_urgency: float = Field(default=0.5, ge=0, le=1)
    seller_urgency: float = Field(default=0.5, ge=0, le=1)


class NegotiationRound(BaseModel):
    round_number: int
    buyer_action: NegotiationAction
    buyer_amount_usd: Optional[int]
    seller_action: NegotiationAction
    seller_amount_usd: Optional[int]
    buyer_reward: float
    seller_reward: float
    cooperative_resilience: float
    nash_distance: float
    timestamp: datetime


class NegotiationState(BaseModel):
    session_id: str
    property_id: int
    status: str
    current_bid_usd: Optional[int]
    current_ask_usd: Optional[int]
    round_count: int
    max_rounds: int
    rounds: List[NegotiationRound]
    buyer_surplus: Optional[float]
    seller_surplus: Optional[float]
    agreed_price_usd: Optional[int]


# ── ZKP Verification ─────────────────────────────────────────────────

class ZKPProofRequest(BaseModel):
    buyer_id: str
    threshold_usd: int = Field(..., gt=0, description="Minimum balance to prove")


class ZKPProofResponse(BaseModel):
    buyer_id: str
    proof_hash: str
    threshold_usd: int
    verified: bool
    verification_tx: Optional[str]
    expires_at: datetime


# ── Settlement / Escrow ───────────────────────────────────────────────

class EscrowCreate(BaseModel):
    negotiation_session_id: str
    amount_usd: int
    buyer_wallet: str
    seller_wallet: str
    settlement_agent_wallet: str
    recipient_bank_account: str = Field(..., description="UAE bank account per DLD mandate")
    recipient_name_on_deed: str


class EscrowState(BaseModel):
    contract_address: Optional[str]
    status: str
    amount_usd: int
    retention_amount_usd: int
    signatures: Dict[str, bool]
    gates: Dict[str, bool]


class SignatureRequest(BaseModel):
    signer_role: str = Field(..., pattern="^(buyer|seller|agent)$")
    wallet_address: str
    signature: str
