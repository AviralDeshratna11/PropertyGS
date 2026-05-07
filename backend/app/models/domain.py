"""
PropOS Domain Models — Factual Data Layer
==========================================
Single source of truth for property metadata, negotiation records,
verification proofs, and the transparency audit ledger.
"""

from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Boolean, DateTime, Text,
    JSON, Enum as SQLEnum, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.services.db import Base
import enum


# ── Enums ─────────────────────────────────────────────────────────────

class PropertyStatus(str, enum.Enum):
    ACTIVE = "active"
    UNDER_OFFER = "under_offer"
    SOLD = "sold"
    OFF_MARKET = "off_market"


class NegotiationStatus(str, enum.Enum):
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COUNTER_OFFER = "counter_offer"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EscrowStatus(str, enum.Enum):
    PENDING = "pending"
    FUNDED = "funded"
    TITLE_VERIFIED = "title_verified"
    INSPECTION_PASSED = "inspection_passed"
    RELEASED = "released"
    DISPUTED = "disputed"


# ── Property ──────────────────────────────────────────────────────────

class Property(Base):
    __tablename__ = "properties"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_id = Column(String(128), unique=True, index=True, comment="DLD / MLS ID")
    title = Column(String(512), nullable=False)
    description = Column(Text)

    # Location
    city = Column(String(128), nullable=False)
    district = Column(String(256))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(Text)

    # Attributes
    property_type = Column(String(64))      # villa, apartment, penthouse, commercial
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    area_sqft = Column(Float)
    floor_number = Column(Integer)
    year_built = Column(Integer)
    parking_spaces = Column(Integer, default=0)

    # Pricing
    asking_price_usd = Column(BigInteger, nullable=False)
    price_per_sqft = Column(Float)
    currency = Column(String(3), default="USD")

    # Lifestyle scores (populated by Agentic Search)
    sunlight_score = Column(Float, comment="0-100 from Shadowmap")
    noise_score = Column(Float, comment="0-100 from HowLoud (higher = quieter)")
    walk_score = Column(Float, comment="0-100 from WalkScore")
    transit_score = Column(Float)
    commute_minutes = Column(JSON, comment='{"DIFC": 18, "JBR": 25}')
    lifestyle_match_score = Column(Float, comment="Composite weighted score")

    # Legal
    title_deed_hash = Column(String(128), comment="SHA-256 of verified deed")
    rera_number = Column(String(64))
    encumbrances = Column(JSON, default=list)

    # Status
    status = Column(SQLEnum(PropertyStatus), default=PropertyStatus.ACTIVE)
    is_verified = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    negotiations = relationship("Negotiation", back_populates="property")

    __table_args__ = (
        Index("idx_city_status", "city", "status"),
        Index("idx_lifestyle", "lifestyle_match_score"),
        Index("idx_price_range", "asking_price_usd"),
    )


# ── Negotiation ───────────────────────────────────────────────────────

class Negotiation(Base):
    __tablename__ = "negotiations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True)
    property_id = Column(BigInteger, ForeignKey("properties.id"), nullable=False)

    buyer_id = Column(String(128), nullable=False)
    seller_id = Column(String(128), nullable=False)

    # MARL state
    status = Column(SQLEnum(NegotiationStatus), default=NegotiationStatus.INITIATED)
    current_bid_usd = Column(BigInteger)
    current_ask_usd = Column(BigInteger)
    round_count = Column(Integer, default=0)
    max_rounds = Column(Integer, default=20)

    # Fiduciary metrics
    buyer_surplus = Column(Float, comment="Buyer value - agreed price")
    seller_surplus = Column(Float, comment="Agreed price - seller reserve")
    cooperative_resilience = Column(Float, comment="Joint surplus metric")
    nash_distance = Column(Float, comment="Distance from Nash equilibrium")

    # Outcome
    agreed_price_usd = Column(BigInteger)
    agreement_reached_at = Column(DateTime(timezone=True))

    # Audit
    agent_decision_log = Column(JSON, default=list, comment="Full MARL action trace")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    property = relationship("Property", back_populates="negotiations")
    escrow = relationship("Escrow", back_populates="negotiation", uselist=False)


# ── Escrow / Settlement ──────────────────────────────────────────────

class Escrow(Base):
    __tablename__ = "escrows"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    negotiation_id = Column(BigInteger, ForeignKey("negotiations.id"), unique=True)

    # On-chain references
    contract_address = Column(String(42), comment="Deployed escrow contract")
    tx_hash_funded = Column(String(66))
    tx_hash_released = Column(String(66))

    # Multi-sig
    buyer_signed = Column(Boolean, default=False)
    seller_signed = Column(Boolean, default=False)
    agent_signed = Column(Boolean, default=False)
    required_signatures = Column(Integer, default=2)

    # Verification gates
    title_verified = Column(Boolean, default=False)
    inspection_passed = Column(Boolean, default=False)
    zkp_funds_verified = Column(Boolean, default=False)

    # Settlement
    amount_usd = Column(BigInteger)
    retention_amount_usd = Column(BigInteger, comment="5% retention per DLD 2026 rule")
    status = Column(SQLEnum(EscrowStatus), default=EscrowStatus.PENDING)

    # UAE Direct Payment compliance
    recipient_bank_account = Column(String(256), comment="Must be UAE-based per DLD mandate")
    recipient_name_on_deed = Column(String(256))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    negotiation = relationship("Negotiation", back_populates="escrow")


# ── Transparency Audit Ledger ─────────────────────────────────────────

class AuditEntry(Base):
    __tablename__ = "audit_ledger"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trace_id = Column(String(64), index=True, nullable=False)
    agent_type = Column(String(64), comment="search_agent | buyer_agent | seller_agent | escrow")
    action = Column(String(128), nullable=False)
    input_summary = Column(JSON)
    output_summary = Column(JSON)
    reasoning = Column(Text, comment="Explainable AI rationale for this decision")
    confidence = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_audit_trace", "trace_id"),
        Index("idx_audit_agent", "agent_type"),
    )


# ── ZKP Verification Record ──────────────────────────────────────────

class ZKPVerification(Base):
    __tablename__ = "zkp_verifications"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    buyer_id = Column(String(128), nullable=False, index=True)
    proof_type = Column(String(64), default="proof_of_funds")

    # Proof data (stored but balance is NEVER stored)
    proof_hash = Column(String(128), nullable=False)
    public_inputs_hash = Column(String(128))
    threshold_usd = Column(BigInteger, comment="Minimum balance proven")
    verified = Column(Boolean, default=False)
    verifier_contract = Column(String(42))
    verification_tx = Column(String(66))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
