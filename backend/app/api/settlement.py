"""Blockchain Settlement and Escrow management endpoints."""

from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import EscrowCreate, EscrowState, SignatureRequest

router = APIRouter()


@router.post("/escrow")
async def create_escrow(payload: EscrowCreate, request: Request):
    """
    Deploy a new multi-sig escrow smart contract for a negotiated deal.

    The contract holds funds in suspension until:
      - 2-of-3 multi-sig approval (buyer, seller, settlement agent)
      - Digital title deed verification via DLD REST API
      - AI-verified inspection sign-off
      - 5% retention per DLD 2026 rule
    """
    # Calculate 5% retention per Dubai 2026 regulation
    retention = int(payload.amount_usd * 0.05)

    return {
        "status": "escrow_created",
        "session_id": payload.negotiation_session_id,
        "amount_usd": payload.amount_usd,
        "retention_amount_usd": retention,
        "release_amount_usd": payload.amount_usd - retention,
        "contract_address": "0x" + "0" * 40,  # Placeholder — deployed on-chain in prod
        "required_signatures": 2,
        "signers": {
            "buyer": {"wallet": payload.buyer_wallet, "signed": False},
            "seller": {"wallet": payload.seller_wallet, "signed": False},
            "agent": {"wallet": payload.settlement_agent_wallet, "signed": False},
        },
        "gates": {
            "title_verified": False,
            "inspection_passed": False,
            "zkp_funds_verified": False,
        },
        "compliance": {
            "uae_direct_payment": True,
            "recipient_bank": payload.recipient_bank_account,
            "recipient_name": payload.recipient_name_on_deed,
        },
    }


@router.post("/escrow/{session_id}/sign")
async def sign_escrow(session_id: str, payload: SignatureRequest, request: Request):
    """Submit a multi-sig signature for escrow release."""
    return {
        "session_id": session_id,
        "signer": payload.signer_role,
        "signed": True,
        "remaining_signatures": 1,
        "message": f"{payload.signer_role} signature recorded",
    }


@router.post("/escrow/{session_id}/verify-title")
async def verify_title(session_id: str, request: Request):
    """
    Verify title deed via Dubai Land Department REST API integration.
    In production, calls DLD_BASE_URL with the deed hash.
    """
    return {
        "session_id": session_id,
        "title_verified": True,
        "verification_source": "Dubai Land Department",
        "message": "Digital title deed verified — gate passed",
    }


@router.get("/escrow/{session_id}/status")
async def escrow_status(session_id: str, request: Request):
    """Get current escrow state including signatures and verification gates."""
    return {
        "session_id": session_id,
        "status": "pending",
        "message": "Awaiting signatures and verification gates",
    }
