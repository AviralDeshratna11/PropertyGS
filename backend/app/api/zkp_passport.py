"""ZK-Investor Passport endpoints."""

from fastapi import APIRouter, HTTPException, Request
import uuid
from datetime import datetime, timezone
from app.core.config import settings
from app.models.schemas import ZKPassportRequest, ZKPassportResponse, ZKPassportVerifyRequest
from app.services.noir_prover import prove_balance_threshold, verify_balance_proof

router = APIRouter()


@router.post("/zkp/passport/request")
async def request_passport(payload: ZKPassportRequest, request: Request):
    """Request a ZK-Investor Passport token."""
    commitment = {"buyer_id": payload.buyer_id, "created_at": datetime.now(timezone.utc).isoformat()}
    proof = await prove_balance_threshold(
        commitment=commitment,
        threshold_usd=payload.threshold_usd,
        prover_url=settings.NOIR_PROVER_URL,
        api_key=settings.NOIR_PROVER_API_KEY,
    )

    token = f"ZKP-PASSPORT-{uuid.uuid4().hex[:12]}"
    expires = datetime.now(timezone.utc).timestamp() + 60 * 60 * 24
    passport = {
        "buyer_id": payload.buyer_id,
        "token": token,
        "threshold_usd": payload.threshold_usd,
        "proof_hash": proof["proof_hash"],
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
        "verified": proof.get("verified", False),
        "note": f"Noir mode: {proof.get('mode', 'unknown')}",
    }
    await request.app.state.redis.cache_set(f"zk_passport:{token}", passport, ttl=60 * 60 * 24)
    return passport


@router.post("/zkp/passport/verify")
async def verify_passport(payload: ZKPassportVerifyRequest, request: Request):
    if not payload.token.startswith("ZKP-PASSPORT-"):
        raise HTTPException(400, "Invalid token")

    cached = await request.app.state.redis.cache_get(f"zk_passport:{payload.token}")
    if not cached:
        raise HTTPException(404, "Passport not found or expired")

    verification = await verify_balance_proof(
        proof_hash=cached.get("proof_hash", ""),
        prover_url=settings.NOIR_PROVER_URL,
        api_key=settings.NOIR_PROVER_API_KEY,
    )

    cached["verified"] = bool(verification.get("verified", False))
    cached["note"] = f"Verification mode: {verification.get('mode', 'unknown')}"
    await request.app.state.redis.cache_set(f"zk_passport:{payload.token}", cached, ttl=60 * 60 * 24)

    return {
        "token": payload.token,
        "verified": cached["verified"],
        "message": "Verification completed",
        "mode": verification.get("mode", "unknown"),
    }
