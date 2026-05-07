"""ZKP Proof-of-Funds verification endpoints."""

from fastapi import APIRouter, Request
from app.models.schemas import ZKPProofRequest, ZKPProofResponse
from app.circuits.zkp_verifier import ZKPVerifier
from datetime import datetime, timezone

router = APIRouter()
verifier = ZKPVerifier()


@router.post("/proof-of-funds", response_model=ZKPProofResponse)
async def verify_proof_of_funds(payload: ZKPProofRequest, request: Request):
    """
    Zero-Knowledge Proof of Funds verification.

    The buyer proves they hold assets ≥ threshold without revealing
    their actual balance. Uses Noir/zk-SNARKs for proof generation
    and on-chain verification (~300 bytes, ~200k gas).
    """
    # Generate proof via Noir prover service
    result = await verifier.generate_proof(
        buyer_id=payload.buyer_id,
        balance_commitment="client_provided",  # In production, from client
        threshold_usd=payload.threshold_usd,
        salt="client_salt",
    )

    if not result.get("success"):
        return ZKPProofResponse(
            buyer_id=payload.buyer_id,
            proof_hash="",
            threshold_usd=payload.threshold_usd,
            verified=False,
            expires_at=datetime.now(timezone.utc),
        )

    # Verify on-chain
    verification = await verifier.verify_on_chain(
        result["proof_hash"], result.get("public_inputs_hash", "")
    )

    # Log to transparency ledger
    redis = request.app.state.redis
    trace_id = getattr(request.state, "trace_id", "")
    await redis.log_decision(trace_id, {
        "agent": "zkp_verifier",
        "action": "proof_of_funds_verified",
        "buyer_id": payload.buyer_id,
        "threshold_usd": payload.threshold_usd,
        "verified": verification["verified"],
        "proof_size_bytes": result.get("proof_size_bytes", 288),
    })

    return ZKPProofResponse(
        buyer_id=payload.buyer_id,
        proof_hash=result["proof_hash"],
        threshold_usd=payload.threshold_usd,
        verified=verification["verified"],
        verification_tx=verification.get("tx_hash"),
        expires_at=ZKPVerifier.proof_expiry(),
    )
