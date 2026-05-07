"""
PropOS Phase 1+2 — Remote Investment Portal
============================================
Core FastAPI application orchestrating:
    - Agentic Lifestyle Search (LLM + API fusion)
    - MARL Negotiation Engine (MAPPO fiduciary bargaining)
    - ZKP Verification Gateway
    - Settlement Layer hooks
    - Perception, inspection, IoT, and voice flows
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time, uuid, logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.api import search, negotiation, verification, settlement, properties, phase2, zkp_passport
from app.services.redis_manager import RedisManager
from app.services.db import engine, Base

logger = logging.getLogger("propos")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    setup_logging()
    logger.info("PropOS Phase 1 starting — initializing services")

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Warm Redis connection pool
    app.state.redis = RedisManager(settings.REDIS_URL)
    await app.state.redis.connect()

    logger.info("All services initialized — PropOS is live")
    yield

    # Shutdown
    await app.state.redis.disconnect()
    await engine.dispose()
    logger.info("PropOS shut down cleanly")


app = FastAPI(
    title="PropOS — Autonomous Real Estate Ecosystem",
    version="1.1.0-phase1-2",
    description="Phase 1+2: Remote Investment Portal with MARL Negotiation, "
                "Agentic Lifestyle Search, ZKP Verification, Smart Contract Settlement, "
                "and Perception / Inspection / IoT / Voice overlays.",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Transparency Ledger Middleware ────────────────────────────────────
@app.middleware("http")
async def transparency_ledger(request: Request, call_next):
    """
    Attach a unique trace-id to every request and log timing.
    All agent decisions reference this trace for audit compliance
    (California EO N-5-26 / UAE RERA transparency requirements).
    """
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-PropOS-Trace-Id"] = trace_id
    response.headers["X-PropOS-Latency-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "request",
        extra={"trace_id": trace_id, "path": request.url.path,
               "method": request.method, "status": response.status_code,
               "latency_ms": round(elapsed_ms, 1)},
    )
    return response


# ── Route Registration ────────────────────────────────────────────────
app.include_router(search.router,       prefix="/api/v1/search",       tags=["Agentic Search"])
app.include_router(negotiation.router,   prefix="/api/v1/negotiation",  tags=["MARL Negotiation"])
app.include_router(verification.router,  prefix="/api/v1/verification", tags=["ZKP Verification"])
app.include_router(settlement.router,    prefix="/api/v1/settlement",   tags=["Blockchain Settlement"])
app.include_router(properties.router,    prefix="/api/v1/properties",   tags=["Property Data"])
app.include_router(phase2.router,        prefix="/api/v1",              tags=["Phase 2 — Perception, Inspection, IoT, Voice"])
app.include_router(zkp_passport.router,  prefix="/api/v1",              tags=["ZKP Passports"])


@app.get("/health")
async def health():
    return {
        "status": "operational",
        "phases": [1, 2],
        "version": "1.1.0",
        "phase2_status": "/api/v1/phase2/status",
    }
