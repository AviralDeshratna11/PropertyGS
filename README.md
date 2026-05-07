# PropOS — Autonomous Real Estate Ecosystem (Phase 1)

> **Remote Investment Portal** targeting international investors in Dubai and NYC markets.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Investor Command Center                    │
│                  (Next.js + Tailwind CSS)                    │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Lifestyle│   MARL   │   ZKP    │Settlement│  Transparency   │
│  Search  │Negotiation│Verification│ Escrow │     Ledger      │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                  FastAPI Orchestration Layer                  │
│                        /api/v1/*                             │
├─────────────────────┬───────────────────────────────────────┤
│   /agents Module    │        /circuits Module                │
│  ┌───────────────┐  │  ┌─────────────────────────────────┐  │
│  │ Lifestyle     │  │  │ ZKP Verifier                    │  │
│  │ Search Engine │  │  │ (Noir Proof-of-Funds Circuit)   │  │
│  ├───────────────┤  │  └─────────────────────────────────┘  │
│  │ MARL/MAPPO    │  │                                       │
│  │ Negotiation   │  │  ┌─────────────────────────────────┐  │
│  │ Engine        │  │  │ Solidity Escrow Contract        │  │
│  └───────────────┘  │  │ (2-of-3 Multi-Sig + DLD 2026)  │  │
│                     │  └─────────────────────────────────┘  │
├─────────────────────┴───────────────────────────────────────┤
│              PostgreSQL          │           Redis           │
│         (Property Metadata)      │    (Agent State + Cache)  │
└──────────────────────────────────┴──────────────────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | FastAPI (Python) | AI service orchestration |
| Frontend | React/Next.js + Tailwind | Investor dashboard |
| AI/ML | PyTorch (MAPPO) | Multi-agent negotiation |
| Blockchain | Solidity (EVM/L2) | Automated escrow |
| ZKP | Noir (Aztec) | Zero-knowledge proofs |
| Database | PostgreSQL + Redis | Property data + agent state |

## Quick Start

```bash
# 1. Clone and configure
cp .env.template .env
# Fill in API keys (see .env.template for details)

# 2. Launch with Docker
docker compose up -d

# 3. Seed sample properties
curl -X POST http://localhost:8000/api/v1/properties/seed

# 4. Open dashboard
open http://localhost:3000
```

## Core Modules

### 1. Agentic Lifestyle Search (`/agents/lifestyle_search.py`)
- LLM interprets natural-language queries into structured weights
- Concurrent API calls to Shadowmap, INRIX, HowLoud, WalkScore
- Composite Lifestyle Match Score with explainable breakdowns
- Context contracts with escalation gates

### 2. MARL Negotiation Engine (`/agents/negotiation_engine.py`)
- MAPPO (Centralized Training, Decentralized Execution)
- 18-dimensional observation space per agent
- 7 discrete actions (hold, concede small/medium/large, counter, accept, walk away)
- Fiduciary reward shaping: surplus + progress + cooperative resilience + info gain
- Nash equilibrium convergence tracking
- Domain-randomized training with market jitter

### 3. ZKP Proof of Funds (`/circuits/`)
- Noir circuit: `balance ≥ threshold` range proof + commitment integrity
- ~288 byte proofs via zk-SNARK/PLONK
- ~185K gas on-chain verification
- 72-hour proof expiry for security

### 4. Smart Contract Escrow (`/contracts/PropOSEscrow.sol`)
- 2-of-3 multi-signature release
- Verification gates: title deed, AI inspection, ZKP
- 5% retention (DLD 2026 rule)
- Direct Payment mandate compliance
- Dispute resolution and cancellation

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/search/lifestyle` | Agentic lifestyle search |
| POST | `/api/v1/negotiation/start` | Start MARL negotiation |
| POST | `/api/v1/negotiation/{id}/round` | Execute one round |
| GET | `/api/v1/negotiation/{id}/status` | Get negotiation state |
| POST | `/api/v1/verification/proof-of-funds` | Generate ZKP |
| POST | `/api/v1/settlement/escrow` | Create escrow |
| POST | `/api/v1/settlement/escrow/{id}/sign` | Submit signature |
| POST | `/api/v1/settlement/escrow/{id}/verify-title` | Verify title deed |
| GET | `/api/v1/properties/` | List properties |
| POST | `/api/v1/properties/seed` | Seed sample data |

## Regulatory Compliance

- **UAE**: DLD 2026 Direct Payment mandate, 5% retention, RERA compliance
- **USA**: California EO N-5-26 AI governance, IRS 1099-S filing hooks
- **Global**: Full transparency audit trail with trace IDs

## Project Structure

```
propos/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── core/
│   │   │   ├── config.py        # Environment configuration
│   │   │   └── logging.py       # Structured JSON logging
│   │   ├── agents/
│   │   │   ├── negotiation_engine.py  # MARL/MAPPO engine
│   │   │   └── lifestyle_search.py    # Agentic search
│   │   ├── circuits/
│   │   │   └── zkp_verifier.py  # ZKP orchestration
│   │   ├── api/                 # Route handlers
│   │   ├── models/
│   │   │   ├── domain.py        # SQLAlchemy models
│   │   │   └── schemas.py       # Pydantic schemas
│   │   └── services/
│   │       ├── db.py            # Database engine
│   │       └── redis_manager.py # Redis state manager
│   ├── requirements.txt
│   └── Dockerfile
├── contracts/
│   ├── PropOSEscrow.sol         # Multi-sig escrow
│   └── circuits/
│       └── proof_of_funds/      # Noir ZKP circuit
├── frontend/                    # Next.js dashboard
├── docker-compose.yml
└── .env.template
```
