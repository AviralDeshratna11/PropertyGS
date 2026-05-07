"""
Centralized configuration — all secrets and API keys loaded from env vars.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────
    APP_NAME: str = "PropOS"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://propos:propos@db:5432/propos"
    REDIS_URL: str = "redis://redis:6379/0"

    # ── CORS ──────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://app.propos.io"]

    # ── Lifestyle Search APIs ─────────────────────────────────────────
    SHADOWMAP_API_KEY: str = ""          # Sunlight / irradiance simulation
    SHADOWMAP_BASE_URL: str = "https://api.shadowmap.org/v1"

    INRIX_APP_ID: str = ""               # Commute / drive-time patterns
    INRIX_APP_KEY: str = ""
    INRIX_BASE_URL: str = "https://api.iq.inrix.com/v1"

    HOWLOUD_API_KEY: str = ""            # Acoustic Soundscore
    HOWLOUD_BASE_URL: str = "https://api.howloud.com/v1"

    WALKSCORE_API_KEY: str = ""          # Walkability / transit scores
    WALKSCORE_BASE_URL: str = "https://api.walkscore.com"

    # ── LLM / Agentic ────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""             # For GPT-4o lifestyle interpretation
    ANTHROPIC_API_KEY: str = ""          # For Claude-based agent orchestration
    LLM_PROVIDER: str = "anthropic"      # "openai" | "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-20250514"

    # ── Blockchain / Settlement ───────────────────────────────────────
    ETH_RPC_URL: str = "https://sepolia.infura.io/v3/YOUR_KEY"
    ESCROW_CONTRACT_ADDRESS: str = ""
    DEPLOYER_PRIVATE_KEY: str = ""       # For contract deployment only

    # ── ZKP ───────────────────────────────────────────────────────────
    NOIR_PROVER_URL: str = "http://zkp-prover:8080"

    # ── External Real Estate Data ─────────────────────────────────────
    DLD_API_KEY: str = ""                # Dubai Land Department REST API
    DLD_BASE_URL: str = "https://api.dubailand.gov.ae/v1"

    # ── Currency / FX ─────────────────────────────────────────────────
    FX_API_KEY: str = ""                 # For USD/AED conversion
    DEFAULT_CURRENCY: str = "USD"
    SUPPORTED_CURRENCIES: List[str] = ["USD", "AED", "EUR", "GBP"]

    # ── MARL Training ─────────────────────────────────────────────────
    MARL_CHECKPOINT_DIR: str = "/data/marl_checkpoints"
    MARL_TRAINING_EPISODES: int = 50_000
    MARL_LEARNING_RATE: float = 3e-4
    MARL_GAMMA: float = 0.99
    MARL_GAE_LAMBDA: float = 0.95
    MARL_CLIP_EPSILON: float = 0.2
    MARL_ENTROPY_COEFF: float = 0.01
    MARL_VALUE_LOSS_COEFF: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
