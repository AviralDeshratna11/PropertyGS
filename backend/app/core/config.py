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
    DATABASE_URL: str = "sqlite+aiosqlite:///./propos.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── CORS ──────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001", "https://app.propos.io"]

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
    NOIR_PROVER_API_KEY: str = ""

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

    # ── PHASE 2: Perception (GSplat) ──────────────────────────────────
    GSPLAT_MODEL_PATH: str = ""         # Path to .splat files or model weights
    GSPLAT_ENABLE: bool = True
    GSPLAT_MAX_GAUSSIANS: int = 1_000_000
    GSPLAT_SH_DEGREE: int = 1           # 0=mobile, 1=balanced, 2/3=desktop

    # ── PHASE 2: Inspection (YOLOv12 / RT-DETR / CracksGPT) ───────────
    YOLOV12_MODEL_PATH: str = ""        # Path to YOLOv12 weights
    RTDETR_MODEL_PATH: str = ""         # Path to RT-DETR weights
    CRACKSGPT_MODEL_PATH: str = ""      # Path to CracksGPT VLM weights
    INSPECTION_ENABLE: bool = True
    INSPECTION_CONFIDENCE_THRESHOLD: float = 0.35
    INSPECTION_NMS_THRESHOLD: float = 0.45
    INSPECTION_DEVICE: str = "cpu"      # "cpu" or "cuda"

    # ── PHASE 2: Voice (STT/TTS/LLM) ──────────────────────────────────
    VOICE_ENABLE: bool = True
    VOICE_STT_PROVIDER: str = "openai"  # "openai" or "local"
    VOICE_TTS_PROVIDER: str = "openai"  # "openai" or "elevenlabs"
    VOICE_LLM_CONTEXT_SIZE: int = 2048
    ELEVENLABS_API_KEY: str = ""        # For alternative TTS

    # ── PHASE 2: IoT ──────────────────────────────────────────────────
    IOT_ENABLE: bool = True
    IOT_MQTT_BROKER: str = "mqtt://localhost:1883"
    IOT_SENSOR_POLL_INTERVAL_SEC: int = 30

    # ── PHASE 2: XR/AR (WebXR, Hand Tracking) ────────────────────────
    XR_ENABLE: bool = False             # Not yet implemented
    XR_WEBXR_SUPPORTED: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
