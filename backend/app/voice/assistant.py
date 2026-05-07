"""
PropOS Voice Assistant — AI Tour Guide for XR Walkthroughs
============================================================
Voice-enabled assistant that accompanies buyers through GSplat tours,
answering contextual questions about the property, neighborhood,
inspection findings, and financials in real time.

Features:
  - Speech-to-Text (Whisper) → intent parsing → LLM response → TTS
  - Spatial context awareness (knows which room the user is in)
  - Live inspection overlay narration
  - IoT data verbal summaries
  - Negotiation status updates
  - Multi-language support (EN, AR, FR, ZH, HI)
"""

import asyncio
import json
import logging
import time
import uuid
import base64
import httpx
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from app.core.config import settings

logger = logging.getLogger("propos.voice")


# ══════════════════════════════════════════════════════════════════════
# §1  VOICE MODELS
# ══════════════════════════════════════════════════════════════════════

class Language(Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    FRENCH = "fr"
    CHINESE = "zh"
    HINDI = "hi"
    SPANISH = "es"
    RUSSIAN = "ru"


class VoiceGender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class IntentType(Enum):
    PROPERTY_INFO = "property_info"
    ROOM_DETAILS = "room_details"
    NEIGHBORHOOD = "neighborhood"
    PRICE_QUESTION = "price_question"
    INSPECTION_STATUS = "inspection_status"
    DEFECT_INQUIRY = "defect_inquiry"
    IOT_READING = "iot_reading"
    COMFORT_QUERY = "comfort_query"
    HVAC_CONTROL = "hvac_control"
    NEGOTIATION_STATUS = "negotiation_status"
    FINANCIAL_QUESTION = "financial_question"
    LEGAL_QUESTION = "legal_question"
    COMPARISON = "comparison"
    SCHEDULE_VISIT = "schedule_visit"
    GENERAL_CHAT = "general_chat"
    NAVIGATION = "navigation"


@dataclass
class SpatialContext:
    """Tracks where the user is in the virtual tour."""
    current_room: str = "entrance"
    floor_level: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 1.5)
    look_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    rooms_visited: List[str] = field(default_factory=list)
    time_in_room_seconds: float = 0.0
    total_tour_seconds: float = 0.0


@dataclass
class VoiceSession:
    """Active voice assistant session."""
    session_id: str
    property_id: int
    language: Language
    voice_gender: VoiceGender
    spatial_context: SpatialContext
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    property_data: Dict[str, Any] = field(default_factory=dict)
    inspection_data: Dict[str, Any] = field(default_factory=dict)
    iot_data: Dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# §2  SPEECH-TO-TEXT (Whisper)
# ══════════════════════════════════════════════════════════════════════

class SpeechToText:
    """
    Converts audio to text using OpenAI Whisper or local whisper model.
    Supports multi-language with automatic detection.
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    async def transcribe(
        self, audio_bytes: bytes, language: str = None
    ) -> Dict[str, Any]:
        """Transcribe audio to text."""
        if self.provider == "openai":
            return await self._transcribe_openai(audio_bytes, language)
        else:
            return await self._transcribe_local(audio_bytes, language)

    async def _transcribe_openai(self, audio_bytes: bytes, language: str) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
                data = {"model": "whisper-1"}
                if language:
                    data["language"] = language

                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    files=files, data=data,
                )
                result = resp.json()
                return {
                    "text": result.get("text", ""),
                    "language": language or "en",
                    "confidence": 0.95,
                }
        except Exception as e:
            logger.error(f"STT error: {e}")
            return {"text": "", "language": "en", "confidence": 0.0, "error": str(e)}

    async def _transcribe_local(self, audio_bytes: bytes, language: str) -> Dict:
        """Local Whisper model fallback."""
        return {"text": "[Local STT not configured]", "language": "en", "confidence": 0.0}


# ══════════════════════════════════════════════════════════════════════
# §3  TEXT-TO-SPEECH
# ══════════════════════════════════════════════════════════════════════

class TextToSpeech:
    """Converts text responses to natural speech audio."""

    VOICE_MAP = {
        (Language.ENGLISH, VoiceGender.FEMALE): "alloy",
        (Language.ENGLISH, VoiceGender.MALE): "onyx",
        (Language.ENGLISH, VoiceGender.NEUTRAL): "nova",
        (Language.ARABIC, VoiceGender.FEMALE): "shimmer",
        (Language.ARABIC, VoiceGender.MALE): "echo",
    }

    def __init__(self, provider: str = "openai"):
        self.provider = provider

    async def synthesize(
        self, text: str, language: Language, gender: VoiceGender
    ) -> bytes:
        """Convert text to speech audio bytes."""
        voice = self.VOICE_MAP.get((language, gender), "nova")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "tts-1",
                        "input": text,
                        "voice": voice,
                        "response_format": "mp3",
                    },
                )
                return resp.content
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""


# ══════════════════════════════════════════════════════════════════════
# §4  INTENT PARSER & CONTEXT ENGINE
# ══════════════════════════════════════════════════════════════════════

TOUR_GUIDE_SYSTEM = """You are PropOS Tour Guide, an expert AI real estate assistant embedded in an immersive 3D property walkthrough.

CONTEXT YOU HAVE ACCESS TO:
- Property details (bedrooms, bathrooms, area, price, year built, etc.)
- Current room the user is in (spatial awareness)
- IoT sensor data (temperature, humidity, air quality, noise per room)
- Inspection findings (any defects detected by YOLOv12/RT-DETR/CracksGPT)
- Negotiation status (if active)
- Neighborhood data (walkability, commute times, noise levels)

PERSONALITY:
- Professional but warm — like a knowledgeable luxury real estate agent
- Proactively highlight positive features when entering new rooms
- Be honest about any inspection findings — trust builds value
- Give practical lifestyle context ("This south-facing window gets morning sun until 11am")
- Reference specific data points when available (sensor readings, scores)

RULES:
- Keep responses concise (2-3 sentences for quick questions, 4-5 for detailed ones)
- Always reference the current room when spatially relevant
- If asked about price/negotiation, be transparent but strategic
- Never fabricate sensor data — say "I don't have readings for that" if unavailable
- For legal questions, advise consulting a professional

Respond in {language}. Be conversational, not robotic."""


class IntentParser:
    """Parse user utterances into structured intents."""

    INTENT_KEYWORDS = {
        IntentType.PROPERTY_INFO: ["how big", "square feet", "area", "size", "bedrooms", "bathrooms", "year built", "parking"],
        IntentType.ROOM_DETAILS: ["this room", "in here", "ceiling", "window", "floor", "wall"],
        IntentType.NEIGHBORHOOD: ["neighborhood", "area around", "nearby", "restaurants", "schools", "commute"],
        IntentType.PRICE_QUESTION: ["price", "cost", "how much", "per square", "valuation", "worth"],
        IntentType.INSPECTION_STATUS: ["inspection", "condition", "structural", "report", "defects"],
        IntentType.DEFECT_INQUIRY: ["crack", "damage", "mold", "leak", "rust", "stain"],
        IntentType.IOT_READING: ["temperature", "humidity", "air quality", "noise", "how warm", "how cold"],
        IntentType.COMFORT_QUERY: ["comfortable", "comfort", "feels like", "thermal"],
        IntentType.HVAC_CONTROL: ["set temperature", "turn on ac", "adjust cooling", "make it cooler", "make it warmer"],
        IntentType.NEGOTIATION_STATUS: ["negotiation", "offer", "bid", "deal", "counter"],
        IntentType.FINANCIAL_QUESTION: ["mortgage", "financing", "roi", "rental yield", "investment"],
        IntentType.LEGAL_QUESTION: ["legal", "title", "deed", "rera", "contract", "escrow"],
        IntentType.NAVIGATION: ["take me to", "go to", "show me", "where is"],
    }

    @staticmethod
    def parse(text: str) -> IntentType:
        text_lower = text.lower()
        for intent, keywords in IntentParser.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return intent
        return IntentType.GENERAL_CHAT


# ══════════════════════════════════════════════════════════════════════
# §5  VOICE ASSISTANT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class VoiceAssistant:
    """
    Full voice assistant pipeline:
    Audio → STT → Intent Parse → Context Build → LLM → TTS → Audio
    """

    def __init__(self):
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self._sessions: Dict[str, VoiceSession] = {}

    def create_session(
        self, property_id: int, language: str = "en",
        gender: str = "neutral", property_data: Dict = None,
    ) -> str:
        session_id = uuid.uuid4().hex[:12]
        session = VoiceSession(
            session_id=session_id,
            property_id=property_id,
            language=Language(language),
            voice_gender=VoiceGender(gender),
            spatial_context=SpatialContext(),
            property_data=property_data or {},
            started_at=time.time(),
        )
        self._sessions[session_id] = session
        return session_id

    async def process_audio(
        self, session_id: str, audio_bytes: bytes
    ) -> Dict[str, Any]:
        """Full pipeline: audio in → audio + text out."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        # Step 1: STT
        transcript = await self.stt.transcribe(
            audio_bytes, session.language.value
        )
        user_text = transcript.get("text", "")
        if not user_text:
            return {"error": "Could not transcribe audio", "transcript": transcript}

        # Step 2: Process text
        result = await self.process_text(session_id, user_text)
        result["transcript"] = user_text
        return result

    async def process_text(
        self, session_id: str, user_text: str
    ) -> Dict[str, Any]:
        """Process text input (for keyboard-based interaction)."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        # Step 1: Parse intent
        intent = IntentParser.parse(user_text)

        # Step 2: Build context
        context = self._build_context(session, intent)

        # Step 3: Generate response via LLM
        response_text = await self._generate_response(session, user_text, context)

        # Step 4: Update conversation history
        session.conversation_history.append({"role": "user", "content": user_text})
        session.conversation_history.append({"role": "assistant", "content": response_text})

        # Keep history manageable
        if len(session.conversation_history) > 20:
            session.conversation_history = session.conversation_history[-16:]

        # Step 5: TTS (async, return text immediately)
        audio_task = asyncio.create_task(
            self.tts.synthesize(response_text, session.language, session.voice_gender)
        )

        try:
            audio_bytes = await asyncio.wait_for(audio_task, timeout=10)
            audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        except asyncio.TimeoutError:
            audio_b64 = None

        return {
            "response_text": response_text,
            "audio_base64": audio_b64,
            "intent": intent.value,
            "current_room": session.spatial_context.current_room,
            "session_id": session_id,
        }

    def update_spatial_context(
        self, session_id: str, room: str = None,
        position: Tuple[float, float, float] = None,
    ):
        """Update user's position in the virtual tour."""
        session = self._sessions.get(session_id)
        if session:
            if room:
                if room != session.spatial_context.current_room:
                    session.spatial_context.rooms_visited.append(room)
                session.spatial_context.current_room = room
            if position:
                session.spatial_context.position = position

    def inject_data(
        self, session_id: str,
        inspection_data: Dict = None,
        iot_data: Dict = None,
    ):
        """Inject live inspection/IoT data into the session context."""
        session = self._sessions.get(session_id)
        if session:
            if inspection_data:
                session.inspection_data = inspection_data
            if iot_data:
                session.iot_data = iot_data

    def _build_context(self, session: VoiceSession, intent: IntentType) -> str:
        """Build rich context string for the LLM based on intent and spatial position."""
        parts = []

        # Always include spatial context
        parts.append(f"[User is currently in: {session.spatial_context.current_room}]")
        parts.append(f"[Rooms visited so far: {', '.join(session.spatial_context.rooms_visited) or 'none yet'}]")

        # Property data
        if session.property_data:
            pd = session.property_data
            parts.append(f"[Property: {pd.get('title', 'N/A')}, "
                        f"{pd.get('bedrooms', '?')}BR/{pd.get('bathrooms', '?')}BA, "
                        f"{pd.get('area_sqft', '?')} sqft, "
                        f"asking ${pd.get('asking_price_usd', '?'):,}, "
                        f"built {pd.get('year_built', '?')}]")

        # Intent-specific context
        if intent in (IntentType.IOT_READING, IntentType.COMFORT_QUERY):
            room = session.spatial_context.current_room
            if session.iot_data and room in session.iot_data.get("rooms", {}):
                rd = session.iot_data["rooms"][room]
                parts.append(f"[IoT readings for {room}: {json.dumps(rd)}]")

        if intent in (IntentType.INSPECTION_STATUS, IntentType.DEFECT_INQUIRY):
            if session.inspection_data:
                parts.append(f"[Inspection summary: {json.dumps(session.inspection_data.get('summary', {}))}]")

        if intent == IntentType.HVAC_CONTROL:
            parts.append("[User wants to adjust HVAC — confirm the action before executing]")

        return "\n".join(parts)

    async def _generate_response(
        self, session: VoiceSession, user_text: str, context: str
    ) -> str:
        """Generate contextual response via LLM."""
        system_prompt = TOUR_GUIDE_SYSTEM.format(language=session.language.value)

        messages = [{"role": "user", "content": f"{context}\n\nUser says: {user_text}"}]

        # Include recent conversation history
        for msg in session.conversation_history[-6:]:
            messages.insert(0, msg)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if settings.LLM_PROVIDER == "anthropic":
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": settings.ANTHROPIC_API_KEY,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": settings.LLM_MODEL,
                            "max_tokens": 300,
                            "system": system_prompt,
                            "messages": messages,
                        },
                    )
                    data = resp.json()
                    return data["content"][0]["text"]
                else:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "system", "content": system_prompt}] + messages,
                            "max_tokens": 300,
                        },
                    )
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"LLM response error: {e}")
            return self._fallback_response(session, user_text)

    def _fallback_response(self, session: VoiceSession, user_text: str) -> str:
        """Rule-based fallback when LLM is unavailable."""
        room = session.spatial_context.current_room
        pd = session.property_data

        if "temperature" in user_text.lower() or "warm" in user_text.lower():
            iot = session.iot_data.get("rooms", {}).get(room, {})
            temp = iot.get("temperature", {}).get("value", "unknown")
            return f"The current temperature in the {room} is {temp}°C."

        if "price" in user_text.lower():
            price = pd.get("asking_price_usd", 0)
            return f"This property is listed at ${price:,}."

        if "bedroom" in user_text.lower():
            beds = pd.get("bedrooms", "?")
            return f"This property has {beds} bedrooms."

        return f"Welcome to the {room}. Feel free to ask about any features you see."

    def get_session_info(self, session_id: str) -> Dict:
        session = self._sessions.get(session_id)
        if not session:
            return {}
        return {
            "session_id": session_id,
            "property_id": session.property_id,
            "language": session.language.value,
            "current_room": session.spatial_context.current_room,
            "rooms_visited": session.spatial_context.rooms_visited,
            "conversation_turns": len(session.conversation_history) // 2,
            "duration_seconds": round(time.time() - session.started_at),
        }
