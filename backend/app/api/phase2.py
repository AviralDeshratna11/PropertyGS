"""Phase 2 API routes — Perception, Inspection, IoT, Voice."""

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from typing import Optional
import numpy as np
import uuid
import time

router = APIRouter()

# ── Lazy singletons ──────────────────────────────────────────────────
_perception = None
_inspection = None
_iot = None
_voice = None

def _get_perception():
    global _perception
    if _perception is None:
        from app.perception.gsplat_pipeline import SceneManager
        _perception = SceneManager()
    return _perception

def _get_inspection():
    global _inspection
    if _inspection is None:
        from app.inspection.defect_detector import InspectionOrchestrator
        _inspection = InspectionOrchestrator()
    return _inspection

def _get_iot():
    global _iot
    if _iot is None:
        from app.iot.sensor_overlay import IoTOrchestrator
        _iot = IoTOrchestrator()
    return _iot

def _get_voice():
    global _voice
    if _voice is None:
        from app.voice.assistant import VoiceAssistant
        _voice = VoiceAssistant()
    return _voice


# ══════════════════════════════════════════════════════════════════════
# PERCEPTION (GSplat)
# ══════════════════════════════════════════════════════════════════════

@router.post("/perception/scenes")
async def create_scene(property_id: int, request: Request):
    """Start GSplat scene creation for a property."""
    mgr = _get_perception()
    scene = await mgr.create_scene(property_id)
    return {
        "scene_id": scene.scene_id,
        "property_id": property_id,
        "num_gaussians": scene.num_gaussians,
        "psnr_db": round(scene.psnr, 2),
        "training_minutes": round(scene.training_duration_minutes, 1),
        "file_size_mb": round(scene.file_size_mb, 2),
    }

@router.get("/perception/scenes")
async def list_scenes(property_id: int = None):
    """List all GSplat scenes, optionally filtered by property."""
    return _get_perception().list_scenes(property_id)

@router.get("/perception/scenes/{scene_id}")
async def get_scene(scene_id: str):
    """Get scene details and WebGL viewer configuration."""
    mgr = _get_perception()
    scene = mgr.get_scene(scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")
    return {
        "scene_id": scene.scene_id,
        "property_id": scene.property_id,
        "num_gaussians": scene.num_gaussians,
        "sh_degree": scene.sh_degree.value,
        "psnr": round(scene.psnr, 2),
        "bounding_box": scene.bounding_box,
        "viewer_config": mgr.get_viewer_config(scene_id),
    }

@router.get("/perception/scenes/{scene_id}/viewer")
async def get_viewer_config(scene_id: str):
    """Get WebGL viewer configuration for embedding."""
    config = _get_perception().get_viewer_config(scene_id)
    if not config:
        raise HTTPException(404, "Scene not found")
    return config


# ══════════════════════════════════════════════════════════════════════
# INSPECTION (YOLOv12 / RT-DETR / CracksGPT)
# ══════════════════════════════════════════════════════════════════════

@router.post("/inspection/scan")
async def run_inspection(property_id: int, num_images: int = 10, request: Request = None):
    """
    Run full two-stage AI inspection.
    Stage 1: Fast YOLOv12 scan (<5s)
    Stage 2: Deep CracksGPT analysis (~30min simulated)
    """
    orch = _get_inspection()

    # Generate synthetic test images for demo
    images = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
              for _ in range(num_images)]
    locations = [f"Room {chr(65+i)}, {['north','south','east','west'][i%4]} wall"
                 for i in range(num_images)]

    report = await orch.run_full_inspection(
        images=images,
        property_id=property_id,
        locations=locations,
    )

    # Serialize defects
    def serialize_defect(d):
        return {
            "defect_id": d.defect_id,
            "category": d.category.value,
            "severity": d.severity.value,
            "confidence": d.confidence,
            "bbox": list(d.bbox),
            "location": d.location_description,
            "width_mm": d.width_mm,
            "length_mm": d.length_mm,
            "area_sq_cm": d.area_sq_cm,
            "source_model": d.source_model,
            "remediation": d.remediation_suggestion,
            "insurance_grade": d.insurance_grade,
        }

    return {
        "report_id": report.report_id,
        "property_id": report.property_id,
        "inspection_date": report.inspection_date,
        "overall_condition": report.overall_condition,
        "structural_risk_score": report.structural_risk_score,
        "ai_verified": report.ai_verified,
        "summary": {
            "total": report.total_defects,
            "critical": report.critical_count,
            "high": report.high_count,
            "medium": report.medium_count,
            "low": report.low_count,
        },
        "stage1": {
            "defects": [serialize_defect(d) for d in report.stage1_defects],
            "duration_seconds": report.stage1_duration_seconds,
            "images_scanned": report.stage1_total_images,
        },
        "stage2": {
            "defects": [serialize_defect(d) for d in report.stage2_defects],
            "duration_minutes": report.stage2_duration_minutes,
            "cross_referenced": report.stage2_cross_referenced,
            "insurance_ready": report.insurance_ready,
        },
    }

@router.get("/inspection/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a completed inspection report."""
    report = _get_inspection().get_report(report_id)
    if not report:
        raise HTTPException(404, "Report not found")
    return {"report_id": report.report_id, "condition": report.overall_condition,
            "risk_score": report.structural_risk_score}


# ══════════════════════════════════════════════════════════════════════
# IoT SENSORS
# ══════════════════════════════════════════════════════════════════════

@router.get("/iot/{property_id}/snapshot")
async def iot_snapshot(property_id: int):
    """Get current environmental snapshot with PMV heatmap data."""
    iot = _get_iot()
    return iot.get_environmental_snapshot(property_id)

@router.post("/iot/{property_id}/hvac")
async def hvac_control(property_id: int, room: str, target_temp: float):
    """Bi-directional HVAC control — set target temperature from XR tour."""
    iot = _get_iot()
    result = await iot.send_hvac_command(property_id, room, target_temp)
    return result


# ══════════════════════════════════════════════════════════════════════
# VOICE ASSISTANT
# ══════════════════════════════════════════════════════════════════════

@router.post("/voice/sessions")
async def create_voice_session(
    property_id: int, language: str = "en", gender: str = "neutral",
):
    """Start a voice assistant session for a property tour."""
    va = _get_voice()
    session_id = va.create_session(property_id, language, gender)
    return {"session_id": session_id, "language": language, "status": "active"}

@router.post("/voice/{session_id}/text")
async def voice_text_input(session_id: str, text: str):
    """Send text input to voice assistant (keyboard mode)."""
    va = _get_voice()
    result = await va.process_text(session_id, text)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result

@router.post("/voice/{session_id}/audio")
async def voice_audio_input(session_id: str, file: UploadFile = File(...)):
    """Send audio input to voice assistant (microphone mode)."""
    va = _get_voice()
    audio_bytes = await file.read()
    result = await va.process_audio(session_id, audio_bytes)
    return result

@router.post("/voice/{session_id}/position")
async def update_position(session_id: str, room: str = None, x: float = 0, y: float = 0, z: float = 1.5):
    """Update user's spatial position in the tour."""
    va = _get_voice()
    va.update_spatial_context(session_id, room, (x, y, z))
    return {"status": "updated", "room": room}

@router.get("/voice/{session_id}")
async def voice_session_info(session_id: str):
    """Get voice session status and conversation stats."""
    return _get_voice().get_session_info(session_id)
