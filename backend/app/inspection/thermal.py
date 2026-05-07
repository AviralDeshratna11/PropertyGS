"""Thermal imaging integration helpers (mocked).

Provides functions to analyze thermal images and detect moisture/insulation anomalies.
"""
from typing import List, Dict
import random


def analyze_thermal_images(thermal_images: List[bytes]) -> List[Dict]:
    """Mock analysis: returns a list of thermal findings per image."""
    findings = []
    for i, _ in enumerate(thermal_images):
        if random.random() < 0.3:
            findings.append({
                "image_index": i,
                "issue": "moisture_intrusion",
                "confidence": round(0.7 + random.random() * 0.25, 2),
                "notes": "Thermal anomaly suggests elevated moisture; recommend moisture meter follow-up",
            })
    return findings
