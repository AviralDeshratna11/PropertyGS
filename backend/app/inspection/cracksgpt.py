"""CracksGPT: Vision-language narrative generation for defects.

Converts defect detections into structured, insurance-ready narratives.
This is a stub that shows expected outputs and will later be hooked to a VLM.
"""
from typing import List, Dict
import hashlib


def generate_narratives(defects: List[Dict]) -> List[Dict]:
    narratives = []
    for d in defects:
        text = f"{d.get('category','defect')} detected at {d.get('location','unknown')}; confidence {d.get('confidence',0):.2f}."
        # Example: infer severity language
        severity = d.get('severity', 'medium')
        narratives.append({
            "defect_id": d.get('defect_id') or hashlib.sha1(str(d).encode()).hexdigest()[:8],
            "narrative": text,
            "severity": severity,
            "insurance_ready": severity in ['high','critical'],
        })
    return narratives
