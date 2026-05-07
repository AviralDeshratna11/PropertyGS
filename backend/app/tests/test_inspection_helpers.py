from app.inspection.thermal import analyze_thermal_images
from app.inspection.cracksgpt import generate_narratives
from app.inspection.repair_estimator import estimate_repairs


def test_thermal_analysis_empty():
    findings = analyze_thermal_images([])
    assert isinstance(findings, list)


def test_cracksgpt_and_estimator():
    defects = [{"defect_id":"d1","category":"crack","location":"wall","severity":"medium","confidence":0.85}]
    narratives = generate_narratives(defects)
    assert isinstance(narratives, list) and narratives[0]["severity"] == 'medium'

    est = estimate_repairs(defects, 200000)
    assert 'total_estimated_cost_usd' in est
