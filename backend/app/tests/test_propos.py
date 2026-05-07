"""
PropOS Test Suite — Phase 1 + Phase 2
=======================================
Run: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, patch

# ══════════════════════════════════════════════════════════════════════
# §1  MARL NEGOTIATION ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestNegotiationEnvironment:
    def _make_env(self):
        from app.agents.negotiation_engine import (
            NegotiationEnvironment, AgentConfig, MarketContext
        )
        buyer = AgentConfig("b1", "buyer", reserve_price_usd=1_100_000,
                           target_price_usd=900_000, urgency=0.5)
        seller = AgentConfig("s1", "seller", reserve_price_usd=900_000,
                            target_price_usd=1_100_000, urgency=0.5)
        market = MarketContext(property_fair_value_usd=1_000_000,
                              days_on_market=30, market_temperature=0.6)
        return NegotiationEnvironment(buyer, seller, market)

    def test_reset_returns_observations(self):
        env = self._make_env()
        buyer_obs, seller_obs = env.reset()
        assert buyer_obs.shape == (18,)
        assert seller_obs.shape == (18,)
        assert not env.done

    def test_step_advances_round(self):
        env = self._make_env()
        env.reset()
        (bo, so), (br, sr), done, info = env.step(1, 1)  # Both concede small
        assert env.round == 1
        assert "cooperative_resilience" in info
        assert "nash_distance" in info

    def test_accept_ends_negotiation(self):
        env = self._make_env()
        env.reset()
        (bo, so), (br, sr), done, info = env.step(5, 0)  # Buyer accepts
        assert done is True
        assert env.agreed_price is not None

    def test_walk_away_ends_without_deal(self):
        env = self._make_env()
        env.reset()
        (bo, so), (br, sr), done, info = env.step(6, 0)  # Buyer walks away
        assert done is True
        assert env.agreed_price is None

    def test_convergence_closes_deal(self):
        env = self._make_env()
        env.reset()
        env.current_bid = 1_000_000
        env.current_ask = 999_000  # Bid exceeds ask
        (bo, so), (br, sr), done, info = env.step(0, 0)
        assert done is True
        assert env.agreed_price is not None

    def test_timeout_after_max_rounds(self):
        env = self._make_env()
        env.reset()
        env.max_rounds = 3
        for _ in range(3):
            env.step(0, 0)  # Both hold
        assert env.done is True

    def test_cooperative_resilience_positive_on_deal(self):
        env = self._make_env()
        env.reset()
        env.current_bid = 1_000_000
        env.current_ask = 1_000_000
        env.step(5, 5)
        assert env._cooperative_resilience() >= 0


class TestMAPPOAgent:
    def test_select_action_returns_valid(self):
        from app.agents.negotiation_engine import MAPPOAgent, NUM_ACTIONS
        agent = MAPPOAgent(role="buyer")
        obs = np.random.randn(18).astype(np.float32)
        action, log_prob = agent.select_action(obs)
        assert 0 <= action < NUM_ACTIONS
        assert isinstance(log_prob, float)

    def test_get_value_returns_scalar(self):
        from app.agents.negotiation_engine import MAPPOAgent
        agent = MAPPOAgent(role="seller")
        global_obs = np.random.randn(36).astype(np.float32)
        value = agent.get_value(global_obs)
        assert isinstance(value, float)


class TestAlternativeAlgorithms:
    def test_qmix_agent_selects_actions(self):
        from app.agents.marl_algorithms import QMIXAgent
        agent = QMIXAgent()
        obs = [np.random.randn(18).astype(np.float32) for _ in range(2)]
        actions = agent.select_actions(obs)
        assert len(actions) == 2
        assert all(0 <= a < 7 for a in actions)

    def test_iql_agent_selects_action(self):
        from app.agents.marl_algorithms import IQLAgent
        agent = IQLAgent(role="buyer")
        obs = np.random.randn(18).astype(np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < 7

    def test_sac_agent_selects_action(self):
        from app.agents.marl_algorithms import SACAgent
        agent = SACAgent(role="seller")
        obs = np.random.randn(18).astype(np.float32)
        action = agent.select_action(obs)
        assert 0 <= action < 7


# ══════════════════════════════════════════════════════════════════════
# §2  IoT / PMV TESTS
# ══════════════════════════════════════════════════════════════════════

class TestPMVCalculator:
    def test_neutral_comfort(self):
        from app.iot.sensor_overlay import PMVCalculator
        pmv, ppd = PMVCalculator.calculate_pmv(
            air_temp_c=23.0, relative_humidity_pct=50.0
        )
        assert -1.0 <= pmv <= 1.0
        assert ppd < 30

    def test_hot_conditions(self):
        from app.iot.sensor_overlay import PMVCalculator
        pmv, ppd = PMVCalculator.calculate_pmv(
            air_temp_c=35.0, relative_humidity_pct=70.0
        )
        assert pmv > 1.5
        assert ppd > 50

    def test_cold_conditions(self):
        from app.iot.sensor_overlay import PMVCalculator
        pmv, ppd = PMVCalculator.calculate_pmv(
            air_temp_c=15.0, relative_humidity_pct=30.0
        )
        assert pmv < -1.0

    def test_comfort_categories(self):
        from app.iot.sensor_overlay import PMVCalculator
        assert PMVCalculator.comfort_category(0.0) == "excellent"
        assert PMVCalculator.comfort_category(0.8) == "good"
        assert PMVCalculator.comfort_category(1.3) == "moderate"
        assert PMVCalculator.comfort_category(2.5) == "extreme"


class TestVirtualSensing:
    def test_interpolation_produces_grid(self):
        from app.iot.sensor_overlay import VirtualSensingEngine, SensorStation, SensorReading, SensorType
        import time
        stations = [
            SensorStation("s1", "living_room", (2.0, 2.0, 1.0),
                         {SensorType.TEMPERATURE: SensorReading("s1_t", SensorType.TEMPERATURE, 24.0, "°C", time.time(), (2.0, 2.0, 1.0), "living_room")}),
            SensorStation("s2", "bedroom", (8.0, 8.0, 1.0),
                         {SensorType.TEMPERATURE: SensorReading("s2_t", SensorType.TEMPERATURE, 22.0, "°C", time.time(), (8.0, 8.0, 1.0), "bedroom")}),
        ]
        grid = VirtualSensingEngine.interpolate_grid(
            stations, SensorType.TEMPERATURE, grid_resolution=(10, 10)
        )
        assert grid.shape == (10, 10)
        assert grid.min() >= 21.0
        assert grid.max() <= 25.0


class TestMQTTSimulation:
    def test_simulation_generates_stations(self):
        from app.iot.sensor_overlay import MQTTSensorClient
        client = MQTTSensorClient()
        stations = client.get_stations()
        assert len(stations) >= 4
        for station in stations.values():
            assert len(station.sensors) >= 3


# ══════════════════════════════════════════════════════════════════════
# §3  INSPECTION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestYOLOv12Detector:
    def test_simulated_detection(self):
        from app.inspection.defect_detector import YOLOv12Detector
        detector = YOLOv12Detector()
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        defects = asyncio.get_event_loop().run_until_complete(
            detector.detect(image, "test_001", "north wall")
        )
        assert isinstance(defects, list)
        for d in defects:
            assert d.confidence >= 0.0
            assert d.confidence <= 1.0
            assert len(d.bbox) == 4

    def test_defect_categories_complete(self):
        from app.inspection.defect_detector import DefectCategory
        assert len(DefectCategory) == 54


class TestCracksGPT:
    def test_fallback_analysis(self):
        from app.inspection.defect_detector import CracksGPT, DetectedDefect, DefectCategory, DefectSeverity
        gpt = CracksGPT()
        defect = DetectedDefect(
            defect_id="test", category=DefectCategory.CRACK_DIAGONAL,
            severity=DefectSeverity.MEDIUM, confidence=0.85,
            bbox=(100, 100, 200, 200), location_description="test wall",
            width_mm=0.5,
        )
        result = gpt._fallback_analysis(defect)
        assert "is_structural" in result
        assert "remediation_level" in result
        assert result["is_structural"] is True

    def test_thermal_anomaly_detection(self):
        from app.inspection.defect_detector import ThermalFusion
        thermal = np.full((100, 100), 25.0)
        thermal[20:40, 30:50] = 18.0  # Cold spot
        anomalies = ThermalFusion.detect_thermal_anomalies(thermal, ambient_temp_c=25.0, threshold_delta_c=3.0)
        assert len(anomalies) >= 1
        assert anomalies[0]["type"] == "cold_spot"


# ══════════════════════════════════════════════════════════════════════
# §4  ZKP TESTS
# ══════════════════════════════════════════════════════════════════════

class TestZKPVerifier:
    def test_create_commitment(self):
        from app.circuits.zkp_verifier import ZKPVerifier
        c1 = ZKPVerifier.create_balance_commitment(750000, "salt1")
        c2 = ZKPVerifier.create_balance_commitment(750000, "salt2")
        assert c1 != c2  # Different salts = different commitments
        assert len(c1) == 64  # SHA-256 hex

    def test_mock_proof_generation(self):
        from app.circuits.zkp_verifier import ZKPVerifier
        verifier = ZKPVerifier()
        result = verifier._mock_proof("buyer_1", 500000)
        assert result["success"] is True
        assert len(result["proof_hash"]) == 64
        assert result["proof_size_bytes"] == 288

    def test_proof_expiry(self):
        from app.circuits.zkp_verifier import ZKPVerifier
        from datetime import datetime, timezone
        expiry = ZKPVerifier.proof_expiry()
        now = datetime.now(timezone.utc)
        delta = (expiry - now).total_seconds()
        assert 71 * 3600 < delta < 73 * 3600  # ~72 hours


# ══════════════════════════════════════════════════════════════════════
# §5  COMPLIANCE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestRON:
    def test_jurisdiction_check(self):
        from app.services.compliance import RONService
        ron = RONService()
        result = ron.check_jurisdiction("FL")
        assert result["ron_available"] is True
        assert result["interstate_recognition"] is True

    def test_non_ron_state(self):
        from app.services.compliance import RONService
        ron = RONService()
        # Check a state not in the list (if any were excluded)
        result = ron.check_jurisdiction("XX")
        assert result["ron_available"] is False


class TestIRS1099S:
    def test_form_generation(self):
        from app.services.compliance import IRS1099SService
        svc = IRS1099SService()
        form = svc.generate_form(
            seller_name="John Doe", seller_tin="123456789",
            property_address="123 Main St, Miami, FL",
            gross_proceeds=3500000.0, closing_date="2026-03-15",
        )
        assert form.gross_proceeds == 3500000.0
        assert form.settlement_type == "fiat"

    def test_form_validation(self):
        from app.services.compliance import IRS1099SService, Form1099S
        svc = IRS1099SService()
        form = Form1099S(
            filer_name="PropOS", filer_tin="XX-XXXXXXX", filer_address="Miami",
            transferor_name="", transferor_tin="", transferor_address="",
            date_of_closing="2026-01-01", gross_proceeds=0,
            address_or_description="",
        )
        result = svc.validate_form(form)
        assert result["valid"] is False
        assert len(result["errors"]) >= 2

    def test_crypto_settlement_requires_fmv(self):
        from app.services.compliance import IRS1099SService, Form1099S
        svc = IRS1099SService()
        form = Form1099S(
            filer_name="PropOS", filer_tin="XX-XXXXXXX", filer_address="Miami",
            transferor_name="Jane", transferor_tin="987654321",
            transferor_address="456 Oak Ave",
            date_of_closing="2026-06-01", gross_proceeds=2000000,
            address_or_description="456 Oak Ave",
            settlement_type="crypto", crypto_fmv_usd=None,
        )
        result = svc.validate_form(form)
        assert not result["valid"]


# ══════════════════════════════════════════════════════════════════════
# §6  GSplat PERCEPTION TESTS
# ══════════════════════════════════════════════════════════════════════

class TestGSplat:
    def test_edge_optimizer_sh_downgrade(self):
        import torch, torch.nn as nn
        from app.perception.gsplat_pipeline import EdgeOptimizer, SHDegree
        params = {"sh_coeffs": nn.Parameter(torch.randn(100, 16, 3))}
        result = EdgeOptimizer.downgrade_sh_degree(params, SHDegree.DEGREE_0)
        assert result["sh_coeffs"].shape[1] == 1  # Degree 0 = 1 coeff

    def test_deployment_stats(self):
        import torch, torch.nn as nn
        from app.perception.gsplat_pipeline import EdgeOptimizer, SHDegree
        params = {"positions": nn.Parameter(torch.randn(50000, 3))}
        stats = EdgeOptimizer.compute_deployment_stats(params, SHDegree.DEGREE_0)
        assert stats["num_gaussians"] == 50000
        assert stats["mobile_compatible"] is True

    def test_speedy_splat_pruning(self):
        import torch, torch.nn as nn
        from app.perception.gsplat_pipeline import SpeedySplat
        params = {
            "positions": nn.Parameter(torch.randn(1000, 3)),
            "scales": nn.Parameter(torch.randn(1000, 3)),
            "opacities": nn.Parameter(torch.randn(1000, 1)),
        }
        scores = torch.rand(1000)
        pruned, stats = SpeedySplat.prune(params, scores, prune_ratio=0.9)
        assert pruned["positions"].shape[0] == 100
        assert stats["reduction_pct"] == 90.0
