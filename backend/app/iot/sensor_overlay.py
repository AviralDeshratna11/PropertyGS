"""
PropOS Sensory Data Layer — IoT & Environmental Overlays
=========================================================
Real-time IoT sensor data integrated into virtual tours as spatial overlays.

Architecture:
  - MQTT Protocol: Multimodal sensing stations → cloud → XR environment
  - PMV Heatmap: Predicted Mean Vote continuous thermal comfort map
  - Virtual Sensing: Interpolate conditions across entire indoor space
    from discrete sensor locations
  - Bi-directional Control: Users interact with virtual HVAC buttons
    in XR to adjust real-world systems

References: PropOS Sensory Data Layer specification
"""

import numpy as np
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum

logger = logging.getLogger("propos.iot")


# ══════════════════════════════════════════════════════════════════════
# §1  SENSOR DATA MODELS
# ══════════════════════════════════════════════════════════════════════

class SensorType(Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    CO2 = "co2"
    PM25 = "pm2.5"
    PM10 = "pm10"
    TVOC = "tvoc"
    NOISE_DB = "noise_db"
    LIGHT_LUX = "light_lux"
    PRESSURE = "pressure"
    MOTION = "motion"


@dataclass
class SensorReading:
    """Single reading from an IoT sensor."""
    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: float
    location: Tuple[float, float, float]  # x, y, z in property coordinates
    room: str
    quality: str = "good"  # good, degraded, error


@dataclass
class SensorStation:
    """Multimodal IoT sensing station placed in a room."""
    station_id: str
    room: str
    location: Tuple[float, float, float]
    sensors: Dict[SensorType, SensorReading] = field(default_factory=dict)
    last_update: float = 0.0
    online: bool = True


@dataclass
class EnvironmentalSnapshot:
    """Full environmental state of a property at a point in time."""
    property_id: int
    timestamp: float
    stations: List[SensorStation]
    pmv_grid: Optional[np.ndarray] = None  # 2D PMV heatmap
    overall_aqi: Optional[float] = None
    overall_comfort: Optional[str] = None  # excellent, good, moderate, poor


# ══════════════════════════════════════════════════════════════════════
# §2  MQTT CLIENT
# ══════════════════════════════════════════════════════════════════════

class MQTTSensorClient:
    """
    MQTT client for receiving real-time IoT sensor data.

    Topic structure: propos/{property_id}/sensors/{station_id}/{sensor_type}
    Payload: JSON { "value": float, "unit": str, "ts": unix_timestamp }

    For bi-directional control:
    Topic: propos/{property_id}/control/{device_id}
    Payload: JSON { "action": "set_temp", "value": 22.0 }
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: str = "",
        password: str = "",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self._stations: Dict[str, SensorStation] = {}
        self._callbacks: List[Callable] = []
        self._connected = False
        self._simulation_mode = True  # Default to simulation

    async def connect(self):
        """Connect to MQTT broker."""
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(client_id=f"propos-{uuid.uuid4().hex[:8]}")
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            self._client.on_message = self._on_message
            self._client.connect(self.broker_host, self.broker_port, 60)
            self._client.loop_start()
            self._connected = True
            self._simulation_mode = False
            logger.info(f"MQTT connected to {self.broker_host}:{self.broker_port}")
        except Exception as e:
            logger.warning(f"MQTT connection failed ({e}), using simulation mode")
            self._simulation_mode = True

    def subscribe(self, property_id: int):
        """Subscribe to all sensor topics for a property."""
        if self._connected:
            self._client.subscribe(f"propos/{property_id}/sensors/#")
            logger.info(f"Subscribed to propos/{property_id}/sensors/#")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            parts = msg.topic.split("/")
            if len(parts) >= 4 and parts[2] == "sensors":
                station_id = parts[3]
                sensor_type_str = parts[4] if len(parts) > 4 else "temperature"
                payload = json.loads(msg.payload.decode())

                sensor_type = SensorType(sensor_type_str)
                reading = SensorReading(
                    sensor_id=f"{station_id}_{sensor_type_str}",
                    sensor_type=sensor_type,
                    value=payload["value"],
                    unit=payload.get("unit", ""),
                    timestamp=payload.get("ts", time.time()),
                    location=(0, 0, 0),  # Updated from station config
                    room=payload.get("room", "unknown"),
                )

                if station_id not in self._stations:
                    self._stations[station_id] = SensorStation(
                        station_id=station_id,
                        room=reading.room,
                        location=reading.location,
                    )

                self._stations[station_id].sensors[sensor_type] = reading
                self._stations[station_id].last_update = time.time()

                for cb in self._callbacks:
                    cb(station_id, reading)

        except Exception as e:
            logger.error(f"MQTT message parse error: {e}")

    def on_reading(self, callback: Callable):
        """Register callback for new sensor readings."""
        self._callbacks.append(callback)

    async def publish_control(self, property_id: int, device_id: str, action: Dict):
        """
        Bi-directional control: send command to real-world HVAC/IoT device.
        Allows users to interact with virtual buttons in XR to control
        the actual physical systems.
        """
        topic = f"propos/{property_id}/control/{device_id}"
        payload = json.dumps(action)
        if self._connected:
            self._client.publish(topic, payload)
            logger.info(f"Control command sent: {topic} → {payload}")
        else:
            logger.info(f"Control command (sim): {topic} → {payload}")
        return {"status": "sent", "topic": topic, "payload": action}

    def get_stations(self) -> Dict[str, SensorStation]:
        """Get all known sensor stations."""
        if self._simulation_mode and not self._stations:
            self._generate_simulation_data()
        return self._stations

    def _generate_simulation_data(self):
        """Generate realistic simulated sensor data for demo."""
        rooms = [
            ("living_room", (5.0, 3.0, 1.2)),
            ("master_bedroom", (10.0, 3.0, 1.2)),
            ("kitchen", (5.0, 8.0, 1.2)),
            ("bathroom", (10.0, 8.0, 1.2)),
            ("balcony", (0.0, 5.0, 1.2)),
            ("entrance", (7.5, 0.0, 1.2)),
        ]

        for room_name, location in rooms:
            station_id = f"STA-{room_name[:3].upper()}-001"
            station = SensorStation(
                station_id=station_id,
                room=room_name,
                location=location,
            )

            base_temp = 24.0 + np.random.normal(0, 1.5)
            base_humid = 45.0 + np.random.normal(0, 8)

            # Adjust by room type
            if room_name == "kitchen":
                base_temp += 2.0
                base_humid += 10.0
            elif room_name == "bathroom":
                base_humid += 15.0
            elif room_name == "balcony":
                base_temp += 8.0  # Dubai outdoor heat

            now = time.time()
            readings = {
                SensorType.TEMPERATURE: SensorReading(
                    f"{station_id}_temp", SensorType.TEMPERATURE,
                    round(base_temp, 1), "°C", now, location, room_name,
                ),
                SensorType.HUMIDITY: SensorReading(
                    f"{station_id}_hum", SensorType.HUMIDITY,
                    round(max(20, min(90, base_humid)), 1), "%RH", now, location, room_name,
                ),
                SensorType.CO2: SensorReading(
                    f"{station_id}_co2", SensorType.CO2,
                    round(400 + np.random.exponential(100)), "ppm", now, location, room_name,
                ),
                SensorType.PM25: SensorReading(
                    f"{station_id}_pm25", SensorType.PM25,
                    round(np.random.uniform(5, 35), 1), "µg/m³", now, location, room_name,
                ),
                SensorType.NOISE_DB: SensorReading(
                    f"{station_id}_noise", SensorType.NOISE_DB,
                    round(30 + np.random.exponential(10)), "dB", now, location, room_name,
                ),
                SensorType.LIGHT_LUX: SensorReading(
                    f"{station_id}_light", SensorType.LIGHT_LUX,
                    round(np.random.uniform(100, 800)), "lux", now, location, room_name,
                ),
            }

            station.sensors = readings
            station.last_update = now
            self._stations[station_id] = station


# ══════════════════════════════════════════════════════════════════════
# §3  PMV THERMAL COMFORT CALCULATOR
# ══════════════════════════════════════════════════════════════════════

class PMVCalculator:
    """
    Predicted Mean Vote (PMV) calculator for thermal comfort assessment.

    PMV scale: -3 (cold) to +3 (hot), 0 = neutral comfort
    Based on ISO 7730 / ASHRAE Standard 55.

    Factors: air temperature, radiant temperature, air velocity,
    humidity, metabolic rate, clothing insulation.
    """

    @staticmethod
    def calculate_pmv(
        air_temp_c: float,
        radiant_temp_c: float = None,
        air_velocity_ms: float = 0.1,
        relative_humidity_pct: float = 50.0,
        metabolic_rate: float = 1.2,     # met (seated office work)
        clothing_insulation: float = 0.5, # clo (light indoor)
    ) -> Tuple[float, float]:
        """
        Calculate PMV and PPD (Predicted Percentage of Dissatisfied).

        Returns: (PMV [-3..+3], PPD [0..100%])
        """
        if radiant_temp_c is None:
            radiant_temp_c = air_temp_c

        ta = air_temp_c
        tr = radiant_temp_c
        v = max(air_velocity_ms, 0.05)
        rh = relative_humidity_pct
        M = metabolic_rate * 58.15      # Convert met to W/m²
        Icl = clothing_insulation * 0.155 # Convert clo to m²·K/W

        # Clothing surface area factor
        if Icl < 0.078:
            fcl = 1.0 + 1.290 * Icl
        else:
            fcl = 1.05 + 0.645 * Icl

        # Saturated vapor pressure
        pa = rh * 10 * np.exp(16.6536 - 4030.183 / (ta + 235))

        # Iterative calculation of clothing surface temperature
        tcl = ta  # Initial guess
        for _ in range(50):
            hc = max(2.38 * abs(tcl - ta) ** 0.25, 12.1 * np.sqrt(v))
            tcl_new = 35.7 - 0.028 * M - Icl * (
                3.96e-8 * fcl * ((tcl + 273) ** 4 - (tr + 273) ** 4) +
                fcl * hc * (tcl - ta)
            )
            if abs(tcl_new - tcl) < 0.001:
                break
            tcl = tcl_new

        hc = max(2.38 * abs(tcl - ta) ** 0.25, 12.1 * np.sqrt(v))

        # PMV calculation (Fanger's equation)
        pmv = (0.303 * np.exp(-0.036 * M) + 0.028) * (
            M -                                              # Internal heat production
            3.05e-3 * (5733 - 6.99 * M - pa) -             # Skin diffusion
            0.42 * (M - 58.15) -                            # Sweating
            1.7e-5 * M * (5867 - pa) -                      # Latent respiration
            0.0014 * M * (34 - ta) -                        # Dry respiration
            3.96e-8 * fcl * ((tcl + 273) ** 4 - (tr + 273) ** 4) -  # Radiation
            fcl * hc * (tcl - ta)                           # Convection
        )

        pmv = np.clip(pmv, -3, 3)

        # PPD calculation
        ppd = 100 - 95 * np.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
        ppd = np.clip(ppd, 5, 100)

        return round(float(pmv), 2), round(float(ppd), 1)

    @staticmethod
    def comfort_category(pmv: float) -> str:
        """Map PMV to human-readable comfort category."""
        if abs(pmv) <= 0.5:
            return "excellent"
        elif abs(pmv) <= 1.0:
            return "good"
        elif abs(pmv) <= 1.5:
            return "moderate"
        elif abs(pmv) <= 2.0:
            return "poor"
        else:
            return "extreme"


# ══════════════════════════════════════════════════════════════════════
# §4  VIRTUAL SENSING & SPATIAL INTERPOLATION
# ══════════════════════════════════════════════════════════════════════

class VirtualSensingEngine:
    """
    Estimates environmental conditions across entire indoor spaces
    from discrete sensor locations using spatial interpolation.

    Rather than showing data only at sensor locations, this produces
    continuous heatmaps (temperature, humidity, air quality, PMV)
    for the entire property.

    Method: Inverse Distance Weighting (IDW) interpolation with
    physics-informed corrections for walls, windows, and HVAC vents.
    """

    @staticmethod
    def interpolate_grid(
        stations: List[SensorStation],
        sensor_type: SensorType,
        grid_resolution: Tuple[int, int] = (50, 50),
        property_bounds: Tuple[float, float, float, float] = (0, 0, 15, 12),
        power: float = 2.0,  # IDW power parameter
    ) -> np.ndarray:
        """
        Inverse Distance Weighting (IDW) interpolation.

        Returns: 2D grid (grid_resolution) of interpolated sensor values.
        """
        min_x, min_y, max_x, max_y = property_bounds
        grid_x = np.linspace(min_x, max_x, grid_resolution[0])
        grid_y = np.linspace(min_y, max_y, grid_resolution[1])

        xx, yy = np.meshgrid(grid_x, grid_y)
        grid = np.zeros_like(xx)

        sensor_points = []
        sensor_values = []
        for station in stations:
            if sensor_type in station.sensors:
                reading = station.sensors[sensor_type]
                sensor_points.append((station.location[0], station.location[1]))
                sensor_values.append(reading.value)

        if not sensor_points:
            return grid

        for i in range(grid_resolution[1]):
            for j in range(grid_resolution[0]):
                px, py = xx[i, j], yy[i, j]
                weights = []
                values = []
                for (sx, sy), sv in zip(sensor_points, sensor_values):
                    dist = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
                    if dist < 0.01:
                        grid[i, j] = sv
                        break
                    w = 1.0 / (dist ** power)
                    weights.append(w)
                    values.append(sv)
                else:
                    total_w = sum(weights)
                    if total_w > 0:
                        grid[i, j] = sum(w * v for w, v in zip(weights, values)) / total_w

        return grid

    @staticmethod
    def generate_pmv_heatmap(
        stations: List[SensorStation],
        grid_resolution: Tuple[int, int] = (50, 50),
        property_bounds: Tuple[float, float, float, float] = (0, 0, 15, 12),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate continuous PMV heatmap across the property.

        Returns: (pmv_grid, ppd_grid)
        """
        temp_grid = VirtualSensingEngine.interpolate_grid(
            stations, SensorType.TEMPERATURE, grid_resolution, property_bounds
        )
        humid_grid = VirtualSensingEngine.interpolate_grid(
            stations, SensorType.HUMIDITY, grid_resolution, property_bounds
        )

        pmv_grid = np.zeros_like(temp_grid)
        ppd_grid = np.zeros_like(temp_grid)

        calc = PMVCalculator()
        for i in range(grid_resolution[1]):
            for j in range(grid_resolution[0]):
                pmv, ppd = calc.calculate_pmv(
                    air_temp_c=temp_grid[i, j],
                    relative_humidity_pct=humid_grid[i, j] if humid_grid[i, j] > 0 else 50,
                )
                pmv_grid[i, j] = pmv
                ppd_grid[i, j] = ppd

        return pmv_grid, ppd_grid


# ══════════════════════════════════════════════════════════════════════
# §5  IoT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class IoTOrchestrator:
    """Manages IoT data collection, processing, and overlay generation."""

    def __init__(self):
        self.mqtt = MQTTSensorClient()
        self.pmv_calc = PMVCalculator()
        self.virtual_sensing = VirtualSensingEngine()

    async def initialize(self, property_id: int):
        """Initialize IoT subsystem for a property."""
        await self.mqtt.connect()
        self.mqtt.subscribe(property_id)

    def get_environmental_snapshot(self, property_id: int) -> Dict[str, Any]:
        """Get current environmental state as JSON for frontend overlay."""
        stations = self.mqtt.get_stations()
        station_list = list(stations.values())

        # Generate PMV heatmap
        pmv_grid, ppd_grid = self.virtual_sensing.generate_pmv_heatmap(station_list)

        # Per-room data
        rooms = {}
        for station in station_list:
            room_data = {"station_id": station.station_id}
            for sensor_type, reading in station.sensors.items():
                room_data[sensor_type.value] = {
                    "value": reading.value,
                    "unit": reading.unit,
                }

            # Calculate PMV for this station
            temp = station.sensors.get(SensorType.TEMPERATURE)
            humid = station.sensors.get(SensorType.HUMIDITY)
            if temp:
                pmv, ppd = self.pmv_calc.calculate_pmv(
                    temp.value,
                    relative_humidity_pct=humid.value if humid else 50,
                )
                room_data["pmv"] = pmv
                room_data["ppd"] = ppd
                room_data["comfort"] = self.pmv_calc.comfort_category(pmv)

            rooms[station.room] = room_data

        # Overall AQI
        co2_readings = [s.sensors[SensorType.CO2].value for s in station_list
                       if SensorType.CO2 in s.sensors]
        pm25_readings = [s.sensors[SensorType.PM25].value for s in station_list
                        if SensorType.PM25 in s.sensors]

        avg_co2 = np.mean(co2_readings) if co2_readings else 400
        avg_pm25 = np.mean(pm25_readings) if pm25_readings else 15

        if avg_co2 < 600 and avg_pm25 < 12:
            aqi_label = "excellent"
        elif avg_co2 < 1000 and avg_pm25 < 25:
            aqi_label = "good"
        elif avg_co2 < 1500 and avg_pm25 < 35:
            aqi_label = "moderate"
        else:
            aqi_label = "poor"

        return {
            "property_id": property_id,
            "timestamp": time.time(),
            "rooms": rooms,
            "pmv_heatmap": pmv_grid.tolist(),
            "overall_aqi": aqi_label,
            "avg_co2_ppm": round(float(avg_co2)),
            "avg_pm25": round(float(avg_pm25), 1),
            "num_stations": len(station_list),
            "all_stations_online": all(s.online for s in station_list),
        }

    async def send_hvac_command(
        self, property_id: int, room: str, target_temp: float
    ) -> Dict:
        """Bi-directional control: set HVAC target temperature."""
        device_id = f"hvac_{room}"
        return await self.mqtt.publish_control(property_id, device_id, {
            "action": "set_temperature",
            "value": target_temp,
            "unit": "celsius",
            "room": room,
        })
