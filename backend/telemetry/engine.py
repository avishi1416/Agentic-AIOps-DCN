"""
Telemetry Engine — orchestrates data loading and synthetic generation.
Attempts to load RCAEval/OpenRCA datasets; falls back to synthetic generation.
"""

import os
import pandas as pd
from typing import Optional, List, Tuple
from .synthetic import SyntheticTelemetryGenerator
from .incidents import IncidentSimulator, Incident


class TelemetryEngine:
    """Main telemetry ingestion engine."""

    REQUIRED_COLS = ["timestamp", "service_id", "cpu_usage", "memory_usage", "latency", "error_rate"]

    def __init__(self, config=None):
        from config import TelemetryConfig
        self.config = config or TelemetryConfig()
        self.generator = SyntheticTelemetryGenerator(
            num_services=self.config.num_services,
            duration_seconds=self.config.duration_seconds,
            sampling_interval=self.config.sampling_interval_seconds,
            baselines={"cpu_usage": self.config.cpu_baseline, "memory_usage": self.config.memory_baseline,
                        "latency": self.config.latency_baseline, "error_rate": self.config.error_rate_baseline},
            noise_stds={"cpu_usage": self.config.cpu_noise_std, "memory_usage": self.config.memory_noise_std,
                         "latency": self.config.latency_noise_std, "error_rate": self.config.error_rate_noise_std})
        self.incident_sim = IncidentSimulator()
        self.raw_telemetry: Optional[pd.DataFrame] = None
        self.telemetry: Optional[pd.DataFrame] = None
        self.incidents: List[Incident] = []

    def load_dataset(self, path: str) -> Optional[pd.DataFrame]:
        """Try loading RCAEval/OpenRCA dataset from CSV."""
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            if all(c in df.columns for c in self.REQUIRED_COLS):
                self.raw_telemetry = df
                self.telemetry = df.copy()
                return df
        except Exception:
            pass
        return None

    def generate_synthetic(self, with_incidents: bool = True) -> Tuple[pd.DataFrame, List[Incident]]:
        """Generate synthetic telemetry with optional incident injection."""
        self.raw_telemetry = self.generator.generate()
        if with_incidents:
            self.telemetry, self.incidents = self.incident_sim.inject_incidents(
                self.raw_telemetry, self.config.incident_probability, self.config.incident_types)
        else:
            self.telemetry = self.raw_telemetry.copy()
            self.incidents = []
        return self.telemetry, self.incidents

    def get_service_telemetry(self, service_id: str) -> pd.DataFrame:
        """Get telemetry for a specific service."""
        if self.telemetry is None:
            raise ValueError("No telemetry data. Call generate_synthetic() or load_dataset() first.")
        return self.telemetry[self.telemetry["service_id"] == service_id].reset_index(drop=True)

    def get_telemetry_at(self, timestamp: float) -> pd.DataFrame:
        """Get telemetry snapshot at a specific timestamp."""
        if self.telemetry is None:
            raise ValueError("No telemetry data.")
        return self.telemetry[self.telemetry["timestamp"] == timestamp].reset_index(drop=True)
