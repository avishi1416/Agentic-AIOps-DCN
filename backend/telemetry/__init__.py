"""Telemetry ingestion engine — synthetic generation and dataset loading."""

from .engine import TelemetryEngine
from .synthetic import SyntheticTelemetryGenerator
from .incidents import IncidentSimulator

__all__ = ["TelemetryEngine", "SyntheticTelemetryGenerator", "IncidentSimulator"]
