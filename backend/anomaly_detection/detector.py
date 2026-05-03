"""
Anomaly Detection Engine.

Implements:
- Threshold-based anomaly detection (static bounds)
- Rolling statistical deviation checks (z-score based)
- Anomaly severity tagging: normal / warning / critical

Generates anomaly events dynamically from telemetry streams.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AnomalyState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """Single anomaly event."""
    timestamp: float
    service_id: str
    state: AnomalyState
    triggered_metrics: List[str]
    details: Dict[str, float] = field(default_factory=dict)


class AnomalyDetector:
    """
    Multi-method anomaly detector combining threshold and statistical approaches.
    
    Detection logic:
    1. Static thresholds for immediate boundary violations
    2. Rolling z-score for statistical deviation from recent behavior
    3. Combined severity = max(threshold_severity, zscore_severity)
    """

    def __init__(self, config=None):
        from config import AnomalyConfig
        self.config = config or AnomalyConfig()
        self._thresholds = {
            "cpu_usage": (self.config.cpu_warning_threshold, self.config.cpu_critical_threshold),
            "memory_usage": (self.config.memory_warning_threshold, self.config.memory_critical_threshold),
            "latency": (self.config.latency_warning_threshold, self.config.latency_critical_threshold),
            "error_rate": (self.config.error_rate_warning_threshold, self.config.error_rate_critical_threshold),
        }
        self.events: List[AnomalyEvent] = []

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection on a telemetry DataFrame.

        Adds columns: anomaly_state, anomaly_score (0-1 composite severity).
        Returns the annotated DataFrame.
        """
        df = df.copy()
        states = []
        scores = []

        for service_id in df["service_id"].unique():
            sdf = df[df["service_id"] == service_id].copy()
            sdf = sdf.sort_values("timestamp").reset_index(drop=True)
            n = len(sdf)

            service_states = [AnomalyState.NORMAL] * n
            service_scores = np.zeros(n)

            for i in range(n):
                row = sdf.iloc[i]
                triggered = []
                max_severity = 0  # 0=normal, 1=warning, 2=critical

                # --- Threshold-based detection ---
                for metric, (warn_th, crit_th) in self._thresholds.items():
                    val = row[metric]
                    if val >= crit_th:
                        max_severity = max(max_severity, 2)
                        triggered.append(metric)
                    elif val >= warn_th:
                        max_severity = max(max_severity, 1)
                        triggered.append(metric)

                # --- Rolling z-score detection ---
                window = self.config.rolling_window_size
                if i >= window:
                    for metric in ["cpu_usage", "memory_usage", "latency", "error_rate"]:
                        window_data = sdf[metric].iloc[max(0, i - window):i].values
                        mu = np.mean(window_data)
                        sigma = np.std(window_data)
                        if sigma > 1e-10:
                            z = abs(row[metric] - mu) / sigma
                            if z >= self.config.z_score_critical:
                                max_severity = max(max_severity, 2)
                                if metric not in triggered:
                                    triggered.append(metric)
                            elif z >= self.config.z_score_warning:
                                max_severity = max(max_severity, 1)
                                if metric not in triggered:
                                    triggered.append(metric)

                # Map severity level
                state = [AnomalyState.NORMAL, AnomalyState.WARNING, AnomalyState.CRITICAL][max_severity]
                service_states[i] = state
                service_scores[i] = max_severity / 2.0  # normalize to [0, 1]

                if state != AnomalyState.NORMAL:
                    self.events.append(AnomalyEvent(
                        timestamp=row["timestamp"], service_id=service_id,
                        state=state, triggered_metrics=triggered,
                        details={m: row[m] for m in triggered}))

            # Write back to df
            mask = df["service_id"] == service_id
            df.loc[mask, "anomaly_state"] = [s.value for s in service_states]
            df.loc[mask, "anomaly_score"] = service_scores

        return df

    def get_events(self, service_id: Optional[str] = None) -> List[AnomalyEvent]:
        """Retrieve anomaly events, optionally filtered by service."""
        if service_id:
            return [e for e in self.events if e.service_id == service_id]
        return self.events
