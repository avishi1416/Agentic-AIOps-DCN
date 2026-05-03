"""
Normalized Severity Function.

Implements: E_n(t) = w_c*(CPU/CPU_max)^2 + w_l*(Latency/L_thr)^2 + w_m*(Memory/M_max)^2 + w_e*log(1+ErrorRate)

Features:
- All metrics normalized between 0 and 1
- Configurable telemetry weights
- Adaptive telemetry weighting based on anomaly feedback
- Latency contributes more aggressively (higher default weight + quadratic)
- Logarithmic scaling for error rates
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class SeverityComputer:
    """Computes instantaneous operational degradation severity."""

    def __init__(self, config=None):
        from config import SeverityConfig
        self.config = config or SeverityConfig()
        self.weights = {
            "cpu": self.config.w_cpu,
            "latency": self.config.w_latency,
            "memory": self.config.w_memory,
            "error": self.config.w_error,
        }
        self._weight_history = [self.weights.copy()]

    def compute(self, cpu: float, latency: float, memory: float, error_rate: float) -> float:
        """
        Compute normalized severity E_n(t) for a single observation.

        E_n(t) = w_c*(CPU/CPU_max)^2 + w_l*(Lat/L_thr)^2 + w_m*(Mem/M_max)^2 + w_e*log(1+err)

        Returns a value in [0, ~1] (can slightly exceed 1 under extreme conditions).
        """
        cpu_norm = np.clip(cpu / self.config.cpu_max, 0, 1)
        lat_norm = np.clip(latency / self.config.latency_threshold, 0, 1)
        mem_norm = np.clip(memory / self.config.memory_max, 0, 1)
        # log(1 + error_rate) with error_rate in [0,1] gives [0, log(2)] ≈ [0, 0.693]
        # Normalize to [0, 1] by dividing by log(2)
        err_norm = np.log(1 + np.clip(error_rate, 0, 1)) / np.log(2)

        severity = (
            self.weights["cpu"] * cpu_norm ** 2
            + self.weights["latency"] * lat_norm ** 2
            + self.weights["memory"] * mem_norm ** 2
            + self.weights["error"] * err_norm
        )
        return float(np.clip(severity, 0, 1))

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        """Compute severity for an entire DataFrame. Adds 'severity' column."""
        severities = df.apply(
            lambda r: self.compute(r["cpu_usage"], r["latency"], r["memory_usage"], r["error_rate"]),
            axis=1)
        return severities

    def adapt_weights(self, anomaly_counts: Dict[str, int], adaptation_rate: Optional[float] = None):
        """
        Adapt weights based on which metrics trigger the most anomalies.
        
        Metrics that cause more anomalies receive higher weights, ensuring
        the severity function focuses on the most problematic dimensions.
        
        Uses exponential moving average for smooth adaptation.
        """
        rate = adaptation_rate or self.config.weight_adaptation_rate
        total = sum(anomaly_counts.values()) or 1
        
        metric_map = {
            "cpu_usage": "cpu", "latency": "latency",
            "memory_usage": "memory", "error_rate": "error"
        }
        
        target_weights = {}
        for metric_name, weight_key in metric_map.items():
            freq = anomaly_counts.get(metric_name, 0) / total
            target_weights[weight_key] = freq

        # EMA update
        for key in self.weights:
            self.weights[key] = (1 - rate) * self.weights[key] + rate * target_weights.get(key, self.weights[key])

        # Renormalize to sum to 1
        w_sum = sum(self.weights.values())
        if w_sum > 0:
            for key in self.weights:
                self.weights[key] /= w_sum

        self._weight_history.append(self.weights.copy())

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()
