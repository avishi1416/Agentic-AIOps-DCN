"""
Stability Estimation Engine.

Implements: S_n(t) = exp(-(mu_E + sigma_E^2 + rho * |dE/dt|))

Where:
- mu_E = moving average severity over a window
- sigma_E^2 = severity variance over a window
- dE/dt = severity rate of change (discrete derivative)
- rho = instability penalty coefficient

Also includes the Downtime Model:
- D_n(t) = 1 if service unavailable, 0 if operational
"""

import numpy as np
from collections import deque
from typing import List


class StabilityEstimator:
    """Measures long-term operational smoothness and recovery stability."""

    def __init__(self, config=None):
        from config import StabilityConfig
        self.config = config or StabilityConfig()
        self._severity_buffer: deque = deque(maxlen=self.config.moving_average_window)
        self._prev_severity: float = 0.0
        self._smoothed_stability: float = 1.0

    def estimate(self, severity: float, dt: float = 5.0) -> float:
        """
        Compute stability score S_n(t).

        S_n(t) = exp(-(mu_E + sigma_E^2 + rho * |dE/dt|))

        Returns a value in (0, 1] where 1 = perfectly stable.
        """
        self._severity_buffer.append(severity)
        buf = np.array(self._severity_buffer)

        # Moving average severity
        mu_e = np.mean(buf)

        # Severity variance
        sigma_e_sq = np.var(buf) if len(buf) > 1 else 0.0

        # Rate of change (discrete derivative)
        de_dt = abs(severity - self._prev_severity) / max(dt, 1e-10)
        self._prev_severity = severity

        # Stability formula
        exponent = -(mu_e + sigma_e_sq + self.config.rho * de_dt)
        raw_stability = np.exp(exponent)

        # Temporal smoothing (EMA)
        alpha = self.config.temporal_smoothing_alpha
        self._smoothed_stability = alpha * raw_stability + (1 - alpha) * self._smoothed_stability

        return float(np.clip(self._smoothed_stability, 0, 1))

    def estimate_series(self, severities: np.ndarray, dt: float = 5.0) -> np.ndarray:
        """Estimate stability for a series of severity values."""
        self.reset()
        return np.array([self.estimate(s, dt) for s in severities])

    def reset(self):
        self._severity_buffer.clear()
        self._prev_severity = 0.0
        self._smoothed_stability = 1.0


class DowntimeModel:
    """
    Binary operational availability model.

    D_n(t) = 1 if service unavailable, 0 if operational.
    
    A service is considered unavailable when:
    - Error rate exceeds a critical threshold, OR
    - Severity exceeds the downtime severity threshold
    """

    def __init__(self, error_rate_threshold: float = 0.5, severity_threshold: float = 0.85):
        self.error_rate_threshold = error_rate_threshold
        self.severity_threshold = severity_threshold

    def is_down(self, error_rate: float, severity: float) -> int:
        """Return 1 if service is down, 0 if operational."""
        if error_rate >= self.error_rate_threshold:
            return 1
        if severity >= self.severity_threshold:
            return 1
        return 0

    def compute_series(self, error_rates: np.ndarray, severities: np.ndarray) -> np.ndarray:
        """Compute downtime indicator for arrays."""
        return np.array([self.is_down(e, s) for e, s in zip(error_rates, severities)])
