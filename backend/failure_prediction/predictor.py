"""
Sigmoid-Based Failure Prediction Engine.

Implements: F_n(t) = 1 / (1 + exp(-k * (E_n(t) - theta)))

Features:
- Probability outputs between 0 and 1
- Configurable sigmoid sensitivity (k) and threshold (theta)
- Temporal sliding window support for smoothing
- Cumulative degradation effects via exponential decay accumulator
"""

import numpy as np
from typing import List, Optional
from collections import deque


class FailurePredictor:
    """Estimates future operational collapse probability."""

    def __init__(self, config=None):
        from config import FailurePredictionConfig
        self.config = config or FailurePredictionConfig()
        self._severity_window: deque = deque(maxlen=self.config.sliding_window_size)
        self._cumulative_severity: float = 0.0

    def predict(self, severity: float) -> float:
        """
        Predict failure probability from current severity.

        F_n(t) = 1 / (1 + exp(-k * (E_effective - theta)))

        Where E_effective combines instantaneous and cumulative severity.
        """
        # Update cumulative severity with exponential decay
        self._cumulative_severity = (
            self.config.cumulative_decay * self._cumulative_severity + severity
        )
        self._severity_window.append(severity)

        # Effective severity = weighted combination of current and cumulative
        window_avg = np.mean(list(self._severity_window))
        e_effective = 0.6 * severity + 0.4 * window_avg

        # Sigmoid failure probability
        exponent = -self.config.k * (e_effective - self.config.theta)
        # Clamp exponent to prevent overflow
        exponent = np.clip(exponent, -500, 500)
        f_prob = 1.0 / (1.0 + np.exp(exponent))

        return float(np.clip(f_prob, 0, 1))

    def predict_series(self, severities: np.ndarray) -> np.ndarray:
        """Predict failure probability for a series of severity values."""
        self.reset()
        return np.array([self.predict(s) for s in severities])

    def reset(self):
        """Reset internal state."""
        self._severity_window.clear()
        self._cumulative_severity = 0.0

    def get_cumulative_severity(self) -> float:
        return self._cumulative_severity
