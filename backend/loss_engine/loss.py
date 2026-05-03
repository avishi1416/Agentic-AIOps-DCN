"""
Main Operational Loss Engine.

Implements the cumulative operational degradation loss:

L = integral[ alpha*E_n(t) + beta*D_n(t) + gamma*F_n(t) + delta*(1-S_n(t)) ] dt

Approximated numerically using discrete timesteps (rectangular rule).
"""

import numpy as np
from typing import Dict, Optional


class OperationalLossEngine:
    """Computes total operational degradation over time."""

    def __init__(self, config=None):
        from config import LossConfig
        self.config = config or LossConfig()
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.gamma = self.config.gamma
        self.delta = self.config.delta
        self.dt = self.config.dt
        self._loss_history: list = []

    def compute_instantaneous(self, severity: float, downtime: int,
                               failure_prob: float, stability: float) -> float:
        """
        Compute instantaneous loss at a single timestep.

        l(t) = alpha*E_n(t) + beta*D_n(t) + gamma*F_n(t) + delta*(1-S_n(t))
        """
        loss = (
            self.alpha * np.clip(severity, 0, 1)
            + self.beta * np.clip(downtime, 0, 1)
            + self.gamma * np.clip(failure_prob, 0, 1)
            + self.delta * np.clip(1 - stability, 0, 1)
        )
        return float(np.clip(loss, 0, 1))

    def compute_cumulative(self, severities: np.ndarray, downtimes: np.ndarray,
                           failure_probs: np.ndarray, stabilities: np.ndarray) -> float:
        """
        Compute cumulative loss via numerical integration (rectangular rule).

        L = sum[ l(t_i) * dt ] for all timesteps.
        Normalized by total time to get average loss in [0, 1].
        """
        n = len(severities)
        if n == 0:
            return 0.0

        instant_losses = np.array([
            self.compute_instantaneous(severities[i], downtimes[i],
                                        failure_probs[i], stabilities[i])
            for i in range(n)
        ])

        # Numerical integration (rectangular rule)
        total_loss = np.sum(instant_losses * self.dt)

        # Normalize by total time for average loss in [0, 1]
        total_time = n * self.dt
        normalized_loss = total_loss / total_time if total_time > 0 else 0.0

        self._loss_history.append(normalized_loss)
        return float(np.clip(normalized_loss, 0, 1))

    def update_weights(self, alpha: float, beta: float, gamma: float, delta: float):
        """Update loss component weights (from optimizer)."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def get_weights(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma, "delta": self.delta}

    def get_loss_history(self) -> list:
        return self._loss_history.copy()
