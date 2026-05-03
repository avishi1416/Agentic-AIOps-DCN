"""
Adaptive Parameter Optimization (AdaGrad-style).

Implements:
    theta_(t+1,i) = theta_(t,i) - eta / sqrt(G_(t,i) + epsilon) * grad(L)

With regularization:
    L_reg = L + lambda * sum((theta_i - theta_prev_i)^2)

Constraint: alpha + beta + gamma + delta = 1 (simplex projection).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class AdaptiveOptimizer:
    """AdaGrad-based optimizer for loss component weights."""

    def __init__(self, config=None):
        from config import OptimizationConfig
        self.config = config or OptimizationConfig()

        # Initialize parameters
        self.params = np.array([0.30, 0.25, 0.25, 0.20])  # [alpha, beta, gamma, delta]
        self.param_names = ["alpha", "beta", "gamma", "delta"]
        self.prev_params = self.params.copy()

        # AdaGrad accumulated squared gradients
        self.G = np.zeros(4)

        # History
        self.param_history: List[np.ndarray] = [self.params.copy()]
        self.loss_history: List[float] = []
        self._step = 0

    def compute_gradient(self, severity: float, downtime: float,
                         failure_prob: float, stability: float,
                         loss: float) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. parameters.

        Since L = alpha*E + beta*D + gamma*F + delta*(1-S),
        the partial derivatives are simply the component values:
            dL/d(alpha) = E_n(t)
            dL/d(beta)  = D_n(t)
            dL/d(gamma) = F_n(t)
            dL/d(delta) = 1 - S_n(t)

        With regularization:
            dL_reg/d(theta_i) = dL/d(theta_i) + 2*lambda*(theta_i - theta_prev_i)
        """
        # Base gradients (partial derivatives of loss components)
        grad = np.array([
            np.clip(severity, 0, 1),
            np.clip(downtime, 0, 1),
            np.clip(failure_prob, 0, 1),
            np.clip(1 - stability, 0, 1)
        ])

        # Add regularization gradient
        reg_grad = 2 * self.config.regularization_lambda * (self.params - self.prev_params)
        grad += reg_grad

        return grad

    def step(self, severity: float, downtime: float,
             failure_prob: float, stability: float, loss: float) -> Dict[str, float]:
        """
        Perform one optimization step.

        Returns updated parameters as a dict.
        """
        self._step += 1
        self.loss_history.append(loss)

        # Compute gradient
        grad = self.compute_gradient(severity, downtime, failure_prob, stability, loss)

        # Update accumulated squared gradients (AdaGrad)
        self.G += grad ** 2

        # Compute adaptive learning rates
        adaptive_lr = self.config.learning_rate / (np.sqrt(self.G + self.config.epsilon))

        # Store previous params for regularization
        self.prev_params = self.params.copy()

        # Parameter update
        self.params = self.params - adaptive_lr * grad

        # Project onto simplex (ensure sum = 1, all params in [min, max])
        self.params = self._project_simplex(self.params)

        self.param_history.append(self.params.copy())
        return dict(zip(self.param_names, self.params))

    def _project_simplex(self, params: np.ndarray) -> np.ndarray:
        """
        Project parameters onto the probability simplex with bounds.

        Ensures: sum(params) = 1, min_weight <= param_i <= max_weight.
        Uses iterative clipping and renormalization.
        """
        p = params.copy()

        # Clip to bounds
        p = np.clip(p, self.config.min_weight, self.config.max_weight)

        # Ensure non-negative
        p = np.maximum(p, 0)

        # Normalize to sum to 1
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum
        else:
            p = np.ones(4) / 4

        # Re-clip after normalization (may slightly violate bounds)
        p = np.clip(p, self.config.min_weight, self.config.max_weight)

        # Final renormalization
        p = p / np.sum(p)

        return p

    def get_params(self) -> Dict[str, float]:
        return dict(zip(self.param_names, self.params))

    def get_history(self) -> List[Dict[str, float]]:
        return [dict(zip(self.param_names, p)) for p in self.param_history]

    def reset(self):
        self.params = np.array([0.30, 0.25, 0.25, 0.20])
        self.prev_params = self.params.copy()
        self.G = np.zeros(4)
        self.param_history = [self.params.copy()]
        self.loss_history = []
        self._step = 0
