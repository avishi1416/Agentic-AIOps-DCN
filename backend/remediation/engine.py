"""
Remediation Engine and Selection Logic.

Implements:
- Candidate remediation actions (restart, scale, rollback)
- Remediation scoring based on telemetry and predicted loss
- History-based effectiveness lookup
- Optimal action selection: a* = argmin E[L | X, a_i]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RemediationCandidate:
    """A candidate remediation action with scoring."""
    action: str
    predicted_loss_after: float
    expected_loss_reduction: float
    historical_success_rate: float
    recovery_time: float
    composite_score: float  # lower = better


@dataclass
class RemediationResult:
    """Outcome of a remediation action."""
    action: str
    success: bool
    loss_before: float
    loss_after: float
    recovery_time: float
    telemetry_after: Dict[str, float]


class RemediationEngine:
    """
    Selects and simulates remediation actions.

    Selection logic: a* = argmin E[L | X, a_i]
    - Estimates expected future loss for each candidate action
    - Uses operational memory for historical effectiveness
    - Selects minimum-loss action
    """

    # Remediation effect profiles (how each action modifies telemetry)
    ACTION_PROFILES = {
        "restart_service": {
            "cpu_reduction": 0.7,      # reduces CPU by 70% of excess
            "memory_reduction": 0.8,   # reduces memory by 80% of excess
            "latency_reduction": 0.5,
            "error_rate_reduction": 0.6,
            "recovery_time": 30.0,
            "success_probability": 0.75,
        },
        "scale_replicas": {
            "cpu_reduction": 0.5,
            "memory_reduction": 0.3,
            "latency_reduction": 0.7,
            "error_rate_reduction": 0.4,
            "recovery_time": 60.0,
            "success_probability": 0.65,
        },
        "rollback_deployment": {
            "cpu_reduction": 0.6,
            "memory_reduction": 0.5,
            "latency_reduction": 0.8,
            "error_rate_reduction": 0.9,
            "recovery_time": 120.0,
            "success_probability": 0.85,
        },
    }

    def __init__(self, config=None, memory=None, loss_engine=None,
                 severity_computer=None, failure_predictor=None,
                 stability_estimator=None):
        from config import RemediationConfig
        self.config = config or RemediationConfig()
        self.memory = memory
        self.loss_engine = loss_engine
        self.severity_computer = severity_computer
        self.failure_predictor = failure_predictor
        self.stability_estimator = stability_estimator
        self.rng = np.random.default_rng(42)

    def evaluate_candidates(self, telemetry: Dict[str, float],
                            current_severity: float, current_loss: float
                            ) -> List[RemediationCandidate]:
        """
        Evaluate all candidate remediation actions.

        For each action, estimate:
        1. Post-remediation telemetry
        2. Predicted severity and loss after action
        3. Historical effectiveness from memory
        4. Composite score for ranking
        """
        candidates = []
        baselines = {"cpu_usage": 25.0, "memory_usage": 40.0,
                     "latency": 50.0, "error_rate": 0.01}

        for action in self.config.actions:
            profile = self.ACTION_PROFILES.get(action, {})

            # Estimate post-remediation telemetry
            post_telemetry = {}
            for metric in ["cpu_usage", "memory_usage", "latency", "error_rate"]:
                current = telemetry.get(metric, baselines[metric])
                baseline = baselines[metric]
                excess = max(0, current - baseline)
                reduction_key = metric.replace("_usage", "").replace("_rate", "_rate") + "_reduction"
                if metric == "cpu_usage":
                    reduction_key = "cpu_reduction"
                elif metric == "memory_usage":
                    reduction_key = "memory_reduction"
                elif metric == "latency":
                    reduction_key = "latency_reduction"
                elif metric == "error_rate":
                    reduction_key = "error_rate_reduction"
                reduction = profile.get(reduction_key, 0.5)
                post_telemetry[metric] = baseline + excess * (1 - reduction)

            # Predict post-remediation severity
            if self.severity_computer:
                pred_severity = self.severity_computer.compute(
                    post_telemetry["cpu_usage"], post_telemetry["latency"],
                    post_telemetry["memory_usage"], post_telemetry["error_rate"])
            else:
                pred_severity = current_severity * (1 - np.mean([
                    profile.get("cpu_reduction", 0.5),
                    profile.get("latency_reduction", 0.5)]))

            # Predicted loss after remediation
            pred_loss = current_loss * (pred_severity / max(current_severity, 1e-10))
            pred_loss = np.clip(pred_loss, 0, 1)

            expected_reduction = max(0, current_loss - pred_loss)

            # Historical effectiveness from memory
            hist_success = profile.get("success_probability", 0.5)
            if self.memory:
                hist = self.memory.get_remediation_effectiveness(action)
                if hist["count"] > 0:
                    hist_success = hist["success_rate"]

            # Composite score: lower = better
            # Weighted combination of predicted loss and historical success
            score = (self.config.prediction_weight * pred_loss
                     + self.config.history_weight * (1 - hist_success))

            candidates.append(RemediationCandidate(
                action=action, predicted_loss_after=pred_loss,
                expected_loss_reduction=expected_reduction,
                historical_success_rate=hist_success,
                recovery_time=profile.get("recovery_time", 60.0),
                composite_score=score))

        # Sort by composite score (ascending = better)
        candidates.sort(key=lambda c: c.composite_score)
        return candidates

    def select_best(self, telemetry: Dict[str, float],
                    current_severity: float, current_loss: float
                    ) -> Tuple[RemediationCandidate, float]:
        """
        Select optimal remediation: a* = argmin E[L | X, a_i]

        Returns (best_candidate, confidence).
        Confidence is 1 - (best_score / worst_score), indicating
        how much better the best action is compared to the worst.
        """
        candidates = self.evaluate_candidates(telemetry, current_severity, current_loss)
        if not candidates:
            raise ValueError("No remediation candidates available")

        best = candidates[0]
        worst = candidates[-1]
        if worst.composite_score > 0:
            confidence = 1.0 - (best.composite_score / worst.composite_score)
        else:
            confidence = 0.5

        return best, float(np.clip(confidence, 0, 1))

    def simulate_remediation(self, action: str, telemetry: Dict[str, float],
                              current_loss: float) -> RemediationResult:
        """
        Simulate executing a remediation action.

        Uses action profiles with stochastic success/failure modeling.
        """
        profile = self.ACTION_PROFILES.get(action, {})
        baselines = {"cpu_usage": 25.0, "memory_usage": 40.0,
                     "latency": 50.0, "error_rate": 0.01}

        success = self.rng.random() < profile.get("success_probability", 0.5)

        if success:
            post = {}
            for metric in ["cpu_usage", "memory_usage", "latency", "error_rate"]:
                current = telemetry.get(metric, baselines[metric])
                baseline = baselines[metric]
                excess = max(0, current - baseline)
                rkey = {"cpu_usage": "cpu_reduction", "memory_usage": "memory_reduction",
                        "latency": "latency_reduction", "error_rate": "error_rate_reduction"}[metric]
                reduction = profile.get(rkey, 0.5)
                # Add some noise to make it realistic
                noise = self.rng.normal(0, 0.05)
                post[metric] = max(baseline, baseline + excess * (1 - reduction + noise))
            loss_after = current_loss * 0.3  # significant reduction on success
        else:
            # Failed remediation — metrics may slightly worsen
            post = {}
            for metric in ["cpu_usage", "memory_usage", "latency", "error_rate"]:
                current = telemetry.get(metric, baselines[metric])
                post[metric] = current * (1 + self.rng.uniform(0, 0.1))
            loss_after = min(1.0, current_loss * 1.1)

        return RemediationResult(
            action=action, success=success,
            loss_before=current_loss, loss_after=loss_after,
            recovery_time=profile.get("recovery_time", 60.0),
            telemetry_after=post)
