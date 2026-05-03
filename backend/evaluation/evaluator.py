"""
Evaluation Engine.

Implements five key evaluation metrics:
1. Loss Reduction Rate (LRR) = (L_before - L_after) / L_before
2. Remediation Success Rate (RSR) = successful / total
3. False Remediation Rate (FRR) = incorrect / total
4. Stability Improvement Score (SIS) = S_after - S_before
5. Mean Time To Recovery (MTTR)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class RemediationOutcome:
    """Record of a single remediation for evaluation."""
    action: str
    loss_before: float
    loss_after: float
    stability_before: float
    stability_after: float
    recovery_time: float  # seconds
    was_anomaly: bool     # was there a real anomaly?
    success: bool         # did remediation resolve the issue?


class EvaluationEngine:
    """Evaluates operational effectiveness and remediation quality."""

    def __init__(self):
        self.outcomes: List[RemediationOutcome] = []

    def record_outcome(self, outcome: RemediationOutcome):
        """Record a remediation outcome for evaluation."""
        self.outcomes.append(outcome)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        if not self.outcomes:
            return {"LRR": 0.0, "RSR": 0.0, "FRR": 0.0, "SIS": 0.0, "MTTR": 0.0}

        n = len(self.outcomes)

        # 1. Loss Reduction Rate
        lrr_values = []
        for o in self.outcomes:
            if o.loss_before > 0:
                lrr_values.append((o.loss_before - o.loss_after) / o.loss_before)
            else:
                lrr_values.append(0.0)
        lrr = float(np.mean(lrr_values))

        # 2. Remediation Success Rate
        successful = sum(1 for o in self.outcomes if o.success)
        rsr = successful / n

        # 3. False Remediation Rate (remediated when no real anomaly)
        false_rems = sum(1 for o in self.outcomes if not o.was_anomaly)
        frr = false_rems / n

        # 4. Stability Improvement Score
        sis_values = [o.stability_after - o.stability_before for o in self.outcomes]
        sis = float(np.mean(sis_values))

        # 5. Mean Time To Recovery
        recovery_times = [o.recovery_time for o in self.outcomes if o.success]
        mttr = float(np.mean(recovery_times)) if recovery_times else 0.0

        return {
            "LRR": round(lrr, 4),
            "RSR": round(rsr, 4),
            "FRR": round(frr, 4),
            "SIS": round(sis, 4),
            "MTTR": round(mttr, 2),
            "total_remediations": n,
            "successful_remediations": successful,
        }

    def reset(self):
        self.outcomes = []
