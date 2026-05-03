"""
Operational Memory System.

Stores and retrieves:
- Telemetry snapshots
- Anomaly events
- Remediation actions and outcomes
- Loss values before/after remediation

Supports similarity-based retrieval using cosine similarity
on telemetry feature vectors for experience-based learning.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryRecord:
    """A single operational memory record."""
    record_id: int
    timestamp: float
    service_id: str
    telemetry_vector: np.ndarray  # [cpu, memory, latency, error_rate]
    anomaly_state: str
    severity: float
    failure_prob: float
    stability: float
    loss_before: float
    remediation_action: Optional[str] = None
    loss_after: Optional[float] = None
    recovery_time: Optional[float] = None
    success: Optional[bool] = None


class OperationalMemory:
    """
    Experience-based operational memory for remediation optimization.

    Enables the system to learn from past incidents by:
    1. Storing telemetry-action-outcome tuples
    2. Retrieving similar historical incidents via cosine similarity
    3. Comparing remediation effectiveness across similar situations
    """

    def __init__(self, config=None):
        from config import MemoryConfig
        self.config = config or MemoryConfig()
        self.records: List[MemoryRecord] = []
        self._counter = 0

    def store(self, timestamp: float, service_id: str,
              telemetry: Dict[str, float], anomaly_state: str,
              severity: float, failure_prob: float, stability: float,
              loss_before: float, remediation_action: Optional[str] = None,
              loss_after: Optional[float] = None, recovery_time: Optional[float] = None,
              success: Optional[bool] = None) -> int:
        """Store a new operational memory record. Returns record_id."""
        self._counter += 1
        vec = np.array([
            telemetry.get("cpu_usage", 0),
            telemetry.get("memory_usage", 0),
            telemetry.get("latency", 0),
            telemetry.get("error_rate", 0)
        ])
        record = MemoryRecord(
            record_id=self._counter, timestamp=timestamp, service_id=service_id,
            telemetry_vector=vec, anomaly_state=anomaly_state,
            severity=severity, failure_prob=failure_prob, stability=stability,
            loss_before=loss_before, remediation_action=remediation_action,
            loss_after=loss_after, recovery_time=recovery_time, success=success)
        self.records.append(record)

        # Evict oldest if over capacity
        if len(self.records) > self.config.max_history_size:
            self.records = self.records[-self.config.max_history_size:]

        return self._counter

    def retrieve_similar(self, telemetry: Dict[str, float],
                         top_k: Optional[int] = None) -> List[MemoryRecord]:
        """
        Retrieve top-k most similar historical incidents.

        Similarity is computed via cosine similarity on normalized
        telemetry feature vectors.
        """
        if not self.records:
            return []

        k = top_k or self.config.retrieval_top_k
        query = np.array([
            telemetry.get("cpu_usage", 0),
            telemetry.get("memory_usage", 0),
            telemetry.get("latency", 0),
            telemetry.get("error_rate", 0)
        ]).reshape(1, -1)

        # Normalize
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        # Build matrix of stored vectors
        vectors = np.array([r.telemetry_vector for r in self.records])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vectors_norm = vectors / norms

        # Cosine similarity
        sims = cosine_similarity(query, vectors_norm).flatten()

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:k]

        return [self.records[i] for i in top_indices if sims[i] >= self.config.similarity_threshold]

    def get_remediation_effectiveness(self, action: str) -> Dict[str, float]:
        """Compute historical effectiveness metrics for a remediation action."""
        relevant = [r for r in self.records if r.remediation_action == action and r.loss_after is not None]
        if not relevant:
            return {"avg_loss_reduction": 0.0, "success_rate": 0.0, "avg_recovery_time": 0.0, "count": 0}

        reductions = []
        successes = 0
        recovery_times = []
        for r in relevant:
            if r.loss_before > 0:
                reductions.append((r.loss_before - r.loss_after) / r.loss_before)
            if r.success:
                successes += 1
            if r.recovery_time is not None:
                recovery_times.append(r.recovery_time)

        return {
            "avg_loss_reduction": float(np.mean(reductions)) if reductions else 0.0,
            "success_rate": successes / len(relevant),
            "avg_recovery_time": float(np.mean(recovery_times)) if recovery_times else 0.0,
            "count": len(relevant)
        }

    def get_history_summary(self) -> Dict[str, Any]:
        """Summary of operational memory contents."""
        return {
            "total_records": len(self.records),
            "unique_services": len(set(r.service_id for r in self.records)),
            "remediations_stored": sum(1 for r in self.records if r.remediation_action),
            "successful_remediations": sum(1 for r in self.records if r.success),
        }
