"""
AIOps Remediation Pipeline — Main Orchestrator.

Orchestrates the full simulation loop:
Telemetry → Anomaly Detection → Severity → Failure Prediction →
Stability → Loss → Optimization → Remediation → Evaluation → Memory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from config import SystemConfig, get_default_config
from backend.telemetry import TelemetryEngine
from backend.anomaly_detection import AnomalyDetector
from backend.severity import SeverityComputer
from backend.failure_prediction import FailurePredictor
from backend.stability import StabilityEstimator
from backend.stability.estimator import DowntimeModel
from backend.loss_engine import OperationalLossEngine
from backend.optimization import AdaptiveOptimizer
from backend.remediation import RemediationEngine
from backend.memory import OperationalMemory
from backend.evaluation import EvaluationEngine
from backend.evaluation.evaluator import RemediationOutcome


class AIOpsRemediationPipeline:
    """
    End-to-end pipeline for adaptive loss-based autonomous remediation.

    Implements the full research prototype flow as described in the paper.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_default_config()
        self.is_initialized = False

        # Initialize all engines
        self.telemetry_engine = TelemetryEngine(self.config.telemetry)
        self.anomaly_detector = AnomalyDetector(self.config.anomaly)
        self.severity_computer = SeverityComputer(self.config.severity)
        self.failure_predictor = FailurePredictor(self.config.failure_prediction)
        self.stability_estimator = StabilityEstimator(self.config.stability)
        self.downtime_model = DowntimeModel()
        self.loss_engine = OperationalLossEngine(self.config.loss)
        self.optimizer = AdaptiveOptimizer(self.config.optimization)
        self.memory = OperationalMemory(self.config.memory)
        self.evaluation = EvaluationEngine()

        self.remediation_engine = RemediationEngine(
            config=self.config.remediation,
            memory=self.memory,
            loss_engine=self.loss_engine,
            severity_computer=self.severity_computer,
            failure_predictor=self.failure_predictor,
            stability_estimator=self.stability_estimator)

        # Pipeline state
        self.telemetry_df: Optional[pd.DataFrame] = None
        self.annotated_df: Optional[pd.DataFrame] = None
        self.incidents: List = []
        self.service_states: Dict[str, Dict[str, Any]] = {}
        self.pipeline_results: Dict[str, Any] = {}

    def run(self, num_services: int = 5, duration_seconds: int = 3600,
            with_incidents: bool = True, run_optimization: bool = True) -> Dict[str, Any]:
        """
        Run the complete simulation pipeline.

        Returns a summary of results.
        """
        # Update config
        self.config.telemetry.num_services = num_services
        self.config.telemetry.duration_seconds = duration_seconds
        self.telemetry_engine = TelemetryEngine(self.config.telemetry)

        # Reset engines
        self.failure_predictor.reset()
        self.stability_estimator.reset()
        self.optimizer.reset()
        self.evaluation.reset()
        self.anomaly_detector.events = []

        # Step 1: Generate telemetry
        self.telemetry_df, self.incidents = self.telemetry_engine.generate_synthetic(with_incidents)

        # Step 2: Anomaly detection
        self.annotated_df = self.anomaly_detector.detect(self.telemetry_df)

        # Step 3-7: Process each service
        services = self.annotated_df["service_id"].unique()
        self.service_states = {}

        for service_id in services:
            sdf = self.annotated_df[self.annotated_df["service_id"] == service_id].copy()
            sdf = sdf.sort_values("timestamp").reset_index(drop=True)

            state = self._process_service(sdf, service_id, run_optimization)
            self.service_states[service_id] = state

        # Aggregate results
        self.pipeline_results = self._aggregate_results()
        self.is_initialized = True

        return self.pipeline_results

    def _process_service(self, sdf: pd.DataFrame, service_id: str,
                         run_optimization: bool) -> Dict[str, Any]:
        """Process a single service through the full pipeline."""
        n = len(sdf)
        severities = np.zeros(n)
        failure_probs = np.zeros(n)
        stabilities = np.zeros(n)
        downtimes = np.zeros(n)
        instant_losses = np.zeros(n)

        # Reset per-service state
        self.failure_predictor.reset()
        self.stability_estimator.reset()

        # Step 3: Severity computation
        severities = self.severity_computer.compute_series(sdf).values

        # Step 4: Failure prediction
        failure_probs = self.failure_predictor.predict_series(severities)

        # Step 5: Stability estimation
        stabilities = self.stability_estimator.estimate_series(
            severities, dt=self.config.telemetry.sampling_interval_seconds)

        # Step 6: Downtime model
        downtimes = self.downtime_model.compute_series(
            sdf["error_rate"].values, severities)

        # Step 7: Instantaneous losses
        for i in range(n):
            instant_losses[i] = self.loss_engine.compute_instantaneous(
                severities[i], downtimes[i], failure_probs[i], stabilities[i])

        # Cumulative loss
        cumulative_loss = self.loss_engine.compute_cumulative(
            severities, downtimes, failure_probs, stabilities)

        # Step 8: Adaptive optimization (on time-averaged values)
        if run_optimization:
            avg_sev = float(np.mean(severities))
            avg_down = float(np.mean(downtimes))
            avg_fail = float(np.mean(failure_probs))
            avg_stab = float(np.mean(stabilities))
            new_params = self.optimizer.step(avg_sev, avg_down, avg_fail, avg_stab, cumulative_loss)
            self.loss_engine.update_weights(**new_params)

        # Step 9: Remediation (for anomalous timesteps)
        remediation_results = []
        anomaly_mask = sdf["anomaly_state"].isin(["warning", "critical"])
        anomaly_indices = sdf.index[anomaly_mask].tolist()

        # Sample remediation events (not every anomaly triggers remediation)
        rng = np.random.default_rng(42)
        if anomaly_indices:
            # Remediate at most 10 incidents per service
            sample_size = min(10, len(anomaly_indices))
            sample_indices = sorted(rng.choice(anomaly_indices, size=sample_size, replace=False))

            for idx in sample_indices:
                row = sdf.loc[idx]
                telemetry = {
                    "cpu_usage": row["cpu_usage"],
                    "memory_usage": row["memory_usage"],
                    "latency": row["latency"],
                    "error_rate": row["error_rate"]}

                current_severity = severities[idx] if idx < n else 0
                current_loss = instant_losses[idx] if idx < n else 0
                stability_before = stabilities[idx] if idx < n else 1.0

                # Select best remediation
                best, confidence = self.remediation_engine.select_best(
                    telemetry, current_severity, current_loss)

                # Simulate remediation
                result = self.remediation_engine.simulate_remediation(
                    best.action, telemetry, current_loss)

                # Compute post-remediation stability
                post_severity = self.severity_computer.compute(
                    result.telemetry_after.get("cpu_usage", 25),
                    result.telemetry_after.get("latency", 50),
                    result.telemetry_after.get("memory_usage", 40),
                    result.telemetry_after.get("error_rate", 0.01))
                stability_after = min(1.0, stability_before + 0.2 if result.success else stability_before - 0.1)

                # Store in memory
                self.memory.store(
                    timestamp=row["timestamp"], service_id=service_id,
                    telemetry=telemetry, anomaly_state=row["anomaly_state"],
                    severity=current_severity, failure_prob=failure_probs[idx] if idx < n else 0,
                    stability=stability_before, loss_before=current_loss,
                    remediation_action=best.action, loss_after=result.loss_after,
                    recovery_time=result.recovery_time, success=result.success)

                # Record evaluation outcome
                self.evaluation.record_outcome(RemediationOutcome(
                    action=best.action,
                    loss_before=current_loss, loss_after=result.loss_after,
                    stability_before=stability_before, stability_after=stability_after,
                    recovery_time=result.recovery_time,
                    was_anomaly=True, success=result.success))

                remediation_results.append({
                    "timestamp": float(row["timestamp"]),
                    "action": best.action,
                    "confidence": confidence,
                    "success": result.success,
                    "loss_before": float(current_loss),
                    "loss_after": float(result.loss_after),
                    "recovery_time": result.recovery_time})

        # Adapt severity weights based on anomaly frequency
        anomaly_events = self.anomaly_detector.get_events(service_id)
        if anomaly_events:
            metric_counts = {}
            for event in anomaly_events:
                for m in event.triggered_metrics:
                    metric_counts[m] = metric_counts.get(m, 0) + 1
            self.severity_computer.adapt_weights(metric_counts)

        return {
            "service_id": service_id,
            "num_points": n,
            "severities": severities.tolist(),
            "failure_probs": failure_probs.tolist(),
            "stabilities": stabilities.tolist(),
            "downtimes": downtimes.tolist(),
            "instant_losses": instant_losses.tolist(),
            "cumulative_loss": cumulative_loss,
            "num_anomalies": int(anomaly_mask.sum()),
            "num_remediations": len(remediation_results),
            "remediations": remediation_results,
            "severity_weights": self.severity_computer.get_weights(),
            "loss_weights": self.loss_engine.get_weights(),
        }

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all services."""
        eval_metrics = self.evaluation.compute_metrics()
        memory_summary = self.memory.get_history_summary()

        all_losses = [s["cumulative_loss"] for s in self.service_states.values()]
        all_anomalies = sum(s["num_anomalies"] for s in self.service_states.values())
        all_remediations = sum(s["num_remediations"] for s in self.service_states.values())

        return {
            "num_services": len(self.service_states),
            "total_anomalies_detected": all_anomalies,
            "total_remediations": all_remediations,
            "avg_cumulative_loss": float(np.mean(all_losses)) if all_losses else 0.0,
            "max_cumulative_loss": float(np.max(all_losses)) if all_losses else 0.0,
            "evaluation_metrics": eval_metrics,
            "memory_summary": memory_summary,
            "optimizer_params": self.optimizer.get_params(),
            "num_incidents_injected": len(self.incidents),
        }

    # ─── API Helpers ──────────────────────────────────────────

    def get_telemetry(self, service_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get telemetry data as list of dicts."""
        if self.annotated_df is None:
            return []
        df = self.annotated_df
        if service_id:
            df = df[df["service_id"] == service_id]
        return df.tail(limit).to_dict(orient="records")

    def get_anomalies(self, service_id: Optional[str] = None) -> List[Dict]:
        events = self.anomaly_detector.get_events(service_id)
        return [{"timestamp": e.timestamp, "service_id": e.service_id,
                 "state": e.state.value, "triggered_metrics": e.triggered_metrics,
                 "details": e.details} for e in events[:200]]  # cap at 200

    def get_metrics(self, service_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.service_states:
            return {}
        if service_id and service_id in self.service_states:
            s = self.service_states[service_id]
            return {
                "service_id": service_id,
                "avg_severity": float(np.mean(s["severities"])),
                "max_severity": float(np.max(s["severities"])),
                "avg_failure_prob": float(np.mean(s["failure_probs"])),
                "avg_stability": float(np.mean(s["stabilities"])),
                "cumulative_loss": s["cumulative_loss"],
                "downtime_fraction": float(np.mean(s["downtimes"])),
                "num_anomalies": s["num_anomalies"],
                "severity_weights": s["severity_weights"],
                "loss_weights": s["loss_weights"],
            }
        # All services summary
        return {svc: {
            "avg_severity": float(np.mean(st["severities"])),
            "cumulative_loss": st["cumulative_loss"],
            "num_anomalies": st["num_anomalies"],
        } for svc, st in self.service_states.items()}

    def get_remediation_recommendation(self, service_id: str) -> Dict[str, Any]:
        if service_id not in self.service_states:
            return {"error": f"Service {service_id} not found"}
        state = self.service_states[service_id]
        # Use the last anomalous state's telemetry for recommendation
        if state["remediations"]:
            last = state["remediations"][-1]
            return {"service_id": service_id, "recommendation": last}
        return {"service_id": service_id, "recommendation": None,
                "message": "No anomalies detected requiring remediation"}

    def execute_remediation(self, service_id: str, action: Optional[str] = None) -> Dict:
        if self.annotated_df is None:
            return {"error": "Pipeline not initialized"}
        sdf = self.annotated_df[self.annotated_df["service_id"] == service_id]
        if sdf.empty:
            return {"error": f"Service {service_id} not found"}
        last = sdf.iloc[-1]
        telemetry = {"cpu_usage": last["cpu_usage"], "memory_usage": last["memory_usage"],
                     "latency": last["latency"], "error_rate": last["error_rate"]}
        sev = self.severity_computer.compute(last["cpu_usage"], last["latency"],
                                              last["memory_usage"], last["error_rate"])
        loss = self.loss_engine.compute_instantaneous(sev, 0, 0.5, 0.5)
        if action is None:
            best, conf = self.remediation_engine.select_best(telemetry, sev, loss)
            action = best.action
        result = self.remediation_engine.simulate_remediation(action, telemetry, loss)
        return {"action": result.action, "success": result.success,
                "loss_before": result.loss_before, "loss_after": result.loss_after,
                "recovery_time": result.recovery_time, "telemetry_after": result.telemetry_after}

    def get_evaluation_metrics(self) -> Dict:
        return self.evaluation.compute_metrics()

    def get_memory_summary(self) -> Dict:
        return self.memory.get_history_summary()

    def get_full_state(self) -> Dict[str, Any]:
        """Full pipeline state for frontend."""
        return {
            "pipeline_results": self.pipeline_results,
            "service_states": {
                svc: {
                    "avg_severity": float(np.mean(st["severities"])),
                    "max_severity": float(np.max(st["severities"])),
                    "avg_failure_prob": float(np.mean(st["failure_probs"])),
                    "avg_stability": float(np.mean(st["stabilities"])),
                    "cumulative_loss": st["cumulative_loss"],
                    "downtime_fraction": float(np.mean(st["downtimes"])),
                    "num_anomalies": st["num_anomalies"],
                    "num_remediations": st["num_remediations"],
                    "remediations": st["remediations"],
                    "severity_weights": st["severity_weights"],
                    "loss_weights": st["loss_weights"],
                } for svc, st in self.service_states.items()
            },
            "evaluation": self.evaluation.compute_metrics(),
            "memory": self.memory.get_history_summary(),
            "optimizer": self.optimizer.get_params(),
            "incidents": [{"id": inc.incident_id, "type": inc.incident_type,
                          "service": inc.service_id, "severity": inc.severity,
                          "start": inc.start_time, "end": inc.end_time}
                         for inc in self.incidents],
        }
