"""
Configuration System for Adaptive Loss-Based Autonomous Remediation.

Centralized configuration management with sensible defaults.
All parameters are mathematically motivated and documented.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TelemetryConfig:
    """Configuration for telemetry ingestion engine."""
    # Synthetic generation parameters
    num_services: int = 5
    duration_seconds: int = 3600          # 1 hour of telemetry
    sampling_interval_seconds: float = 5.0  # 5-second sampling
    
    # Metric bounds (realistic operational ranges)
    cpu_max: float = 100.0                # percentage
    memory_max: float = 100.0             # percentage
    latency_max: float = 5000.0           # milliseconds
    error_rate_max: float = 1.0           # fraction [0, 1]
    
    # Baseline operational values (healthy state)
    cpu_baseline: float = 25.0
    memory_baseline: float = 40.0
    latency_baseline: float = 50.0        # ms
    error_rate_baseline: float = 0.01
    
    # Noise parameters (Gaussian noise std dev)
    cpu_noise_std: float = 5.0
    memory_noise_std: float = 3.0
    latency_noise_std: float = 10.0
    error_rate_noise_std: float = 0.005
    
    # Incident simulation
    incident_probability: float = 0.15    # probability of incident per window
    incident_types: List[str] = field(default_factory=lambda: [
        "cpu_overload", "memory_leak", "latency_spike",
        "cascading_degradation", "service_instability"
    ])


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection engine."""
    # Threshold-based detection
    cpu_warning_threshold: float = 70.0
    cpu_critical_threshold: float = 90.0
    memory_warning_threshold: float = 75.0
    memory_critical_threshold: float = 92.0
    latency_warning_threshold: float = 200.0   # ms
    latency_critical_threshold: float = 1000.0  # ms
    error_rate_warning_threshold: float = 0.05
    error_rate_critical_threshold: float = 0.15
    
    # Rolling statistical deviation
    rolling_window_size: int = 20         # number of samples
    z_score_warning: float = 2.0          # standard deviations
    z_score_critical: float = 3.0


@dataclass
class SeverityConfig:
    """Configuration for normalized severity function E_n(t)."""
    # Telemetry weights (must sum to 1.0)
    w_cpu: float = 0.25
    w_latency: float = 0.35               # latency contributes more aggressively
    w_memory: float = 0.25
    w_error: float = 0.15
    
    # Normalization bounds
    cpu_max: float = 100.0
    latency_threshold: float = 1000.0      # L_thr in the formula
    memory_max: float = 100.0
    
    # Adaptive weighting
    enable_adaptive_weights: bool = True
    weight_adaptation_rate: float = 0.05   # how fast weights adapt


@dataclass
class FailurePredictionConfig:
    """Configuration for sigmoid-based failure prediction F_n(t)."""
    k: float = 10.0                        # sigmoid sensitivity (steepness)
    theta: float = 0.5                     # severity threshold for 50% failure probability
    sliding_window_size: int = 10          # temporal sliding window
    cumulative_decay: float = 0.95         # exponential decay for cumulative effects


@dataclass
class StabilityConfig:
    """Configuration for stability estimation S_n(t)."""
    moving_average_window: int = 20        # window for mu_E computation
    variance_window: int = 20             # window for sigma_E^2 computation
    rho: float = 1.5                       # derivative-based instability penalty weight
    temporal_smoothing_alpha: float = 0.1  # exponential smoothing parameter


@dataclass
class LossConfig:
    """Configuration for operational loss L."""
    # Loss component weights (must sum to 1.0)
    alpha: float = 0.30                    # severity weight
    beta: float = 0.25                     # downtime weight
    gamma: float = 0.25                    # failure prediction weight
    delta: float = 0.20                    # stability weight
    
    # Integration parameters
    dt: float = 5.0                        # discrete timestep (seconds)


@dataclass
class OptimizationConfig:
    """Configuration for adaptive parameter optimization."""
    learning_rate: float = 0.01            # eta
    epsilon: float = 1e-8                  # numerical stability
    regularization_lambda: float = 0.01    # change penalty
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    
    # Parameter bounds
    min_weight: float = 0.05               # minimum weight per component
    max_weight: float = 0.60               # maximum weight per component


@dataclass
class RemediationConfig:
    """Configuration for remediation engine."""
    # Available remediation actions
    actions: List[str] = field(default_factory=lambda: [
        "restart_service", "scale_replicas", "rollback_deployment"
    ])
    
    # Remediation effectiveness priors (initial estimates)
    restart_effectiveness: float = 0.6
    scale_effectiveness: float = 0.4
    rollback_effectiveness: float = 0.8
    
    # Scoring parameters
    history_weight: float = 0.4            # weight for historical effectiveness
    prediction_weight: float = 0.6         # weight for predicted loss reduction
    
    # Simulation parameters
    restart_recovery_time: float = 30.0    # seconds
    scale_recovery_time: float = 60.0
    rollback_recovery_time: float = 120.0


@dataclass
class MemoryConfig:
    """Configuration for operational memory system."""
    max_history_size: int = 10000          # maximum stored events
    similarity_threshold: float = 0.7      # cosine similarity threshold
    retrieval_top_k: int = 5               # top-k similar incidents to retrieve


@dataclass
class EvaluationConfig:
    """Configuration for evaluation engine."""
    mttr_window: int = 50                  # timesteps to measure recovery


@dataclass
class SystemConfig:
    """Master configuration aggregating all sub-configurations."""
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    failure_prediction: FailurePredictionConfig = field(default_factory=FailurePredictionConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    remediation: RemediationConfig = field(default_factory=RemediationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Global settings
    random_seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"


def get_default_config() -> SystemConfig:
    """Return a SystemConfig instance with all default values."""
    return SystemConfig()
