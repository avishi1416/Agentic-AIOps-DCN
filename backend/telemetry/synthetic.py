"""
Synthetic Telemetry Generator.

Generates realistic temporal telemetry behavior using:
- Ornstein-Uhlenbeck processes for mean-reverting metrics
- Diurnal patterns for CPU/memory usage
- Correlated noise across metrics
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class TelemetryPoint:
    """Single telemetry observation."""
    timestamp: float
    service_id: str
    cpu_usage: float
    memory_usage: float
    latency: float
    error_rate: float


class SyntheticTelemetryGenerator:
    """
    Generates synthetic telemetry streams with realistic temporal behavior.
    
    Uses Ornstein-Uhlenbeck (OU) processes for mean-reverting dynamics:
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    
    This ensures metrics naturally fluctuate around baselines rather than
    producing unrealistic random walks.
    """
    
    def __init__(
        self,
        num_services: int = 5,
        duration_seconds: int = 3600,
        sampling_interval: float = 5.0,
        baselines: Optional[Dict[str, float]] = None,
        noise_stds: Optional[Dict[str, float]] = None,
        seed: int = 42
    ):
        self.num_services = num_services
        self.duration_seconds = duration_seconds
        self.sampling_interval = sampling_interval
        self.rng = np.random.default_rng(seed)
        
        # Baseline (healthy) values
        self.baselines = baselines or {
            "cpu_usage": 25.0,
            "memory_usage": 40.0,
            "latency": 50.0,
            "error_rate": 0.01
        }
        
        # Noise standard deviations
        self.noise_stds = noise_stds or {
            "cpu_usage": 5.0,
            "memory_usage": 3.0,
            "latency": 10.0,
            "error_rate": 0.005
        }
        
        # OU process mean-reversion rate (higher = faster reversion)
        self.ou_theta = {
            "cpu_usage": 0.3,
            "memory_usage": 0.1,
            "latency": 0.5,
            "error_rate": 0.4
        }
        
        # Metric bounds
        self.bounds = {
            "cpu_usage": (0.0, 100.0),
            "memory_usage": (0.0, 100.0),
            "latency": (1.0, 5000.0),
            "error_rate": (0.0, 1.0)
        }
        
        self.num_steps = int(duration_seconds / sampling_interval)
        self.service_ids = [f"service_{i}" for i in range(num_services)]
    
    def _generate_ou_process(
        self, mu: float, sigma: float, theta: float,
        x0: float, n_steps: int, dt: float
    ) -> np.ndarray:
        """
        Generate an Ornstein-Uhlenbeck process trajectory.
        
        Parameters
        ----------
        mu : float
            Long-term mean.
        sigma : float
            Volatility (noise intensity).
        theta : float
            Mean-reversion rate.
        x0 : float
            Initial value.
        n_steps : int
            Number of timesteps.
        dt : float
            Time increment.
        
        Returns
        -------
        np.ndarray
            Array of shape (n_steps,) with OU process values.
        """
        trajectory = np.zeros(n_steps)
        trajectory[0] = x0
        
        sqrt_dt = np.sqrt(dt)
        noise = self.rng.standard_normal(n_steps)
        
        for i in range(1, n_steps):
            drift = theta * (mu - trajectory[i - 1]) * dt
            diffusion = sigma * sqrt_dt * noise[i]
            trajectory[i] = trajectory[i - 1] + drift + diffusion
        
        return trajectory
    
    def _add_diurnal_pattern(
        self, trajectory: np.ndarray, amplitude: float,
        period_seconds: float = 86400.0, phase: float = 0.0
    ) -> np.ndarray:
        """
        Add a diurnal (sinusoidal) pattern to simulate daily load cycles.
        
        This models the natural ebb and flow of traffic/load that real
        production systems experience.
        """
        t = np.arange(len(trajectory)) * self.sampling_interval
        diurnal = amplitude * np.sin(2 * np.pi * t / period_seconds + phase)
        return trajectory + diurnal
    
    def generate(self) -> pd.DataFrame:
        """
        Generate a complete synthetic telemetry DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Columns: timestamp, service_id, cpu_usage, memory_usage, latency, error_rate
        """
        records = []
        timestamps = np.arange(self.num_steps) * self.sampling_interval
        
        for service_id in self.service_ids:
            # Generate OU processes for each metric
            cpu = self._generate_ou_process(
                mu=self.baselines["cpu_usage"],
                sigma=self.noise_stds["cpu_usage"],
                theta=self.ou_theta["cpu_usage"],
                x0=self.baselines["cpu_usage"] + self.rng.normal(0, 2),
                n_steps=self.num_steps,
                dt=self.sampling_interval
            )
            
            memory = self._generate_ou_process(
                mu=self.baselines["memory_usage"],
                sigma=self.noise_stds["memory_usage"],
                theta=self.ou_theta["memory_usage"],
                x0=self.baselines["memory_usage"] + self.rng.normal(0, 1),
                n_steps=self.num_steps,
                dt=self.sampling_interval
            )
            
            latency = self._generate_ou_process(
                mu=self.baselines["latency"],
                sigma=self.noise_stds["latency"],
                theta=self.ou_theta["latency"],
                x0=self.baselines["latency"] + self.rng.normal(0, 5),
                n_steps=self.num_steps,
                dt=self.sampling_interval
            )
            
            error_rate = self._generate_ou_process(
                mu=self.baselines["error_rate"],
                sigma=self.noise_stds["error_rate"],
                theta=self.ou_theta["error_rate"],
                x0=self.baselines["error_rate"],
                n_steps=self.num_steps,
                dt=self.sampling_interval
            )
            
            # Add diurnal patterns for CPU and memory
            cpu = self._add_diurnal_pattern(cpu, amplitude=8.0, period_seconds=3600.0)
            memory = self._add_diurnal_pattern(memory, amplitude=5.0, period_seconds=3600.0, phase=np.pi / 4)
            
            # Clip to bounds
            cpu = np.clip(cpu, *self.bounds["cpu_usage"])
            memory = np.clip(memory, *self.bounds["memory_usage"])
            latency = np.clip(latency, *self.bounds["latency"])
            error_rate = np.clip(error_rate, *self.bounds["error_rate"])
            
            for i in range(self.num_steps):
                records.append({
                    "timestamp": timestamps[i],
                    "service_id": service_id,
                    "cpu_usage": round(cpu[i], 2),
                    "memory_usage": round(memory[i], 2),
                    "latency": round(latency[i], 2),
                    "error_rate": round(error_rate[i], 6)
                })
        
        df = pd.DataFrame(records)
        df = df.sort_values(["timestamp", "service_id"]).reset_index(drop=True)
        return df
