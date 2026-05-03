"""
Incident Simulator.

Injects realistic failure patterns into telemetry streams:
- CPU overload, Memory leaks, Latency spikes
- Cascading degradation, Service instability
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Incident:
    """Describes a simulated incident."""
    incident_id: str
    incident_type: str
    service_id: str
    start_time: float
    end_time: float
    severity: str
    affected_metrics: List[str] = field(default_factory=list)


class IncidentSimulator:
    """Injects realistic incident patterns into telemetry data."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.incidents: List[Incident] = []
        self._counter = 0

    def inject_incidents(self, df: pd.DataFrame, incident_probability: float = 0.15,
                         incident_types: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[Incident]]:
        if incident_types is None:
            incident_types = ["cpu_overload", "memory_leak", "latency_spike",
                              "cascading_degradation", "service_instability"]
        df = df.copy()
        self.incidents = []
        services = df["service_id"].unique()
        timestamps = sorted(df["timestamp"].unique())
        total_time = timestamps[-1] - timestamps[0]
        window_size = max(total_time * 0.1, 60.0)
        num_windows = int(total_time / window_size)

        for svc in services:
            smask = df["service_id"] == svc
            for w in range(num_windows):
                if self.rng.random() > incident_probability:
                    continue
                ws = timestamps[0] + w * window_size
                we = ws + window_size
                itype = self.rng.choice(incident_types)
                dur = self.rng.uniform(0.2, 0.8)
                ist = ws + self.rng.uniform(0, window_size * 0.2)
                ien = ist + window_size * dur
                mask = smask & (df["timestamp"] >= ist) & (df["timestamp"] <= ien)
                if mask.sum() == 0:
                    continue
                idx = df.index[mask]
                n = len(idx)
                df = getattr(self, f"_inject_{itype}")(df, idx, n)
                self._counter += 1
                self.incidents.append(Incident(
                    incident_id=f"INC-{self._counter:04d}", incident_type=itype,
                    service_id=svc, start_time=ist, end_time=ien,
                    severity="critical" if self.rng.random() > 0.4 else "warning",
                    affected_metrics=self._affected(itype)))
        return df, self.incidents

    def _inject_cpu_overload(self, df, idx, n):
        sig = 1.0 / (1.0 + np.exp(-np.linspace(-6, 6, n)))
        tgt = self.rng.uniform(85, 98)
        df.loc[idx, "cpu_usage"] = np.clip(df.loc[idx, "cpu_usage"].values + sig * (tgt - df.loc[idx, "cpu_usage"].values), 0, 100)
        df.loc[idx, "latency"] = np.clip(df.loc[idx, "latency"].values * (1 + 0.5 * sig), 1, 5000)
        df.loc[idx, "error_rate"] = np.clip(df.loc[idx, "error_rate"].values + 0.02 * sig, 0, 1)
        return df

    def _inject_memory_leak(self, df, idx, n):
        ramp = np.maximum.accumulate(np.clip(np.linspace(0, 1, n) + self.rng.normal(0, 0.02, n), 0, 1))
        tgt = self.rng.uniform(88, 99)
        base = df.loc[idx, "memory_usage"].values[0]
        df.loc[idx, "memory_usage"] = np.clip(base + ramp * (tgt - base), 0, 100)
        df.loc[idx, "latency"] = np.clip(df.loc[idx, "latency"].values * (1 + 0.3 * ramp), 1, 5000)
        return df

    def _inject_latency_spike(self, df, idx, n):
        pp = max(1, n // 4)
        t = np.arange(n, dtype=float)
        rise = np.exp(3.0 * t[:pp] / pp) - 1
        rise = rise / (rise[-1] if rise[-1] != 0 else 1)
        decay = np.exp(-0.05 * (t[pp:] - pp))
        sp = np.concatenate([rise, decay])[:n]
        mag = self.rng.uniform(500, 3000)
        df.loc[idx, "latency"] = np.clip(df.loc[idx, "latency"].values + mag * sp, 1, 5000)
        df.loc[idx, "error_rate"] = np.clip(df.loc[idx, "error_rate"].values + 0.1 * sp, 0, 1)
        return df

    def _inject_cascading_degradation(self, df, idx, n):
        t = np.linspace(0, 1, n)
        s = lambda shift, k: 1.0 / (1.0 + np.exp(-k * (t - shift)))
        df.loc[idx, "latency"] = np.clip(df.loc[idx, "latency"].values + 800 * s(0.1, 15), 1, 5000)
        df.loc[idx, "cpu_usage"] = np.clip(df.loc[idx, "cpu_usage"].values + 50 * s(0.3, 12), 0, 100)
        df.loc[idx, "memory_usage"] = np.clip(df.loc[idx, "memory_usage"].values + 40 * s(0.5, 10), 0, 100)
        df.loc[idx, "error_rate"] = np.clip(df.loc[idx, "error_rate"].values + 0.3 * s(0.6, 10), 0, 1)
        return df

    def _inject_service_instability(self, df, idx, n):
        t = np.linspace(0, 4 * np.pi, n)
        env = np.linspace(0.3, 1.0, n)
        osc = env * np.sin(t)
        df.loc[idx, "cpu_usage"] = np.clip(df.loc[idx, "cpu_usage"].values + 30 * osc, 0, 100)
        df.loc[idx, "memory_usage"] = np.clip(df.loc[idx, "memory_usage"].values + 25 * np.abs(osc), 0, 100)
        df.loc[idx, "latency"] = np.clip(df.loc[idx, "latency"].values + 200 * np.abs(osc), 1, 5000)
        df.loc[idx, "error_rate"] = np.clip(df.loc[idx, "error_rate"].values + 0.15 * np.maximum(0, osc) ** 2, 0, 1)
        return df

    @staticmethod
    def _affected(itype):
        return {"cpu_overload": ["cpu_usage", "latency", "error_rate"],
                "memory_leak": ["memory_usage", "latency"],
                "latency_spike": ["latency", "error_rate"],
                "cascading_degradation": ["cpu_usage", "memory_usage", "latency", "error_rate"],
                "service_instability": ["cpu_usage", "memory_usage", "latency", "error_rate"]}.get(itype, [])
