# Adaptive Loss-Based Autonomous Remediation in Simulated AIOps Systems

## Research Prototype

A modular, mathematically grounded backend for simulating telemetry-driven autonomous remediation in AIOps systems. Inspired by [AIOpsLab (arXiv: 2501.06706)](https://arxiv.org/abs/2501.06706).

---

## Architecture

```
backend/
├── telemetry/          # Telemetry ingestion & synthetic generation
├── anomaly_detection/  # Threshold + z-score anomaly detection
├── severity/           # Normalized severity function E_n(t)
├── failure_prediction/ # Sigmoid-based failure prediction F_n(t)
├── stability/          # Stability estimation S_n(t) + downtime model
├── loss_engine/        # Cumulative operational loss L
├── optimization/       # AdaGrad adaptive parameter optimization
├── remediation/        # Remediation selection (argmin E[L|X,a])
├── memory/             # Cosine-similarity operational memory
├── evaluation/         # LRR, RSR, FRR, SIS, MTTR metrics
├── api/                # FastAPI JSON endpoints
└── pipeline.py         # Main orchestrator
```

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Simulation
```bash
python run_simulation.py
```

### Start API Server
```bash
uvicorn backend.api.server:app --host 0.0.0.0 --port 7860
```

### API Documentation
Once running, visit: `http://localhost:7860/docs`

## Key Equations

| Component | Formula |
|-----------|---------|
| Severity | `E_n(t) = w_c(CPU/CPU_max)² + w_l(Lat/L_thr)² + w_m(Mem/M_max)² + w_e·log(1+err)` |
| Failure | `F_n(t) = 1/(1+e^(-k(E_n(t)-θ)))` |
| Stability | `S_n(t) = exp(-(μ_E + σ²_E + ρ·|dE/dt|))` |
| Loss | `L = ∫[α·E_n + β·D_n + γ·F_n + δ·(1-S_n)] dt` |
| Optimization | `θ_(t+1) = θ_t - η/√(G_t+ε) · ∇L` |
| Selection | `a* = argmin E[L | X, a_i]` |

## HuggingFace Spaces

This backend is compatible with HuggingFace Spaces (Docker SDK):

1. Set space SDK to **Docker**
2. The `app.py` entry point exposes the FastAPI app
3. All endpoints available under `/api/`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System status |
| POST | `/api/run` | Run full pipeline |
| GET | `/api/telemetry` | Get telemetry data |
| GET | `/api/anomalies` | Get anomaly events |
| GET | `/api/metrics` | Get computed metrics |
| GET | `/api/remediation?service_id=X` | Get recommendation |
| POST | `/api/remediate` | Execute remediation |
| GET | `/api/evaluation` | Evaluation metrics |
| GET | `/api/memory` | Memory summary |
| GET | `/api/pipeline/state` | Full state snapshot |
# Agentic-AIOps-DCN
