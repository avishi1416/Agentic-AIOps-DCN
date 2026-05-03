"""
FastAPI Server — exposes backend outputs as structured JSON.

Endpoints:
- GET  /api/status           — system status
- POST /api/run              — run full pipeline simulation
- GET  /api/telemetry        — get current telemetry
- GET  /api/anomalies        — get anomaly events
- GET  /api/metrics          — get computed metrics (severity, failure, stability, loss)
- GET  /api/remediation      — get remediation recommendation
- POST /api/remediate        — execute remediation
- GET  /api/evaluation       — get evaluation metrics
- GET  /api/memory           — get operational memory summary
- GET  /api/pipeline/state   — get full pipeline state snapshot

Designed for HuggingFace Spaces compatibility (serves on 0.0.0.0:7860).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Adaptive Loss-Based Autonomous Remediation API",
        description="Research prototype for telemetry-driven autonomous remediation in simulated AIOps systems.",
        version="1.0.0",
    )

    # CORS for HuggingFace frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Pipeline State ───────────────────────────────────────
    from backend.pipeline import AIOpsRemediationPipeline
    pipeline = AIOpsRemediationPipeline()

    # ─── Pydantic Models ──────────────────────────────────────
    class RunConfig(BaseModel):
        num_services: int = 5
        duration_seconds: int = 3600
        with_incidents: bool = True
        run_optimization: bool = True

    class RemediateRequest(BaseModel):
        service_id: str
        action: Optional[str] = None  # None = auto-select

    # ─── Endpoints ────────────────────────────────────────────
    @app.get("/api/status")
    def get_status():
        return {"status": "ready", "version": "1.0.0", "pipeline_initialized": pipeline.is_initialized}

    @app.post("/api/run")
    def run_pipeline(config: RunConfig):
        """Run the full simulation pipeline."""
        try:
            results = pipeline.run(
                num_services=config.num_services,
                duration_seconds=config.duration_seconds,
                with_incidents=config.with_incidents,
                run_optimization=config.run_optimization)
            return {"status": "success", "summary": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/telemetry")
    def get_telemetry(service_id: Optional[str] = None, limit: int = 100):
        """Get telemetry data."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized. Call /api/run first.")
        data = pipeline.get_telemetry(service_id=service_id, limit=limit)
        return {"telemetry": data}

    @app.get("/api/anomalies")
    def get_anomalies(service_id: Optional[str] = None):
        """Get anomaly events."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        events = pipeline.get_anomalies(service_id=service_id)
        return {"anomalies": events}

    @app.get("/api/metrics")
    def get_metrics(service_id: Optional[str] = None):
        """Get computed metrics snapshot."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        return pipeline.get_metrics(service_id=service_id)

    @app.get("/api/remediation")
    def get_remediation(service_id: str):
        """Get remediation recommendation for a service."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        return pipeline.get_remediation_recommendation(service_id)

    @app.post("/api/remediate")
    def execute_remediation(req: RemediateRequest):
        """Execute a remediation action."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        result = pipeline.execute_remediation(req.service_id, req.action)
        return {"result": result}

    @app.get("/api/evaluation")
    def get_evaluation():
        """Get evaluation metrics."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        return pipeline.get_evaluation_metrics()

    @app.get("/api/memory")
    def get_memory():
        """Get operational memory summary."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        return pipeline.get_memory_summary()

    @app.get("/api/pipeline/state")
    def get_pipeline_state():
        """Get full pipeline state snapshot (for frontend consumption)."""
        if not pipeline.is_initialized:
            raise HTTPException(status_code=400, detail="Pipeline not initialized.")
        return pipeline.get_full_state()

    # Serve the vanilla HTML/JS UI
    from fastapi.staticfiles import StaticFiles
    import os
    if os.path.isdir("frontend"):
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


# For direct execution and HuggingFace Spaces
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
