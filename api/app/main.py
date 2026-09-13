"""OpenBiometrics -- FastAPI application entry point.

Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app import deps
from app.routes import router as api_router
from openbiometrics import __version__
from openbiometrics.config import (
    BiometricConfig,
    DocumentConfig,
    EventsConfig,
    FaceConfig,
    IdentityConfig,
    LivenessConfig,
    PersonConfig,
    VideoConfig,
)

logger = logging.getLogger("openbiometrics.api")

DASHBOARD_DIR = Path(__file__).parent.parent.parent / "packages" / "dashboard" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = os.environ.get("OPENBIOMETRICS_MODELS_DIR") or str(
        Path(__file__).parent.parent.parent / "models"
    )
    # ONNX device: -1 = CPU (default), >= 0 = CUDA device ID.
    # GPU inference requires the onnxruntime-gpu build:
    #   pip install -e "engine[gpu]"  (or swap onnxruntime for onnxruntime-gpu)
    ctx_id = int(os.environ.get("OPENBIOMETRICS_CTX_ID", "-1"))

    config = BiometricConfig(
        face=FaceConfig(
            models_dir=models_dir,
            ctx_id=ctx_id,
        ),
        document=DocumentConfig(
            enabled=True,
            models_dir=models_dir,
            enable_face_extraction=True,
            ctx_id=ctx_id,
        ),
        liveness=LivenessConfig(
            enabled=True,
        ),
        person=PersonConfig(
            enabled=True,
            models_dir=models_dir,
            ctx_id=ctx_id,
        ),
        video=VideoConfig(
            enabled=True,
        ),
        events=EventsConfig(
            enabled=True,
            webhooks_enabled=True,
        ),
        identity=IdentityConfig(
            enabled=True,
            watchlist_dir="./watchlists",
        ),
    )

    deps.init_services(config)

    kernel = deps.get_kernel()
    health = kernel.health()
    print("OpenBiometrics ready.")
    print(f"  Models: {models_dir}")
    print(f"  Device: {'GPU:' + str(config.face.ctx_id) if config.face.ctx_id >= 0 else 'CPU'}")
    print(f"  Modules: {', '.join(m for m, ok in health.modules.items() if ok)}")

    yield

    deps.shutdown_services()
    print("Shutting down.")


app = FastAPI(
    title="OpenBiometrics",
    description="Open-source biometric platform -- face detection, recognition, liveness, quality",
    version=__version__,
    lifespan=lifespan,
)


# -- Audit logging middleware --------------------------------------------------

@app.middleware("http")
async def audit_log_middleware(request: Request, call_next):
    """Log every request with method, path, status code, duration, and version header."""
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["X-OpenBiometrics-Version"] = __version__
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# -- Routes --------------------------------------------------------------------

app.include_router(api_router, prefix="/api/v1")

# Serve React dashboard
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")
