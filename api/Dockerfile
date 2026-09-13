# Stage 1: build the React dashboard
FROM node:22-slim AS dashboard-builder

WORKDIR /workspace

# Install pnpm
RUN npm install -g pnpm

# Copy workspace manifests first for better layer caching
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/dashboard/package.json packages/dashboard/

RUN pnpm install --frozen-lockfile --filter @openbiometrics/dashboard

# Copy dashboard source and build
COPY packages/dashboard/ packages/dashboard/
RUN pnpm --filter @openbiometrics/dashboard build

# Stage 2: Python runtime — serves API + built dashboard
FROM python:3.12-slim

# System libs required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages
COPY engine/ engine/
COPY api/ api/

# Optional extras for the engine (e.g. "liveness,document,person")
# Pass at build time: docker build --build-arg ENGINE_EXTRAS=liveness,document .
ARG ENGINE_EXTRAS=""

RUN pip install --no-cache-dir ./engine && \
    if [ -n "$ENGINE_EXTRAS" ]; then \
        pip install --no-cache-dir "./engine[$ENGINE_EXTRAS]"; \
    fi && \
    pip install --no-cache-dir ./api

# Copy the built dashboard so FastAPI's StaticFiles mount picks it up
COPY --from=dashboard-builder /workspace/packages/dashboard/dist/ packages/dashboard/dist/

# Models are mounted at runtime; declare the volume and point the engine at it
VOLUME ["/app/models"]
ENV OPENBIOMETRICS_MODELS_DIR=/app/models

# doctr OCR model weights — mounted as a volume so they persist across container restarts
VOLUME ["/app/doctr-cache"]
ENV DOCTR_CACHE_DIR=/app/doctr-cache

WORKDIR /app/api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
