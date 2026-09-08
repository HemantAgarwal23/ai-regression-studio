# AI Regression Studio - Dockerfile
#
# Multi-stage build: compilers and build headers stay in the builder stage so
# the shipped image contains only the runtime and the installed wheels.

# ---------- Stage 1: build the dependency set ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# build-essential is needed to compile any package without a prebuilt wheel.
# It lives only in this stage and is never copied into the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so the install layer is cached across code changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

# curl is required by the health check below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user. A container process running as root can escalate
# a container escape into host root, so nothing here needs those privileges.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Only the files the app actually serves.
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser assets/ ./assets/
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
