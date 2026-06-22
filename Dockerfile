# ====================================================
# STAGE 1: BUILDER STAGE
# ====================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV Optimization: Avoid re-downloading Python, use system interpreter
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Step 1: Install dependencies only (Caching layer)
# Use --mount=type=cache for faster subsequent builds
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Step 2: Copy source code and sync project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# ====================================================
# STAGE 2: FINAL PRODUCTION IMAGE
# ====================================================
FROM python:3.12-slim-bookworm AS production

# Environment variables for Python & Uvicorn
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PADDLE_PDX_CACHE_HOME=/app/.paddlex \
    YOLO_CONFIG_DIR=/app/ultralytics \
    MPLCONFIGDIR=/app/matplotlib

# Install system dependencies for OpenCV (Required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    dumb-init \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup non-root user for security (UID 1000 is standard for host users)
RUN groupadd --system --gid 1000 nonroot \
    && useradd --system --gid 1000 --uid 1000 --create-home nonroot

# Copy built application from builder stage
COPY --from=builder --chown=nonroot:nonroot /app /app

# Copy entrypoint and make executable
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# Pre-create PaddleX & Ultralytics config directories with proper ownership
RUN mkdir -p /app/.paddlex /app/ultralytics /app/matplotlib && chown nonroot:nonroot /app/.paddlex /app/ultralytics /app/matplotlib

# Use `/app` as the working directory
WORKDIR /app

# Health check to ensure service availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${APP_PORT}/api/v1/health || exit 1

# Dumb-init PID 1 → entrypoint.sh (fix perms) → setpriv nonroot → uvicorn
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint.sh"]

# Run the application (using shell expansion for environment variables)
CMD ["/bin/sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${APP_PORT} --no-access-log"]
