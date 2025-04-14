# Build stage
FROM python:3.12-slim AS builder

# Set shell to bash and enable pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV UV_CACHE_DIR=/opt/uv-cache/
ENV UV_LINK_MODE=copy
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bluez \
        ffmpeg \
        libudev-dev \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libgammu-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
        libpcap-dev \
        libturbojpeg0 \
        libyaml-dev \
        libxml2 \
        build-essential \
        git \
        cmake \
        libyaml-dev \
        libxml2 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (in a separate layer)
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/opt/uv-cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy the project and install it
COPY ./src /code/src
RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/opt/uv-cache \
    uv sync --frozen --all-groups --dev

# Production stage
FROM python:3.12-slim

# Set shell to bash and enable pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /code

# Install minimal runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libyaml-dev \
        libxml2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the virtual environment from builder
COPY --from=builder /code/.venv /code/.venv

# Create non-root user with specific UID/GID
RUN groupadd -r nonroot --gid=1000 && \
    useradd -r -g nonroot --uid=1000 --shell=/bin/bash nonroot && \
    chown -R nonroot:nonroot /code

USER nonroot

# Set environment variables
ENV PATH="/code/.venv/bin:$PATH"
ENV SHELL="/bin/bash"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
