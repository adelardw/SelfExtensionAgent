# Use uv for fast builds
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Final stage
FROM python:3.13-slim-bookworm

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV ENV=prod

COPY . .

# Create data directory for sqlite
RUN mkdir -p data

ENTRYPOINT ["python", "main.py"]
