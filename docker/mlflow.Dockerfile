# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN uv sync --locked --no-install-project --no-dev

CMD ["uv", "run", "--no-sync", "--frozen", "--no-dev", "mlflow", "server"]
