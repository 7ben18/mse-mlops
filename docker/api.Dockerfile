FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY uv.lock pyproject.toml README.md /app/
RUN uv sync --locked --no-install-project --no-dev --group api

COPY src /app/src
RUN uv sync --locked --no-dev --group api

CMD ["uv", "run", "--no-sync", "--frozen", "--no-dev", "--group", "api", "serve-api"]
