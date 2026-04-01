FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY uv.lock pyproject.toml README.md /app/
RUN uv sync --locked --no-install-project --no-dev --group ui

COPY src /app/src
COPY scripts /app/scripts
RUN uv sync --locked --no-dev --group ui

CMD ["uv", "run", "--no-sync", "--frozen", "--no-dev", "--group", "ui", "python", "scripts/serve_ui.py"]
