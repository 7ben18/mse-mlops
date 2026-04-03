# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Change the working directory to the `app` directory
WORKDIR /app

# Copy the lockfile and project metadata into the image
COPY uv.lock pyproject.toml README.md /app/

# Install dependencies
RUN uv sync --locked --no-install-project --no-dev

# Copy only the training code and default config needed at runtime
COPY src /app/src
COPY scripts /app/scripts
COPY config /app/config

# Sync the project
RUN uv sync --locked --no-dev

CMD ["uv", "run", "--no-sync", "--frozen", "--no-dev", "python", "scripts/train.py"]
