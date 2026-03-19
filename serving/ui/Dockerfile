FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync --no-dev

COPY app.py .

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=7777", "--server.address=0.0.0.0"]
