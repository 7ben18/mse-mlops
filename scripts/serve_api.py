from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("mse_mlops.serving.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
