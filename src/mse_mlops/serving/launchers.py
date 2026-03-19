from __future__ import annotations

import os
import sys
from pathlib import Path


def run_api() -> None:
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")  # noqa: S104
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("mse_mlops.serving.api:app", host=host, port=port)


def run_ui() -> None:
    ui_path = Path(__file__).with_name("ui.py")
    port = os.environ.get("STREAMLIT_PORT", "7777")
    os.execv(  # noqa: S606
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.port",
            port,
            "--server.address",
            "0.0.0.0",  # noqa: S104
        ],
    )
