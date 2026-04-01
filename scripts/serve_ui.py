from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    ui_path = Path(__file__).resolve().parents[1] / "src" / "mse_mlops" / "serving" / "ui.py"
    port = os.environ.get("STREAMLIT_PORT", "7777")
    os.execv(
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
            "0.0.0.0",
        ],
    )


if __name__ == "__main__":
    main()
