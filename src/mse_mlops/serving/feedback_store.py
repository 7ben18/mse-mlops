from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_feedback_entry(feedback_file: Path, entry: dict[str, Any]) -> None:
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    with feedback_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_feedback_entries(feedback_file: Path) -> list[dict[str, Any]]:
    if not feedback_file.exists():
        return []

    entries: list[dict[str, Any]] = []
    with feedback_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_feedback_entries(feedback_file: Path, entries: list[dict[str, Any]]) -> None:
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(entry) for entry in entries)
    if payload:
        payload += "\n"
    feedback_file.write_text(payload, encoding="utf-8")
