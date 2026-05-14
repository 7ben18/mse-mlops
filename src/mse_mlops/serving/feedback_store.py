from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VALID_FEEDBACK_LABELS = frozenset({"benign", "malignant"})

# Both labeled upload paths store image bytes, so they are promotable.
PROMOTABLE_FEEDBACK_SOURCES = frozenset({"upload_labeled", "doctor_review"})

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

def is_promotable_feedback(entry: dict[str, Any]) -> bool:
    """Return whether a feedback entry is ready for train-set promotion.

    Only labeled feedback entries are counted. Prediction-only feedback remains
    excluded because it may not have a stored image file.
    """
    label = str(entry.get("label", "")).strip().lower()

    return (
        label in VALID_FEEDBACK_LABELS
        and entry.get("source") in PROMOTABLE_FEEDBACK_SOURCES
        and not entry.get("promoted_to_train", False)
    )

def count_unpromoted_labeled_entries(feedback_file: Path) -> int:
    """Count labeled uploaded feedback entries that have not been promoted."""
    return sum(
        1
        for entry in load_feedback_entries(feedback_file)
        if is_promotable_feedback(entry)
    )
