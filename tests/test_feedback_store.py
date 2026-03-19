from pathlib import Path

from mse_mlops.serving.feedback_store import (
    append_feedback_entry,
    load_feedback_entries,
    write_feedback_entries,
)


def test_append_and_load_feedback_entries(tmp_path: Path):
    feedback_file = tmp_path / "feedback.jsonl"

    append_feedback_entry(feedback_file, {"image_id": "1", "label": None})
    append_feedback_entry(feedback_file, {"image_id": "2", "label": "benign"})

    assert load_feedback_entries(feedback_file) == [
        {"image_id": "1", "label": None},
        {"image_id": "2", "label": "benign"},
    ]


def test_write_feedback_entries_overwrites_file(tmp_path: Path):
    feedback_file = tmp_path / "feedback.jsonl"
    append_feedback_entry(feedback_file, {"image_id": "1", "label": None})

    write_feedback_entries(feedback_file, [{"image_id": "1", "label": "malignant"}])

    assert load_feedback_entries(feedback_file) == [{"image_id": "1", "label": "malignant"}]
