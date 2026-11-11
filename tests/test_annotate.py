"""Tests for the terminal labeller's input parsing and saving.

Dummy sheet rows with ids like "t1" and placeholder text; nothing touches data/.
"""

import pytest

from support_reply_gap.labeling.annotate import (
    KEY_TO_LABEL,
    InputError,
    apply_labels,
    current_labels,
    load,
    parse_line,
    save,
)
from support_reply_gap.labeling.gold import SHEET_COLUMNS, read_sheet, validate_sheet
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


def test_every_label_has_one_key():
    assert sorted(KEY_TO_LABEL.values()) == sorted(LABEL_NAMES)


@pytest.mark.parametrize("line", ["1 8", "18", " 8 1 "])
def test_keys_with_or_without_spaces(line):
    assert parse_line(line) == ({"crash_stability", "support_experience"}, None)


def test_note_after_hash():
    assert parse_line("9 # trading view") == ({"feature_gap"}, "trading view")


@pytest.mark.parametrize("line", ["x", "1 x", "# only a note", "- 1", "0 9"])
def test_rejects_bad_input(line):
    with pytest.raises(InputError):
        parse_line(line)


def test_save_round_trip_passes_check(tmp_path):
    path = tmp_path / "sheet.csv"
    rows = [
        ",".join(["t1", "a", "1", "2026-05-01", "x", *[""] * len(LABEL_NAMES), ""]),
        ",".join(["t2", "a", "5", "2026-05-01", "y", *[""] * len(LABEL_NAMES), ""]),
    ]
    path.write_text(",".join(SHEET_COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    sheet = load(path)
    apply_labels(sheet, 0, {"crash_stability"}, "a note")
    apply_labels(sheet, 1, {"no_complaint"}, None)
    apply_labels(sheet, 0, {"performance"}, None)  # relabelling replaces, keeps the note
    save(sheet, path)

    reloaded = load(path)
    assert current_labels(reloaded.loc[0]) == ["performance"]
    assert reloaded.loc[0, "notes"] == "a note"
    assert len(validate_sheet(read_sheet(path), expected_ids={"t1", "t2"})) == 2
