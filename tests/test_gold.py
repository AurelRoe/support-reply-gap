"""Validator tests for the gold labelling sheet.

The frames below are dummy validator inputs (ids like "t1", no review text).
They are not labelled examples and never touch data/gold/.
"""

import pandas as pd
import pytest

from support_reply_gap.labeling.gold import SHEET_COLUMNS, SheetError, validate_sheet
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


def make_sheet(rows: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Build a sheet where every cell is blank except the ones given per review_id."""
    records = []
    for review_id, cells in rows.items():
        record = {col: "" for col in SHEET_COLUMNS}
        record["review_id"] = review_id
        record.update(cells)
        records.append(record)
    return pd.DataFrame(records, columns=SHEET_COLUMNS)


def test_valid_sheet_returns_ids_and_labels():
    sheet = make_sheet(
        {"t1": {"crash_stability": 1, "support_experience": "1"}, "t2": {"no_complaint": 1}}
    )
    result = validate_sheet(sheet, expected_ids={"t1", "t2"})
    assert list(result.columns) == ["review_id", *LABEL_NAMES]
    assert result.set_index("review_id").loc["t1"].sum() == 2


def test_blank_and_zero_both_mean_not_this_label():
    sheet = make_sheet({"t1": {"crash_stability": 1, "performance": 0}})
    result = validate_sheet(sheet)
    assert result.loc[0, "performance"] == 0
    assert result.loc[0, "login_access"] == 0


def test_rejects_values_other_than_0_1_blank():
    sheet = make_sheet({"t1": {"crash_stability": 2}, "t2": {"performance": "x"}})
    with pytest.raises(SheetError, match="crash_stability.*not 0, 1"):
        validate_sheet(sheet)


def test_rejects_unlabelled_rows_unless_partial():
    sheet = make_sheet({"t1": {"crash_stability": 1}, "t2": {}})
    with pytest.raises(SheetError, match="no label"):
        validate_sheet(sheet)
    assert list(validate_sheet(sheet, allow_partial=True)["review_id"]) == ["t1"]


@pytest.mark.parametrize("exclusive", ["no_complaint", "complaint_unspecified"])
def test_rejects_exclusive_label_with_another(exclusive):
    sheet = make_sheet({"t1": {exclusive: 1, "fees_pricing": 1}})
    with pytest.raises(SheetError, match="exclusive"):
        validate_sheet(sheet)


def test_rejects_changed_label_columns():
    sheet = make_sheet({"t1": {"crash_stability": 1}}).rename(columns={"performance": "speed"})
    with pytest.raises(SheetError, match="missing columns.*performance"):
        validate_sheet(sheet)


def test_rejects_ids_that_do_not_match_the_sample():
    sheet = make_sheet({"t1": {"crash_stability": 1}, "t1b": {"crash_stability": 1}})
    with pytest.raises(SheetError, match="not in the drawn sample"):
        validate_sheet(sheet, expected_ids={"t1", "t2"})


def test_day_cap_does_not_starve_rare_star_ratings():
    from support_reply_gap.labeling.sampling import stratified_sample

    # One busy day: twenty 5-star reviews and two 2-star reviews for the same app.
    day = pd.Timestamp("2026-05-01")
    rows = [{"review_id": f"t5_{i}", "score": 5} for i in range(20)]
    rows += [{"review_id": f"t2_{i}", "score": 2} for i in range(2)]
    pool = pd.DataFrame(rows).assign(app_key="a", posted_at=day)

    sample = stratified_sample(pool, per_score={2: 2, 5: 2})
    assert (sample["score"] == 2).sum() == 2
