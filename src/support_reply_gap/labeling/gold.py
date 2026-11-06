"""Stage B5 gold set: draw the labelling sheet and validate it once it is filled in.

    python -m support_reply_gap.labeling.gold draw
    python -m support_reply_gap.labeling.gold check [--allow-partial]

The sheet holds review text and stays local (data/labelling/ is gitignored). Only
review ids and labels are written to data/gold/, which is committed. This module
never sets a label value. It only copies the 0/1 cells the labeller typed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from support_reply_gap.labeling.taxonomy import EXCLUSIVE_LABELS, LABEL_NAMES

# src/support_reply_gap/labeling/gold.py -> repo root. Anchoring here rather than on
# the working directory means the files land in the same place from any folder.
ROOT = Path(__file__).resolve().parents[3]

DEV_IDS_PATH = ROOT / "data/gold/dev_sample_ids.txt"
SAMPLE_IDS_PATH = ROOT / "data/gold/gold_sample_ids.txt"
LABELS_PATH = ROOT / "data/gold/gold_labels.csv"
SHEET_PATH = ROOT / "data/labelling/gold_sheet.csv"

INFO_COLUMNS = ["review_id", "app_key", "score", "posted_at", "content"]
SHEET_COLUMNS = [*INFO_COLUMNS, *LABEL_NAMES, "notes"]

SMALL_CLASS = 20


class SheetError(ValueError):
    """The labelling sheet breaks a rule. The message lists every problem found."""


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def build_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """Lay out the sample as a blank sheet: review info, one column per label, notes."""
    sheet = sample[INFO_COLUMNS].copy()
    sheet["posted_at"] = pd.to_datetime(sheet["posted_at"]).dt.date
    for name in LABEL_NAMES:
        sheet[name] = ""
    sheet["notes"] = ""
    return sheet


def validate_sheet(
    sheet: pd.DataFrame,
    expected_ids: set[str] | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Check a filled-in sheet and return review_id plus the 0/1 label columns.

    A blank cell counts as 0, so only the labels that apply need typing. A row that
    is entirely blank is unlabelled. It is an error unless allow_partial is set, and
    in that case it is dropped from the result.
    """
    problems: list[str] = []

    missing = [c for c in ["review_id", *LABEL_NAMES] if c not in sheet.columns]
    unknown = [c for c in sheet.columns if c not in SHEET_COLUMNS]
    if missing:
        problems.append(f"missing columns: {missing}")
    if unknown:
        problems.append(f"unknown columns (taxonomy is frozen): {unknown}")
    if problems:
        raise SheetError("\n".join(problems))

    ids = sheet["review_id"].astype(str).str.strip()
    duplicated = sorted(ids[ids.duplicated()].unique())
    if duplicated:
        problems.append(f"duplicate review_ids: {duplicated[:10]}")
    if expected_ids is not None:
        absent = expected_ids - set(ids)
        extra = set(ids) - expected_ids
        if absent:
            problems.append(f"{len(absent)} sampled reviews missing from the sheet")
        if extra:
            problems.append(f"{len(extra)} review_ids not in the drawn sample")

    labels = pd.DataFrame(index=sheet.index)
    for name in LABEL_NAMES:
        raw = sheet[name]
        blank = raw.isna() | (raw.astype(str).str.strip() == "")
        value = pd.to_numeric(raw.where(~blank), errors="coerce")
        bad = ~blank & ~value.isin([0, 1])
        for i in sheet.index[bad][:10]:
            problems.append(f"{ids[i]} / {name}: {raw[i]!r} is not 0, 1 or blank")
        labels[name] = value.fillna(0).astype(int)

    n_labels = labels.sum(axis=1)
    for name in EXCLUSIVE_LABELS:
        clash = (labels[name] == 1) & (n_labels > 1)
        for i in sheet.index[clash]:
            others = [n for n in LABEL_NAMES if labels.at[i, n] == 1 and n != name]
            problems.append(f"{ids[i]}: {name} is exclusive but also has {others}")

    unlabelled = n_labels == 0
    if unlabelled.any() and not allow_partial:
        problems.append(
            f"{int(unlabelled.sum())} rows have no label, e.g. {list(ids[unlabelled][:5])}"
        )

    if problems:
        raise SheetError("\n".join(problems))

    result = pd.concat([ids.rename("review_id"), labels], axis=1)
    return result.loc[~unlabelled].reset_index(drop=True)


def class_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-class support, with classes too small to judge flagged."""
    counts = labels[list(LABEL_NAMES)].sum().sort_values(ascending=False)
    return pd.DataFrame(
        {
            "count": counts,
            "share": (counts / len(labels)).round(3),
            "small": counts < SMALL_CLASS,
        }
    )


def read_sheet(path: Path) -> pd.DataFrame:
    # utf-8-sig accepts files with or without a byte-order mark, which spreadsheet
    # apps add or drop inconsistently on export.
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": str})


def draw() -> None:
    """Draw the gold sample and write a blank sheet. Never overwrites."""
    if SHEET_PATH.exists() or SAMPLE_IDS_PATH.exists():
        sys.exit(f"refusing to overwrite: {SHEET_PATH.name} or {SAMPLE_IDS_PATH.name} exists")

    # Imported here so `check` runs without Postgres or lingua.
    from support_reply_gap.analysis.data import load_reviews
    from support_reply_gap.config import ANALYSIS_WINDOW_START
    from support_reply_gap.labeling.sampling import (
        GOLD_SAMPLE_PER_SCORE,
        add_language,
        add_text_flags,
        drop_repeated_text,
        eligible_for_labelling,
        stratified_sample,
    )

    df = load_reviews(since=ANALYSIS_WINDOW_START)
    eligible = drop_repeated_text(eligible_for_labelling(add_language(add_text_flags(df))))

    dev_ids = read_ids(DEV_IDS_PATH)
    pool = eligible[~eligible["review_id"].isin(dev_ids)]
    print(f"{len(eligible):,} eligible, {len(pool):,} after excluding the B4 dev sample")

    sample = stratified_sample(pool, per_score=GOLD_SAMPLE_PER_SCORE)
    if set(sample["review_id"]) & dev_ids:
        raise RuntimeError("gold sample overlaps the dev sample")

    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_sheet(sample).to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")
    SAMPLE_IDS_PATH.write_text("\n".join(sorted(sample["review_id"])) + "\n", encoding="utf-8")

    print(f"{len(sample)} drawn")
    print(pd.crosstab(sample["app_key"], sample["score"], margins=True))
    print(f"wrote {SHEET_PATH.relative_to(ROOT)} (local only)")
    print(f"wrote {SAMPLE_IDS_PATH.relative_to(ROOT)} (commit this)")


def check(allow_partial: bool) -> None:
    """Validate the filled sheet. When complete, write the committed labels file."""
    try:
        labels = validate_sheet(
            read_sheet(SHEET_PATH), read_ids(SAMPLE_IDS_PATH), allow_partial=allow_partial
        )
    except SheetError as err:
        sys.exit(f"sheet has problems:\n{err}")

    total = len(read_ids(SAMPLE_IDS_PATH))
    print(f"{len(labels)} of {total} reviews labelled")
    if len(labels):
        print(class_summary(labels))
        print("labels per review:", labels[list(LABEL_NAMES)].sum(axis=1).value_counts().to_dict())

    if len(labels) < total:
        print("partial sheet: nothing written")
        return
    if LABELS_PATH.exists():
        # Git is the safety net for changes to committed labels: delete, re-run, review the diff.
        sys.exit(f"refusing to overwrite {LABELS_PATH.name}; delete it first to regenerate")
    labels.sort_values("review_id").to_csv(LABELS_PATH, index=False)
    print(f"wrote {LABELS_PATH.relative_to(ROOT)} (commit this)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m support_reply_gap.labeling.gold")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draw", help="draw the gold sample and write a blank sheet")
    check_parser = sub.add_parser("check", help="validate the sheet; write labels when complete")
    check_parser.add_argument(
        "--allow-partial", action="store_true", help="accept unlabelled rows to check progress"
    )
    args = parser.parse_args()

    if args.command == "draw":
        draw()
    elif args.command == "check":
        check(args.allow_partial)


if __name__ == "__main__":
    main()
