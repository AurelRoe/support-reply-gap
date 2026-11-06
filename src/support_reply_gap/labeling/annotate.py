"""Keyboard labelling in the terminal, one review at a time.

    python -m support_reply_gap.labeling.gold label            # gold sheet
    python -m support_reply_gap.labeling.gold label --pass2    # blind second pass

Reads and writes the same CSV sheet that `check` validates, so it can pick up
a sheet already partly labelled elsewhere. It saves after every review. It only
records the keys typed; it never suggests or pre-fills a label.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pandas as pd

from support_reply_gap.labeling.taxonomy import EXCLUSIVE_LABELS, LABEL_NAMES

# One key per label, in taxonomy order: 1-9, then 0, then "-".
KEY_TO_LABEL: dict[str, str] = dict(zip("1234567890-", LABEL_NAMES, strict=True))
LABEL_TO_KEY = {label: key for key, label in KEY_TO_LABEL.items()}

HELP = (
    "labels e.g. '1 8' or '18'  ·  '# text' adds a note  ·  Enter = skip  ·  b = back  ·  q = quit"
)


class InputError(ValueError):
    """The typed line cannot be turned into a valid label set."""


def parse_line(line: str) -> tuple[set[str], str | None]:
    """Turn '1 8 # borderline' into ({crash_stability, support_experience}, 'borderline')."""
    keys_part, _, note = line.partition("#")
    keys = [ch for ch in keys_part if not ch.isspace()]
    unknown = sorted({k for k in keys if k not in KEY_TO_LABEL})
    if unknown:
        raise InputError(f"unknown key(s): {' '.join(unknown)}")
    labels = {KEY_TO_LABEL[k] for k in keys}
    if not labels:
        raise InputError("no label given")
    for name in EXCLUSIVE_LABELS:
        if name in labels and len(labels) > 1:
            raise InputError(f"{name} is exclusive and cannot be combined")
    return labels, (note.strip() or None) if "#" in line else None


def is_labelled(row: pd.Series) -> bool:
    return any(_is_one(row[name]) for name in LABEL_NAMES)


def current_labels(row: pd.Series) -> list[str]:
    return [name for name in LABEL_NAMES if _is_one(row[name])]


def apply_labels(sheet: pd.DataFrame, i: int, labels: set[str], note: str | None) -> None:
    """Set row i to exactly these labels: 1 for each, blank for the rest."""
    for name in LABEL_NAMES:
        sheet.at[i, name] = "1" if name in labels else ""
    if note is not None:
        sheet.at[i, "notes"] = note


def load(path: Path) -> pd.DataFrame:
    # Everything as text with blanks kept as "", so saving changes only the edited cells.
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def save(sheet: pd.DataFrame, path: Path) -> None:
    # Write to a temporary file first, so a crash mid-write cannot leave a half-written sheet.
    tmp = path.with_suffix(".tmp")
    sheet.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _is_one(value: str) -> bool:
    try:
        return float(value) == 1
    except ValueError:
        return False


def _show(sheet: pd.DataFrame, i: int, done: int) -> None:
    row = sheet.loc[i]
    print("\033[2J\033[H", end="")  # clear the terminal so only one review is visible
    print(f"[{done}/{len(sheet)} labelled]  row {i + 1}  ·  {row['app_key']} · {row['score']}★\n")
    print(textwrap.fill(row["content"], width=90) + "\n")
    names = list(LABEL_NAMES)
    for a, b in zip(names[:6], [*names[6:], ""], strict=True):
        right = f"{LABEL_TO_KEY[b]} {b}" if b else ""
        print(f"  {LABEL_TO_KEY[a]} {a:<24}{right}")
    if is_labelled(row):
        print(
            f"\n  current: {' '.join(LABEL_TO_KEY[n] for n in current_labels(row))}"
            f"  ({', '.join(current_labels(row))})"
        )
    if row.get("notes"):
        print(f"  note: {row['notes']}")
    print(f"\n{HELP}")


def run(path: Path, include_labelled: bool = False) -> None:
    """Step through the sheet in its current row order, saving after each answer."""
    sheet = load(path)
    queue = [i for i in sheet.index if include_labelled or not is_labelled(sheet.loc[i])]
    if not queue:
        print("every row is labelled; use --all to revisit")
        return

    pos, message = 0, ""
    while 0 <= pos < len(queue):
        i = queue[pos]
        _show(sheet, i, sum(is_labelled(sheet.loc[j]) for j in sheet.index))
        if message:
            print(f"! {message}")
        message = ""
        line = input("labels> ").strip()

        if line == "q":
            break
        if line == "b":
            pos = max(pos - 1, 0)
            continue
        if line == "":
            pos += 1
            continue
        try:
            labels, note = parse_line(line)
        except InputError as err:
            message = str(err)
            continue
        apply_labels(sheet, i, labels, note)
        save(sheet, path)
        pos += 1

    done = sum(is_labelled(sheet.loc[j]) for j in sheet.index)
    print(f"\n{done}/{len(sheet)} labelled, saved to {path.name}")
