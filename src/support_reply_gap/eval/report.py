"""Build reports/eval.md, the eval table, from the result files in reports/eval/.

Generated, never edited by hand: every number traces back to a result file, and the
human ceiling is recomputed from the committed self-agreement labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from support_reply_gap.eval.compare import paired_difference
from support_reply_gap.eval.harness import FEW_EXAMPLES, RESULTS_DIR, gold_fingerprint
from support_reply_gap.eval.metrics import exact_match, macro_f1
from support_reply_gap.labeling.gold import LABELS_PATH, PASS2_PATH, ROOT
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

EVAL_TABLE_PATH = ROOT / "reports/eval.md"
REFERENCE_MODEL = "baseline"


def gold_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_PATH, dtype={"review_id": str})


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def latest_per_model(results: list[dict]) -> list[dict]:
    """The most recent run of each model, in the order models first appeared."""
    latest: dict[str, dict] = {}
    for result in sorted(results, key=lambda r: r["run_at"]):
        latest[result["model"]] = result
    return list(latest.values())


def ceiling() -> dict:
    """Human self-agreement, recomputed from the committed labels rather than typed in."""
    from support_reply_gap.labeling.agreement import align

    read = {"dtype": {"review_id": str}}
    p1, p2 = align(pd.read_csv(LABELS_PATH, **read), pd.read_csv(PASS2_PATH, **read))
    return {"n": len(p1), "macro_f1": macro_f1(p1, p2), "exact_match": exact_match(p1, p2)}


def fmt_f1(split: dict | None) -> str:
    if split is None:
        return "–"
    low, high = split["macro_f1_ci95"]
    return f"{split['macro_f1']:.2f} ({low:.2f}–{high:.2f})"


def fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def summary_table(results: list[dict], human: dict, fingerprint: str) -> list[str]:
    header = (
        "| Model | Val macro-F1 (95% CI) | Test macro-F1 (95% CI) | Test exact match "
        "| Cost / 1k reviews | p95 latency | Run |"
    )
    human_row = (
        f"| Human self-agreement ({human['n']} reviews) | {human['macro_f1']:.2f} | | "
        f"{human['exact_match']:.2f} | | | ceiling |"
    )
    lines = [header, "|---|---:|---:|---:|---:|---:|---|", human_row]
    for r in results:
        val, test = r["splits"].get("val"), r["splits"].get("test")
        timing = test or val
        stale = "" if r["gold_fingerprint"] == fingerprint else " ⚠ older gold set"
        lines.append(
            f"| {r['model']} | {fmt_f1(val)} | {fmt_f1(test)} | "
            f"{'–' if test is None else format(test['exact_match'], '.2f')} | "
            f"${timing['cost_per_1k_usd']:.2f} | {fmt_ms(timing['latency_p95_ms'])} | "
            f"{r['run_at'][:10]}, `{r['commit']}`{stale} |"
        )
    return lines


def per_class_table(result: dict) -> list[str]:
    splits = [s for s in ("val", "test") if s in result["splits"]]
    rows = {s: {c["label"]: c for c in result["splits"][s]["per_class"]} for s in splits}
    lines = [
        "| Label | " + " | ".join(f"{s.capitalize()} F1 (support)" for s in splits) + " |",
        "|---|" + "---:|" * len(splits),
    ]
    for label in LABEL_NAMES:
        cells = []
        for s in splits:
            c = rows[s][label]
            f1 = "n/a" if c["f1"] is None or pd.isna(c["f1"]) else f"{c['f1']:.2f}"
            text = f"{f1} ({c['support']})"
            cells.append(f"*{text}*" if c["few_examples"] else text)
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    return lines


def confusion_table(result: dict, split: str = "val") -> list[str]:
    matrix = result["splits"][split]["confusion"]
    short = [name[:8] for name in LABEL_NAMES]
    lines = [
        "| True ↓ / predicted → | " + " | ".join(short) + " | missed |",
        "|---|" + "---:|" * (len(LABEL_NAMES) + 1),
    ]
    for label in LABEL_NAMES:
        row = matrix[label]
        cells = [f"**{row[p]}**" if p == label else str(row[p] or "·") for p in LABEL_NAMES]
        lines.append(f"| `{label}` | " + " | ".join(cells) + f" | {row['missed']} |")
    return lines


def fmt_diff(d: dict) -> str:
    low, high = d["ci95"]
    return f"{d['diff']:+.2f} ({low:+.2f} to {high:+.2f})"


def difference_table(results: list[dict], gold: pd.DataFrame) -> list[str]:
    """Each model against the baseline on the same reviews, with a paired interval."""
    reference = next((r for r in results if r["model"] == REFERENCE_MODEL), None)
    others = [r for r in results if r["model"] != REFERENCE_MODEL]
    if reference is None or not others:
        return []
    lines = [
        "",
        f"### Difference to {REFERENCE_MODEL}",
        "",
        "Macro-F1 minus the baseline's, on the same reviews. The 95% interval resamples",
        "reviews once for both models, so it is narrower than comparing the two intervals",
        "above. An interval that excludes 0 means the difference holds across resamples.",
        "",
        "| Model | Val Δ macro-F1 (95% CI) | Test Δ macro-F1 (95% CI) |",
        "|---|---:|---:|",
    ]
    for r in others:
        cells = []
        for split in ("val", "test"):
            try:
                cells.append(fmt_diff(paired_difference(r, reference, split, gold)))
            except (ValueError, KeyError) as err:
                cells.append(f"n/a ({err})")
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    return lines


def render(
    results: list[dict], human: dict, fingerprint: str, gold: pd.DataFrame | None = None
) -> str:
    lines = [
        "# Eval table",
        "",
        "Generated by `python -m support_reply_gap.eval`. Do not edit by hand.",
        "",
        "Scores are read against the human ceiling, not against 1.0. The ceiling is",
        "measured on a random 50 of the gold set, the models on the validation and test",
        "splits, so the comparison is indicative. Validation scores drive decisions;",
        "test scores are reported and never used to choose anything. The 95% interval",
        "comes from resampling the split's reviews 1,000 times.",
        "",
        *summary_table(results, human, fingerprint),
        *(difference_table(results, gold) if gold is not None else []),
    ]
    for r in results:
        lines += [
            "",
            f"## {r['model']}",
            "",
            f"Settings: `{json.dumps(r['config'])}`",
            "",
            f"*Italic*: fewer than {FEW_EXAMPLES} examples in that split, not interpretable.",
            "",
            *per_class_table(r),
        ]
        if "val" in r["splits"]:
            lines += [
                "",
                "Confusion on validation. A cell counts reviews with that true and that",
                "predicted label; the diagonal (bold) is correct hits, `missed` counts true",
                "labels not predicted at all. Test confusion is kept in the result file only,",
                "so test errors do not steer changes.",
                "",
                *confusion_table(r),
            ]
    return "\n".join(lines) + "\n"


def write(results_dir: Path = RESULTS_DIR, path: Path = EVAL_TABLE_PATH) -> Path:
    results = latest_per_model(load_results(results_dir))
    path.write_text(render(results, ceiling(), gold_fingerprint(), gold_labels()))
    return path
