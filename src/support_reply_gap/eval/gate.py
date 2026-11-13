"""C4 eval gate: checks the committed eval results, so CI needs no database and no API.

    python -m support_reply_gap.eval --gate

Models are scored locally, where the review text is, and their result files are
committed. CI then checks, per model:

1. a result exists, from a clean commit, against the current gold set;
2. it is current: no file that can change the score was edited after it was made;
3. its numbers are genuine: validation macro-F1 recomputed from the stored
   per-review predictions matches the stored value;
4. it clears the committed floor on validation macro-F1.

And that reports/eval.md matches what the result files would generate.

The gate trusts that the result file came from a real local run. That is the price
of keeping review text off GitHub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from support_reply_gap.eval import report
from support_reply_gap.eval.harness import gold_fingerprint
from support_reply_gap.eval.metrics import macro_f1
from support_reply_gap.labeling.gold import ROOT
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

# Validation, not test: a gate that failed on test would invite tuning until test passed.
# Baseline scores 0.486; the floor leaves room for nothing but a real step back.
VAL_MACRO_F1_FLOOR = {"baseline": 0.45}

# Files whose change can move any model's score. Paths relative to the repo root.
SHARED_INPUTS = (
    "src/support_reply_gap/eval/harness.py",
    "src/support_reply_gap/eval/metrics.py",
    "src/support_reply_gap/labeling/taxonomy.py",
    "data/gold/gold_labels.csv",
    "data/gold/splits.csv",
)
MODEL_INPUTS = {
    "baseline": (
        "src/support_reply_gap/models/baseline.py",
        "src/support_reply_gap/models/data.py",
        "src/support_reply_gap/eval/registry.py",
    ),
}


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    # check=False: a non-zero exit is an answer here (e.g. "not an ancestor"), not an error
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def last_change(repo: Path, paths: tuple[str, ...]) -> str | None:
    """The most recent commit that touched any of the paths."""
    out = git(repo, "log", "-1", "--format=%H", "--", *paths).stdout.strip()
    return out or None


def is_current(repo: Path, paths: tuple[str, ...], result_commit: str) -> bool:
    """True when the result's commit already contains the last change to the paths.

    The result commit must include that change (be it, or come after it in history).
    A squash merge rewrites commit ids, so it would make every result look stale.
    """
    changed = last_change(repo, paths)
    if changed is None:
        return True
    return git(repo, "merge-base", "--is-ancestor", changed, result_commit).returncode == 0


def recomputed_val_f1(result: dict, gold: pd.DataFrame) -> float:
    preds = result["splits"]["val"]["predictions"]
    ids = sorted(preds)
    y_true = gold.set_index("review_id").loc[ids, list(LABEL_NAMES)].reset_index(drop=True)
    y_pred = pd.DataFrame([{n: int(n in preds[rid]) for n in LABEL_NAMES} for rid in ids])
    return macro_f1(y_true, y_pred)


def check_model(
    name: str, result: dict | None, gold: pd.DataFrame, fingerprint: str, repo: Path = ROOT
) -> list[str]:
    """Problems with one model's latest result; an empty list means it passes."""
    if result is None:
        return [
            f"{name}: no result in reports/eval/; run `python -m support_reply_gap.eval --model {name}`"
        ]
    problems = []
    commit = result["commit"]
    if commit == "unknown" or commit.endswith("-dirty"):
        problems.append(
            f"{name}: result made from uncommitted code ({commit}); commit, then re-run"
        )
    elif not is_current(repo, SHARED_INPUTS + MODEL_INPUTS.get(name, ()), commit):
        problems.append(
            f"{name}: code or gold data changed after the result ({commit}) was made; re-run it"
        )
    if result["gold_fingerprint"] != fingerprint:
        problems.append(f"{name}: result was measured against a different gold set; re-run it")
        return problems  # its numbers can't be recomputed against today's labels

    val = result["splits"].get("val", {})
    if "predictions" not in val:
        return [*problems, f"{name}: result has no per-review predictions; re-run it"]
    recomputed = recomputed_val_f1(result, gold)
    if abs(recomputed - val["macro_f1"]) > 1e-9:
        problems.append(
            f"{name}: stored val macro-F1 {val['macro_f1']:.4f} does not match the "
            f"predictions ({recomputed:.4f})"
        )
    floor = VAL_MACRO_F1_FLOOR.get(name)
    if floor is not None and recomputed < floor:
        problems.append(f"{name}: val macro-F1 {recomputed:.3f} is below the floor {floor:.2f}")
    return problems


def run_gate(repo: Path = ROOT) -> list[str]:
    latest = {r["model"]: r for r in report.latest_per_model(report.load_results())}
    gold, fingerprint = report.gold_labels(), gold_fingerprint()
    problems = []
    for name in VAL_MACRO_F1_FLOOR:
        problems += check_model(name, latest.get(name), gold, fingerprint, repo)

    expected = report.render(list(latest.values()), report.ceiling(), fingerprint, gold)
    if not report.EVAL_TABLE_PATH.exists() or report.EVAL_TABLE_PATH.read_text() != expected:
        problems.append(
            "reports/eval.md is out of date; run `python -m support_reply_gap.eval --report-only`"
        )
    return problems
