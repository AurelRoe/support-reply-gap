"""Tests for the C4 eval gate. Throwaway git repos and hand-made results; no database."""

import subprocess

import pandas as pd
import pytest

from support_reply_gap.eval.gate import check_model, is_current
from support_reply_gap.eval.metrics import macro_f1
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

# ---- freshness: is the result newer than the code it depends on? ----------------


@pytest.fixture
def repo(tmp_path):
    def run(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    def commit(path: str, text: str) -> str:
        (tmp_path / path).write_text(text)
        run("add", path)
        run("commit", "-m", f"edit {path}")
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    run("init", "-q")
    return tmp_path, commit


def test_result_after_the_code_change_is_current(repo):
    path, commit = repo
    commit("model.py", "v1")
    result_commit = commit("result.json", "{}")
    assert is_current(path, ("model.py",), result_commit)


def test_code_change_after_the_result_makes_it_stale(repo):
    path, commit = repo
    result_commit = commit("model.py", "v1")
    commit("model.py", "v2")
    assert not is_current(path, ("model.py",), result_commit)


def test_unrelated_change_after_the_result_does_not(repo):
    path, commit = repo
    result_commit = commit("model.py", "v1")
    commit("README.md", "docs")
    assert is_current(path, ("model.py",), result_commit)


# ---- the numbers: genuine, and above the floor ------------------------------------

GOLD_SETS = [{name} for name in LABEL_NAMES] * 2
IDS = [f"r{i:02d}" for i in range(len(GOLD_SETS))]


def gold() -> pd.DataFrame:
    return pd.DataFrame(
        [{"review_id": i, **{n: int(n in s) for n in LABEL_NAMES}} for i, s in zip(IDS, GOLD_SETS)]
    )


def result(pred_sets: list[set[str]], stored_f1: float | None = None, commit: str = "HEAD") -> dict:
    preds = {rid: [n for n in LABEL_NAMES if n in s] for rid, s in zip(IDS, pred_sets)}
    truth = pd.DataFrame([{n: int(n in s) for n in LABEL_NAMES} for s in GOLD_SETS])
    actual = macro_f1(
        truth, pd.DataFrame([{n: int(n in s) for n in LABEL_NAMES} for s in pred_sets])
    )
    return {
        "model": "baseline",
        "commit": commit,
        "gold_fingerprint": "abc",
        "splits": {
            "val": {"predictions": preds, "macro_f1": actual if stored_f1 is None else stored_f1}
        },
    }


def problems(r: dict | None) -> list[str]:
    # Results default to commit "HEAD", which is always current in this repo, so these
    # tests isolate the number checks from the freshness check above.
    from support_reply_gap.labeling.gold import ROOT

    return check_model("baseline", r, gold(), "abc", repo=ROOT)


def mostly_wrong() -> list[set[str]]:
    return [{"feature_gap"}] * len(GOLD_SETS)


def test_good_result_passes():
    assert problems(result(GOLD_SETS)) == []


def test_result_below_the_floor_fails():
    found = problems(result(mostly_wrong()))
    assert any("below the floor" in p for p in found)


def test_stored_score_that_does_not_match_the_predictions_fails():
    found = problems(result(mostly_wrong(), stored_f1=0.9))
    assert any("does not match" in p for p in found)


def test_missing_result_fails():
    assert "no result" in problems(None)[0]


def test_dirty_result_fails():
    found = problems(result(GOLD_SETS, commit="abc1234-dirty"))
    assert any("uncommitted" in p for p in found)


def test_result_on_another_gold_set_fails():
    r = result(GOLD_SETS)
    r["gold_fingerprint"] = "old"
    assert any("different gold set" in p for p in problems(r))
