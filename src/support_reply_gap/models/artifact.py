"""The router's baseline as a file: trained once locally, loaded by the API.

    python -m support_reply_gap.models.artifact   # needs Postgres: trains, saves, checks

Training needs the gold set's review text, which exists only in the local
database. The deployed API has no database, so it never trains: it loads the
file this module writes. The file holds the fitted TF-IDF vocabulary (single
words and character fragments) and the model weights, with no full review, no
author and no labels.

Same training data and settings as the C6 router (the 180 train reviews), so the
router row in reports/eval.md describes what the API serves.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import sklearn

from support_reply_gap.labeling.gold import ROOT
from support_reply_gap.models.baseline import CHOSEN, fit_chosen, predict_proba
from support_reply_gap.models.data import load_gold, xy
from support_reply_gap.models.router import ROUTER_MARGIN

ARTIFACT_PATH = ROOT / "artifacts/router_baseline.joblib"


def save(model, path: Path, trained_on: int) -> None:
    """Write the fitted model plus the settings and version it needs to be used as measured."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "trained_on": trained_on,
        "threshold": CHOSEN["threshold"],
        "margin_cut": ROUTER_MARGIN,
        # A pickled scikit-learn model is only guaranteed to load in the same version.
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, path)


def load(path: Path = ARTIFACT_PATH) -> dict:
    """Read the bundle back. Only ever load a file this project wrote: unpickling runs code."""
    bundle = joblib.load(path)
    if bundle["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(
            f"{path.name} was saved with scikit-learn {bundle['sklearn_version']}, "
            f"this environment has {sklearn.__version__}; re-run "
            "`python -m support_reply_gap.models.artifact` or pin the version."
        )
    return bundle


def main() -> None:
    gold = load_gold()
    texts, y = xy(gold, "train")
    model = fit_chosen(texts, y)
    save(model, ARTIFACT_PATH, trained_on=len(texts))

    # The file must behave exactly like the model that was just trained.
    val_texts, _ = xy(gold, "val")
    reloaded = load()["model"]
    same = np.allclose(predict_proba(model, val_texts), predict_proba(reloaded, val_texts))
    if not same:
        raise SystemExit("reloaded model disagrees with the trained one")

    size_kb = ARTIFACT_PATH.stat().st_size / 1024
    print(
        f"wrote {ARTIFACT_PATH.relative_to(ROOT)} ({size_kb:.0f} KB), trained on "
        f"{len(texts)} reviews; reloaded file gives identical predictions on validation"
    )


if __name__ == "__main__":
    main()
