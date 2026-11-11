"""Drawing review samples for reading and labelling.

Implements the cleaning rules decided in stage B3: German or uncertain language
only, no emoji-only or one-word reviews, stratified by star rating and app, and
capped per app-day so one incident cannot dominate a stratum.

Nothing is deleted from the database — these are filters applied at sampling
time, so every rule is reversible.
"""

from __future__ import annotations

import pandas as pd

# Languages worth distinguishing: German plus the ones most widely spoken by
# people living in Germany. A restricted candidate list matters because short
# German praise is otherwise detected as Slovene or Latin.
CANDIDATE_LANGUAGE_NAMES = (
    "GERMAN", "ENGLISH", "TURKISH", "RUSSIAN", "UKRAINIAN", "POLISH", "ARABIC",
    "ROMANIAN", "ITALIAN", "SPANISH", "FRENCH", "CROATIAN", "BOSNIAN", "SERBIAN",
    "BULGARIAN", "PERSIAN", "PORTUGUESE", "DUTCH", "GREEK",
)  # fmt: skip

MIN_LANGUAGE_CONFIDENCE = 0.7

# Default shape of the stage B4 reading sample: weighted toward complaints,
# but covering 4-5 star reviews so the no_complaint boundary is drawn from real
# cases rather than assumed.
DEV_SAMPLE_PER_SCORE: dict[int, int] = {1: 50, 2: 35, 3: 35, 4: 15, 5: 15}

# Stage B5 gold set, 400 reviews. 2-star is the thinnest pool (about 150 eligible
# once the dev sample is removed); 4-5 star are mostly no_complaint, so 60 of them
# are enough to draw that boundary.
GOLD_SAMPLE_PER_SCORE: dict[int, int] = {1: 170, 2: 90, 3: 80, 4: 30, 5: 30}


def add_text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add length and low-information flags used by the eligibility filter."""
    out = df.copy()
    out["n_chars"] = out["content"].str.len()
    out["n_words"] = out["content"].str.split().str.len()
    # [^\W\d_] is any Unicode letter; [a-zA-Z] would miss umlauts and ß.
    has_letter = out["content"].str.contains(r"[^\W\d_]", regex=True, na=False)
    out["no_letters"] = ~has_letter
    out["one_word"] = has_letter & (out["n_words"] == 1)
    return out


def add_language(df: pd.DataFrame) -> pd.DataFrame:
    """Detect language with lingua, marking low-confidence results as uncertain."""
    from lingua import Language, LanguageDetectorBuilder

    languages = [getattr(Language, name) for name in CANDIDATE_LANGUAGE_NAMES]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    results = detector.compute_language_confidence_values_in_parallel(
        df["content"].fillna("").tolist()
    )
    out = df.copy()
    out["lang"] = [r[0].language.name.lower() if r else "unknown" for r in results]
    out["lang_confidence"] = [r[0].value if r else 0.0 for r in results]
    out.loc[out["lang_confidence"] < MIN_LANGUAGE_CONFIDENCE, "lang"] = "uncertain"
    return out


def eligible_for_labelling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the B3 cleaning rules. Expects add_text_flags and add_language first."""
    keep = (
        df["lang"].isin(["german", "uncertain"])
        & ~df["no_letters"]
        & ~df["one_word"]
        & df["content"].notna()
    )
    return df[keep].copy()


def drop_repeated_text(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one review per distinct normalised text.

    Repeated text is mostly short praise, but a duplicate in a reading sample
    costs attention twice and teaches nothing the first copy did not.
    """
    out = df.copy()
    normalised = out["content"].str.lower().str.replace(r"[\W_]+", " ", regex=True).str.strip()
    return out.loc[~normalised.duplicated()].copy()


def stratified_sample(
    df: pd.DataFrame,
    per_score: dict[int, int] | None = None,
    max_per_app_day: int = 2,
    seed: int = 20261105,
) -> pd.DataFrame:
    """Sample by star rating, spread as evenly as possible across apps.

    Even spread across apps rather than proportional: Revolut is 71% of the
    corpus, and a proportional sample would barely show C24's complaints, which
    are the ones most likely to break the taxonomy.
    """
    per_score = per_score or DEV_SAMPLE_PER_SCORE
    work = df.copy()
    work["day"] = work["posted_at"].dt.date

    # Cap per app-day so an incident day cannot fill a stratum. The cap is per star
    # rating too: across all ratings, a day's many 5-star reviews take the slots and
    # starve the thin 2-3 star strata.
    work = (
        work.sample(frac=1.0, random_state=seed)
        .groupby(["app_key", "day", "score"], sort=False)
        .head(max_per_app_day)
    )

    apps = sorted(work["app_key"].unique())
    chunks: list[pd.DataFrame] = []

    for score, total in per_score.items():
        pool = work[work["score"] == score]
        if pool.empty:
            continue
        target_per_app = max(total // len(apps), 1)
        taken: list[pd.DataFrame] = []
        for app in apps:
            app_pool = pool[pool["app_key"] == app]
            taken.append(app_pool.sample(min(target_per_app, len(app_pool)), random_state=seed))
        chunk = pd.concat(taken)

        # Top up from whatever is left if an app could not fill its share.
        shortfall = total - len(chunk)
        if shortfall > 0:
            rest = pool.loc[~pool.index.isin(chunk.index)]
            if len(rest):
                chunk = pd.concat(
                    [chunk, rest.sample(min(shortfall, len(rest)), random_state=seed)]
                )
        chunks.append(chunk)

    return (
        pd.concat(chunks)
        .sample(frac=1.0, random_state=seed)  # shuffle, so reading order is not by star
        .reset_index(drop=True)
    )


def proportional_subset(
    df: pd.DataFrame, by: list[str], keep: int, seed: int = 20261109
) -> pd.DataFrame:
    """Keep exactly `keep` rows at random, preserving each group's share as closely as possible.

    Each group gets keep * share rows, rounded down; the rows left over go to the groups
    with the largest rounding remainders (largest-remainder method).
    """
    sizes = df.groupby(by).size()
    exact = sizes * keep / len(df)
    quota = exact.astype(int)
    leftover = keep - int(quota.sum())
    # Sorting on the remainder, then on the group key, keeps ties deterministic.
    remainders = (exact - quota).rename("r").reset_index()
    order = remainders.sort_values(["r", *by], ascending=[False, *[True] * len(by)]).index
    for idx in order[:leftover]:
        quota.iloc[idx] += 1

    kept = [
        group.sample(int(quota[key]), random_state=seed)
        for key, group in df.groupby(by)
        if quota[key] > 0
    ]
    return pd.concat(kept)
