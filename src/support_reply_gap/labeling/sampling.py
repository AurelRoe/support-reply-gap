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
    normalised = (
        out["content"].str.lower().str.replace(r"[\W_]+", " ", regex=True).str.strip()
    )
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

    # Cap per app-day before sampling, so an incident day cannot fill a stratum.
    work = (
        work.sample(frac=1.0, random_state=seed)
        .groupby(["app_key", "day"], sort=False)
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
                chunk = pd.concat([chunk, rest.sample(min(shortfall, len(rest)), random_state=seed)])
        chunks.append(chunk)

    return (
        pd.concat(chunks)
        .sample(frac=1.0, random_state=seed)  # shuffle, so reading order is not by star
        .reset_index(drop=True)
    )