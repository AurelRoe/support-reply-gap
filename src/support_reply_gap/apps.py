"""The app set under analysis, fixed in stage B1.

Selected by measuring reply rates across three product categories — see
reports/app_probe.csv and the README. All three are German neobanks: complaint
base rates differ between neobanks and brokers, so a mixed set would confound
the comparison.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class App:
    """One app under analysis. Frozen because this set is fixed for the project."""

    key: str  # short identifier, used as the database key
    package_id: str  # Google Play package id
    label: str  # display name
    category: str  # product category


APPS: tuple[App, ...] = (
    App(key="n26", package_id="de.number26.android", label="N26", category="neobank"),
    App(key="revolut", package_id="com.revolut.revolut", label="Revolut", category="neobank"),
    App(key="c24", package_id="de.c24.bankapp", label="C24 Bank", category="neobank"),
)


def by_key(key: str) -> App:
    """Look up one app by its database key."""
    for app in APPS:
        if app.key == key:
            return app
    raise KeyError(f"unknown app key: {key!r}")
