"""Project-wide constants.

The analysis window is an empirical finding, not a convention — see the
quarterly reply-rate table in the README.
"""

from datetime import date

# N26 started replying regularly in 2025Q2 and C24 in 2026Q1. This is the
# earliest date from which all three apps are actively replying, so pooling
# across apps does not mix a reply regime with a no-reply regime.
ANALYSIS_WINDOW_START = date(2026, 4, 1)

# The full ingested window. Used for exploring the policy changes themselves,
# which are only visible from outside the analysis window.
FULL_WINDOW_START = date(2024, 11, 1)