"""Regenerate docs/taxonomy.md from the taxonomy definitions.

    python -m support_reply_gap.labeling
"""

from pathlib import Path

from support_reply_gap.labeling.taxonomy import render_markdown

OUTPUT = Path("docs/taxonomy.md")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()