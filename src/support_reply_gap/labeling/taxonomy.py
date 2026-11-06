"""The complaint label taxonomy — the single source of truth.

docs/taxonomy.md is generated from this file (`python -m support_reply_gap.labeling`),
so the definitions cannot drift apart from the code that uses them.

Multi-label: a review can carry several labels. `no_complaint` and
`complaint_unspecified` are exclusive. Multi-label is settled and changing it
would mean relabelling the gold set.

Frozen for the B5 gold set: the labelling sheet's columns are checked against
LABEL_NAMES, so changing a label here means relabelling the gold set.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Label:
    name: str
    title: str
    covers: str
    excludes: str
    # Real review text, filled in during B4. Two or three per label, chosen
    # because they are typical rather than because they are clear-cut.
    examples: tuple[str, ...] = field(default_factory=tuple)


LABELS: tuple[Label, ...] = (
    Label(
        name="crash_stability",
        title="Crashes and stability",
        covers="Crashes, freezes, app will not start, white screen, bugs that block use.",
        excludes=(
            "Slow but working, which is performance. Missing features, which is feature_gap."
        ),
        examples=(
            "seit Update ruckelt die App und hängt fast jedes Mal beim Start",
            (
                "Leider schreibt die App immer das ich die neueste Version brauche obwohl ich "
                "die App schon 4 mal neu Installiert habe."
            ),
            "nee jetzt kann ich nichts mehr lesen weder Kontostand noch sonst wass",
        ),
    ),
    Label(
        name="performance",
        title="Performance",
        covers=(
            "Slow but working: long load times, lag, sluggishness, the app getting "
            "slower over time."
        ),
        excludes=(
            "Freezes, hangs or anything that blocks use, which is crash_stability. "
            "Payments that take long to arrive, which is payments_transfers."
        ),
        examples=(
            (
                "ganz gut, aber App wird über Dauer hinweg gefühlt immer langsamer und lädt "
                "eben ewig."
            ),
            (
                "Seit kurzem nervt die App extrem. Jedes laden dauert minuten. Ein kleiner "
                "Popup zur bestätigung einer kartennutzung dauert mehrere minuten."
            ),
            "die App ist quälend langsam geworden.",
        ),
    ),
    Label(
        name="login_access",
        title="Login and access",
        covers="Cannot log in, 2FA/TAN problems, locked out, password reset loops.",
        excludes=(
            "First-time account opening, which is kyc_onboarding. The bank restricting "
            "or closing the account, which is account_restricted."
        ),
        examples=(
            (
                'Danach kein reinkommen mehr in die App. Passcode wird als "falsch" '
                "angezeigt, andere Verifizierung funktioniert nicht."
            ),
            (
                "werde allerdings ziemlich oft abgemeldet und dann steht dort das sich ein "
                "anderes Smartphone angemeldet."
            ),
            (
                "Der link ist aber abgelaufen.. Ein neuer wird nicht versendet... ich bin "
                "ERNEUT ausgesperrt.."
            ),
        ),
    ),
    Label(
        name="kyc_onboarding",
        title="Verification and onboarding",
        covers=(
            "Identity verification, document upload, video ident, account opening "
            "rejected or stuck."
        ),
        excludes=(
            "Being locked out of an existing account, which is login_access. "
            "Re-identification demanded to lift a restriction on an existing account, "
            "which is account_restricted."
        ),
        examples=(
            (
                "Mann muss die Verifizierung sehr oft machen ich bin jetzt schon beim 4ten "
                "mal und trotzdem nicht verifiziert"
            ),
            (
                "die mit Abstand beste Bank die meinen Antrag auf ein Konto ablehnt. ganz "
                "großes Kino"
            ),
            (
                'immer wieder kommt die Anzeige hoch "wir konnten ihr Ausweisdokument nicht '
                'lesen bitte versuchen sie es nochmals"'
            ),
        ),
    ),
    Label(
        name="account_restricted",
        title="Account restricted by the bank",
        covers=(
            "Bank-initiated restriction of an existing account: closed or terminated, "
            "frozen or blocked, funds unreachable, re-identification forced to regain "
            "access."
        ),
        excludes=(
            "The user failing to log in to a working account, which is login_access. "
            "Opening an account, which is kyc_onboarding. A single payment or card "
            "blocked while the account works, which is payments_transfers."
        ),
        examples=(
            ("Konto wird ohne Ankündigung eingeschränkt. Man kommt an sein Geld nicht mehr ran!!"),
            ("Konto wurde seitens revolut ohne Angabe von gründen gekündigt und geschlossen."),
            (
                "Die haben ohne jeglichen Grund mein Konto gesperrt aufgrund tatverdachts auf "
                "dubiosen Überweisungen und das seit Einer Woche!"
            ),
        ),
    ),
    Label(
        name="payments_transfers",
        title="Payments and transfers",
        covers=(
            "Transfers failing or delayed, deposits not arriving, card declined or "
            "blocked, blocked payments, trade orders not executed."
        ),
        excludes="Complaints about the cost of a transfer, which is fees_pricing.",
        examples=(
            "Neue Karte bestellt, erhalten- nun gehen beide Karten nicht mehr. 🫣",
            "die Überweisungen sind jetzt erst mal für drei Stunden blockiert",
            (
                "wenn ich jetzt von meinem Stromanbieter aber eine Rückzahlung erhalte, an "
                "den ich jeden einzelnen Monat überweise, dann erhalte ich das Geld nicht, "
                'weil "Informationen" fehlen.'
            ),
        ),
    ),
    Label(
        name="fees_pricing",
        title="Fees, pricing and money owed",
        covers=(
            "Fees, exchange spreads, subscription cost, unexpected charges, interest "
            "rates, and money the user believes they are owed: rewards, bonuses, "
            "referral payouts, interest not passed on."
        ),
        excludes="A transfer failing for a non-cost reason, which is payments_transfers.",
        examples=(
            ("leider werden nicht mehr die aktuellen Zinsen an Bestandskunden weitergegeben."),
            (
                "Wenn Konto nicht gedeckt ist wird bei Lastschrift eine Gebühr verlangt wenn "
                "diese fehlgeschlagen sind"
            ),
            "alle Bedingungen sind erfüllt!! ihr bezahlt aber die Premie NICHT!",
        ),
    ),
    Label(
        name="support_experience",
        title="Support experience",
        covers=(
            "The experience of contacting support: unreachable, slow, canned replies, "
            "no human. A meta-label that co-occurs often."
        ),
        excludes="The underlying problem they contacted support about — label that too.",
        examples=(
            (
                "jetzt weiß man gar nicht mehr, wo man den findet und dann kommt nur ein "
                "Chatbot. Wir wollen keinen Chatbot, sondern einen echten MENSCHEN."
            ),
            (
                "ich kann den Support nicht kontaktieren... telefon? Natürlich nur Premium "
                "Kunden... chat? DAFÜR MUSST DU DICH EINLOGGEN!!!"
            ),
            "Und auf die Deutsche kannst du vergessen das jimand da ist. Nur English.",
        ),
    ),
    Label(
        name="feature_gap",
        title="Missing features and usability",
        covers=(
            "Missing functionality, feature requests, UI/UX complaints about what exists, "
            "including ads and upselling inside the app."
        ),
        excludes=(
            "Features that exist but are broken, which is crash_stability. Slow but "
            "working, which is performance."
        ),
        examples=(
            "Wann wird Wero in die App integriert?",
            "Wieso kann ich die Bildschirmsperre nicht deaktivieren?",
            ("die abos sollten einfacher zum herabstufen sein. warum so kompliziert machen ?"),
        ),
    ),
    Label(
        name="complaint_unspecified",
        title="Unspecified complaint",
        covers=(
            "Clearly negative, but with no subject that any other label could name: "
            "insults, 'Abzocke', 'never again'. Exclusive — never combined."
        ),
        excludes=(
            "Praise, neutral or off-topic, which is no_complaint. Any complaint with a "
            "nameable subject, however vague — label that subject instead."
        ),
        examples=(
            "VORSICHT Abzocke!!!!...nicht weiter zu empfehlen...!!!!! keine Sterne",
            "sorry aber das ist der letzte Dreck!",
            "Nein danke! Nie wieder!",
        ),
    ),
    Label(
        name="no_complaint",
        title="No complaint",
        covers="Praise, neutral, off-topic, unintelligible. Exclusive — never combined.",
        excludes="Anything containing a complaint, however mild.",
        examples=(
            "Macht bis jetzt was sie soll.",
            (
                "Tut mir leid ,ich kann die App nicht bewerten da ich diese noch nicht "
                "genutzt habe ."
            ),
            "Interessiert mich nicht",
        ),
    ),
)

LABEL_NAMES: tuple[str, ...] = tuple(label.name for label in LABELS)
# complaint_unspecified is exclusive by definition: a nameable subject means another label.
EXCLUSIVE_LABELS: frozenset[str] = frozenset({"no_complaint", "complaint_unspecified"})


def by_name(name: str) -> Label:
    for label in LABELS:
        if label.name == name:
            return label
    raise KeyError(f"unknown label: {name!r}")


def render_markdown() -> str:
    """Render the taxonomy as the committed reference document."""
    lines = [
        "# Complaint taxonomy",
        "",
        "Generated from `src/support_reply_gap/labeling/taxonomy.py`.",
        "Run `python -m support_reply_gap.labeling` to regenerate.",
        "",
        "Multi-label: a review can carry several labels. `no_complaint` and",
        "`complaint_unspecified` are exclusive.",
        "",
    ]
    for label in LABELS:
        lines += [
            f"## `{label.name}` — {label.title}",
            "",
            f"**Covers.** {label.covers}",
            "",
            f"**Does not cover.** {label.excludes}",
            "",
        ]
        if label.examples:
            lines.append("**Examples.**")
            lines.append("")
            # Blank line after each quote, or Markdown merges them into one paragraph.
            for example in label.examples:
                lines += [f"> {example}", ""]
        else:
            lines += ["_Examples to be filled in during stage B4._", ""]
    return "\n".join(lines)
