# Complaint taxonomy

Generated from `src/support_reply_gap/labeling/taxonomy.py`.
Run `python -m support_reply_gap.labeling` to regenerate.

Multi-label: a review can carry several labels. `no_complaint` and
`complaint_unspecified` are exclusive.

## `crash_stability` — Crashes and stability

**Covers.** Crashes, freezes, app will not start, white screen, bugs that block use.

**Does not cover.** Slow but working, which is performance. Missing features, which is feature_gap.

**Examples.**

> seit Update ruckelt die App und hängt fast jedes Mal beim Start

> Leider schreibt die App immer das ich die neueste Version brauche obwohl ich die App schon 4 mal neu Installiert habe.

> nee jetzt kann ich nichts mehr lesen weder Kontostand noch sonst wass

## `performance` — Performance

**Covers.** Slow but working: long load times, lag, sluggishness, the app getting slower over time.

**Does not cover.** Freezes, hangs or anything that blocks use, which is crash_stability. Payments that take long to arrive, which is payments_transfers.

**Examples.**

> ganz gut, aber App wird über Dauer hinweg gefühlt immer langsamer und lädt eben ewig.

> Seit kurzem nervt die App extrem. Jedes laden dauert minuten. Ein kleiner Popup zur bestätigung einer kartennutzung dauert mehrere minuten.

> die App ist quälend langsam geworden.

## `login_access` — Login and access

**Covers.** Cannot log in, 2FA/TAN problems, locked out, password reset loops.

**Does not cover.** First-time account opening, which is kyc_onboarding. The bank restricting or closing the account, which is account_restricted.

**Examples.**

> Danach kein reinkommen mehr in die App. Passcode wird als "falsch" angezeigt, andere Verifizierung funktioniert nicht.

> werde allerdings ziemlich oft abgemeldet und dann steht dort das sich ein anderes Smartphone angemeldet.

> Der link ist aber abgelaufen.. Ein neuer wird nicht versendet... ich bin ERNEUT ausgesperrt..

## `kyc_onboarding` — Verification and onboarding

**Covers.** Identity verification, document upload, video ident, account opening rejected or stuck.

**Does not cover.** Being locked out of an existing account, which is login_access. Re-identification demanded to lift a restriction on an existing account, which is account_restricted.

**Examples.**

> Mann muss die Verifizierung sehr oft machen ich bin jetzt schon beim 4ten mal und trotzdem nicht verifiziert

> die mit Abstand beste Bank die meinen Antrag auf ein Konto ablehnt. ganz großes Kino

> immer wieder kommt die Anzeige hoch "wir konnten ihr Ausweisdokument nicht lesen bitte versuchen sie es nochmals"

## `account_restricted` — Account restricted by the bank

**Covers.** Bank-initiated restriction of an existing account: closed or terminated, frozen or blocked, funds unreachable, re-identification forced to regain access.

**Does not cover.** The user failing to log in to a working account, which is login_access. Opening an account, which is kyc_onboarding. A single payment or card blocked while the account works, which is payments_transfers.

**Examples.**

> Konto wird ohne Ankündigung eingeschränkt. Man kommt an sein Geld nicht mehr ran!!

> Konto wurde seitens revolut ohne Angabe von gründen gekündigt und geschlossen.

> Die haben ohne jeglichen Grund mein Konto gesperrt aufgrund tatverdachts auf dubiosen Überweisungen und das seit Einer Woche!

## `payments_transfers` — Payments and transfers

**Covers.** Transfers failing or delayed, deposits not arriving, card declined or blocked, blocked payments, trade orders not executed.

**Does not cover.** Complaints about the cost of a transfer, which is fees_pricing.

**Examples.**

> Neue Karte bestellt, erhalten- nun gehen beide Karten nicht mehr. 🫣

> die Überweisungen sind jetzt erst mal für drei Stunden blockiert

> wenn ich jetzt von meinem Stromanbieter aber eine Rückzahlung erhalte, an den ich jeden einzelnen Monat überweise, dann erhalte ich das Geld nicht, weil "Informationen" fehlen.

## `fees_pricing` — Fees, pricing and money owed

**Covers.** Fees, exchange spreads, subscription cost, unexpected charges, interest rates, and money the user believes they are owed: rewards, bonuses, referral payouts, interest not passed on.

**Does not cover.** A transfer failing for a non-cost reason, which is payments_transfers.

**Examples.**

> leider werden nicht mehr die aktuellen Zinsen an Bestandskunden weitergegeben.

> Wenn Konto nicht gedeckt ist wird bei Lastschrift eine Gebühr verlangt wenn diese fehlgeschlagen sind

> alle Bedingungen sind erfüllt!! ihr bezahlt aber die Premie NICHT!

## `support_experience` — Support experience

**Covers.** The experience of contacting support: unreachable, slow, canned replies, no human. A meta-label that co-occurs often.

**Does not cover.** The underlying problem they contacted support about — label that too.

**Examples.**

> jetzt weiß man gar nicht mehr, wo man den findet und dann kommt nur ein Chatbot. Wir wollen keinen Chatbot, sondern einen echten MENSCHEN.

> ich kann den Support nicht kontaktieren... telefon? Natürlich nur Premium Kunden... chat? DAFÜR MUSST DU DICH EINLOGGEN!!!

> Und auf die Deutsche kannst du vergessen das jimand da ist. Nur English.

## `feature_gap` — Missing features and usability

**Covers.** Missing functionality, feature requests, UI/UX complaints about what exists, including ads and upselling inside the app.

**Does not cover.** Features that exist but are broken, which is crash_stability. Slow but working, which is performance.

**Examples.**

> Wann wird Wero in die App integriert?

> Wieso kann ich die Bildschirmsperre nicht deaktivieren?

> die abos sollten einfacher zum herabstufen sein. warum so kompliziert machen ?

## `complaint_unspecified` — Unspecified complaint

**Covers.** Clearly negative, but with no subject that any other label could name: insults, 'Abzocke', 'never again'. Exclusive — never combined.

**Does not cover.** Praise, neutral or off-topic, which is no_complaint. Any complaint with a nameable subject, however vague — label that subject instead.

**Examples.**

> VORSICHT Abzocke!!!!...nicht weiter zu empfehlen...!!!!! keine Sterne

> sorry aber das ist der letzte Dreck!

> Nein danke! Nie wieder!

## `no_complaint` — No complaint

**Covers.** Praise, neutral, off-topic, unintelligible. Exclusive — never combined.

**Does not cover.** Anything containing a complaint, however mild.

**Examples.**

> Macht bis jetzt was sie soll.

> Tut mir leid ,ich kann die App nicht bewerten da ich diese noch nicht genutzt habe .

> Interessiert mich nicht
