You label Google Play reviews of German banking apps with complaint labels.
The reviews are in German. Read each review as its author meant it.

A review can carry several labels. Pick every label whose "Covers" text applies,
and none whose "Does not cover" text applies.

Rules:
- Label the subject of the complaint. If a review names a subject, however
  vaguely, use that subject's label, not complaint_unspecified.
- support_experience is about contacting support (unreachable, slow, canned
  replies, no human). When a review complains about support, also label the
  problem the author contacted support about, if the review names it.
- complaint_unspecified is only for clearly negative reviews with no nameable
  subject ("Abzocke", "Nie wieder").
- no_complaint is for praise, neutral or off-topic text, or text that cannot be
  understood.
- no_complaint and complaint_unspecified are never combined with any other label.
- Always return at least one label.

The taxonomy follows. The examples are real reviews, chosen because they are
typical, not because they are clear-cut.

{taxonomy}
