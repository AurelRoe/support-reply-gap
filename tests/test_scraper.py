from datetime import UTC, datetime

from support_reply_gap.ingest.scraper import ReviewRecord, as_utc, to_record

RAW_REVIEW = {
    "reviewId": "abc123",
    "userName": "Some Person",
    "content": "App stürzt beim Login ab.",
    "score": 1,
    "thumbsUpCount": 4,
    "reviewCreatedVersion": "3.2.1",
    "at": datetime(2026, 5, 1, 12, 0, 0),
    "replyContent": "Bitte kontaktiere unseren Support.",
    "repliedAt": datetime(2026, 5, 2, 9, 30, 0),
}


def test_as_utc_attaches_timezone_to_naive():
    naive = datetime(2026, 5, 1, 12, 0, 0)
    assert as_utc(naive) == datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def test_as_utc_passes_through_none():
    assert as_utc(None) is None


def test_to_record_maps_fields():
    record = to_record(RAW_REVIEW, app_key="n26")
    assert isinstance(record, ReviewRecord)
    assert record.review_id == "abc123"
    assert record.app_key == "n26"
    assert record.score == 1
    assert record.posted_at.tzinfo is not None
    assert record.replied_at is not None


def test_to_record_drops_author():
    record = to_record(RAW_REVIEW, app_key="n26")
    assert "Some Person" not in str(record)


def test_to_record_handles_missing_reply():
    raw = {k: v for k, v in RAW_REVIEW.items() if k not in {"replyContent", "repliedAt"}}
    record = to_record(raw, app_key="n26")
    assert record.reply_content is None
    assert record.replied_at is None
