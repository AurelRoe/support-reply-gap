-- Schema for scraped Google Play reviews.
--
-- Applied with CREATE ... IF NOT EXISTS throughout so it is safe to run on every
-- ingest: the pipeline must be re-runnable without a manual setup step.

create table if not exists apps (
    app_key    text primary key,
    package_id text not null unique,
    label      text not null,
    category   text not null
);

create table if not exists reviews (
    -- Google's own review id. Using it as the primary key is what makes
    -- re-running the scraper idempotent instead of duplicating rows.
    review_id     text primary key,

    app_key       text not null references apps (app_key),

    score         smallint not null check (score between 1 and 5),
    content       text,
    posted_at     timestamptz not null,

    -- Null means no reply at the time of scraping. A reply can appear later,
    -- which is why the loader updates these columns on conflict.
    reply_content text,
    replied_at    timestamptz,

    -- Kept for the exploration stage: thumbs_up helps spot pile-on reviews,
    -- app_version helps spot crashes tied to one release.
    thumbs_up     integer,
    app_version   text,

    -- When this row was last written. Lets you tell a review that genuinely has
    -- no reply from one that was scraped before the reply arrived.
    scraped_at    timestamptz not null default now()
);

-- The analysis always filters by app and date range, and ranks by recency.
create index if not exists reviews_app_posted_idx on reviews (app_key, posted_at desc);

-- Reply rates are computed per app per score band.
create index if not exists reviews_app_score_idx on reviews (app_key, score);