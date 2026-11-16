# Slim image: full python:3.11 is ~1GB, slim is ~150MB. Smaller = faster deploys.
FROM python:3.11-slim

# Don't write .pyc files; don't buffer stdout (so logs appear immediately in the
# platform's log viewer rather than being held back).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency files first, source second: Docker caches each step, and this way
# editing src/ doesn't invalidate the (slow) dependency install.
COPY pyproject.toml ./
COPY src/ ./src/
# The API loads the trained baseline and the LLM prompt from these two folders.
COPY artifacts/ ./artifacts/
COPY prompts/ ./prompts/

# Editable install, as on the development machine: the code then runs from
# /app/src, so paths it finds relative to its own location (artifacts/, prompts/)
# point into /app. A normal install would move the code elsewhere and break them.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Run as a non-root user. A container breakout is far less useful to an attacker
# if the process inside wasn't root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Shell form so ${PORT} expands: hosting platforms assign a port at runtime and
# expect the app to bind to it. Falls back to 8000 locally.
CMD ["sh", "-c", "uvicorn support_reply_gap.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
