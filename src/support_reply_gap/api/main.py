"""Minimal API surface.

This is the walking skeleton: one endpoint, no models, no database. It exists so
that deployment is solved as its own problem, before there is anything
complicated to deploy. The /classify endpoint replaces this stub in D2.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Support Reply Gap",
    description="Complaint classification for German fintech app reviews.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Hosting platforms poll this to decide if the container is up."""
    return {"status": "ok"}