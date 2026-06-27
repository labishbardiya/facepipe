"""
Metrics API router.

GET /api/v1/metrics — Prometheus metrics endpoint
GET /api/v1/events — Query event store
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from facepipe.api.dependencies import get_event_store
from facepipe.api.schemas import EventQueryResponse, EventResponse
from facepipe.storage.event_store import EventStore

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus-format metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/events", response_model=EventQueryResponse)
async def query_events(
    event_type: str | None = None,
    identity_id: str | None = None,
    limit: int = 50,
    event_store: EventStore = Depends(get_event_store),
) -> EventQueryResponse:
    """Query the event store."""
    events = event_store.query(
        event_type=event_type,
        identity_id=identity_id,
        limit=limit,
    )

    return EventQueryResponse(
        events=[
            EventResponse(
                event_id=e["event_id"],
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                identity_id=e.get("identity_id"),
                payload=e.get("payload", "{}"),
            )
            for e in events
        ],
        total=len(events),
    )
