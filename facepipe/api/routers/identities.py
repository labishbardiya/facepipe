"""
Identity management API router.

CRUD operations for enrolled identities.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from facepipe.api.dependencies import get_event_store, get_identity_manager, get_pipeline
from facepipe.api.schemas import (
    IdentityListResponse,
    IdentityResponse,
    IdentityUpdateRequest,
)
from facepipe.core.pipeline import RecognitionPipeline
from facepipe.storage.event_store import EventStore, EventType
from facepipe.storage.identity_manager import IdentityManager

router = APIRouter()


@router.get("/identities", response_model=IdentityListResponse)
async def list_identities(
    identity_mgr: IdentityManager = Depends(get_identity_manager),
) -> IdentityListResponse:
    """List all enrolled identities."""
    records = identity_mgr.list_all()
    return IdentityListResponse(
        identities=[
            IdentityResponse(
                identity_id=r.identity_id,
                name=r.name,
                created_at=r.created_at,
                last_seen=r.last_seen,
                embedding_count=r.embedding_count,
                cluster_count=r.cluster_count,
                model_version=r.model_version,
                is_active=r.is_active,
            )
            for r in records
        ],
        total=len(records),
    )


@router.get("/identities/{identity_id}", response_model=IdentityResponse)
async def get_identity(
    identity_id: str,
    identity_mgr: IdentityManager = Depends(get_identity_manager),
) -> IdentityResponse:
    """Get a specific identity by ID."""
    record = identity_mgr.get(identity_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Identity not found.")

    return IdentityResponse(
        identity_id=record.identity_id,
        name=record.name,
        created_at=record.created_at,
        last_seen=record.last_seen,
        embedding_count=record.embedding_count,
        cluster_count=record.cluster_count,
        model_version=record.model_version,
        is_active=record.is_active,
    )


@router.put("/identities/{identity_id}", response_model=IdentityResponse)
async def update_identity(
    identity_id: str,
    request: IdentityUpdateRequest,
    identity_mgr: IdentityManager = Depends(get_identity_manager),
    event_store: EventStore = Depends(get_event_store),
) -> IdentityResponse:
    """Update identity metadata."""
    record = identity_mgr.get(identity_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Identity not found.")

    updated = identity_mgr.update(identity_id, name=request.name)
    if not updated:
        raise HTTPException(status_code=500, detail="Update failed.")

    event_store.append(
        EventType.IDENTITY_UPDATED,
        identity_id=identity_id,
        payload={"name": request.name},
    )

    return await get_identity(identity_id, identity_mgr)


@router.delete("/identities/{identity_id}")
async def delete_identity(
    identity_id: str,
    identity_mgr: IdentityManager = Depends(get_identity_manager),
    event_store: EventStore = Depends(get_event_store),
    pipeline: RecognitionPipeline = Depends(get_pipeline),
) -> dict:
    """Delete an identity (soft delete)."""
    record = identity_mgr.get(identity_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Identity not found.")

    identity_mgr.delete(identity_id, soft=True)

    # Remove from vector store
    pipeline.vector_store.remove([identity_id])
    pipeline.identity_clusters.pop(identity_id, None)

    event_store.append(
        EventType.IDENTITY_DELETED,
        identity_id=identity_id,
        payload={"name": record.name},
    )

    return {"message": f"Identity '{record.name}' deleted.", "identity_id": identity_id}
