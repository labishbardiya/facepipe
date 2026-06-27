"""
FastAPI application entrypoint for Uvicorn.
"""

from __future__ import annotations

import uvicorn


def start() -> None:
    """Start the uvicorn server."""
    uvicorn.run("entrypoints.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
