"""
DEPRECATED backend entrypoint.

This project standardizes on a single backend entrypoint: `uvicorn src.main:app`.
`backend_api.py` remains as a thin compatibility wrapper so existing commands
like `python backend_api.py` or `uvicorn backend_api:app` keep working.
"""

import os
import uvicorn

# Re-export the canonical FastAPI app
from src.main import app  # noqa: F401


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    reload_default = False if os.name == "nt" else True
    reload = _env_flag("UVICORN_RELOAD", default=reload_default)

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )

