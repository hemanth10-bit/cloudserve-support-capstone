"""Run: python -m ui   (serves http://127.0.0.1:8000, this machine only)"""

import uvicorn

from .app import create_app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="warning")
