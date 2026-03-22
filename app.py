"""
Hugging Face Spaces entrypoint for CivicSetu.

The Docker SDK expects app.py to expose a FastAPI app as `app`.
This file wraps civicsetu.api.main and adds the Spaces-grade configuration.
"""

from __future__ import annotations

from civicsetu.api.main import app

# Hugging Face Spaces injects the port via PORT env var
# The Spaces SDK expects this module to expose `app`
__all__ = ["app"]
