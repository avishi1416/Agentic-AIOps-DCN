"""
HuggingFace Spaces entry point.

This file is the standard entry point for HuggingFace Spaces
running a FastAPI backend (Docker SDK space type).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api.server import app

# HuggingFace Spaces will auto-detect this as the app to serve.
