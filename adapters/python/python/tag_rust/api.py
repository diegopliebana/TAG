"""Compatibility import; implementation lives in tag_rust.engine.api."""
import sys
from .engine import api as _implementation
sys.modules[__name__] = _implementation
