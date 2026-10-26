"""Compatibility import; implementation lives in tag_rust.episodes.batch."""
import sys
from .episodes import batch as _implementation
sys.modules[__name__] = _implementation
