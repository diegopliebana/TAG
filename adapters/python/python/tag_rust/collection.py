"""Compatibility import; implementation lives in tag_rust.episodes.collection."""
import sys
from .episodes import collection as _implementation
sys.modules[__name__] = _implementation
