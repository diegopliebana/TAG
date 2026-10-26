"""Compatibility import; implementation lives in tag_rust.search.scoring."""
import sys
from .search import scoring as _implementation
sys.modules[__name__] = _implementation
