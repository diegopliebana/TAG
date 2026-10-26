"""Compatibility import; implementation lives in tag_rust.search.grouping."""
import sys
from .search import grouping as _implementation
sys.modules[__name__] = _implementation
