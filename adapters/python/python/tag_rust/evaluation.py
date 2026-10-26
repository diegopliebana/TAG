"""Compatibility import; implementation lives in tag_rust.search.evaluation."""
import sys
from .search import evaluation as _implementation
sys.modules[__name__] = _implementation
