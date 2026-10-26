"""Compatibility import; implementation lives in tag_rust.learning.collection."""
import sys
from .learning import collection as _implementation
sys.modules[__name__] = _implementation
