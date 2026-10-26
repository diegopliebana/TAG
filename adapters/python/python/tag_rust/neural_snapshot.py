"""Compatibility import; implementation lives in tag_rust.learning.snapshot."""
import sys
from .learning import snapshot as _implementation
sys.modules[__name__] = _implementation
