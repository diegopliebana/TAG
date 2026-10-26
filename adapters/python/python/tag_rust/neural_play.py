"""Compatibility import; implementation lives in tag_rust.learning.play."""
import sys
from .learning import play as _implementation
sys.modules[__name__] = _implementation
