"""Compatibility import; implementation lives in tag_rust.learning.network."""
import sys
from .learning import network as _implementation
sys.modules[__name__] = _implementation
