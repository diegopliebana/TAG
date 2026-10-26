"""Compatibility import; implementation lives in tag_rust.learning.seeds."""
import sys
from .learning import seeds as _implementation
sys.modules[__name__] = _implementation
