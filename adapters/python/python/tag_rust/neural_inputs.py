"""Compatibility import; implementation lives in tag_rust.learning.inputs."""
import sys
from .learning import inputs as _implementation
sys.modules[__name__] = _implementation
