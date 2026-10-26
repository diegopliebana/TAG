"""Compatibility import; implementation lives in tag_rust.episodes.gym."""
import sys
from .episodes import gym as _implementation
sys.modules[__name__] = _implementation
