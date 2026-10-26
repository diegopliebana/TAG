"""Resolve optional experiment seeds from OS randomness, never from the clock."""
import secrets


def resolve_seed(seed=None):
    """Resolve None once; callers retain/report the integer for reproducibility.

    Explicit nonnegative Python integers are unchanged. Existing workflow-specific
    upper bounds still apply; versioned derived-stream recipes are unchanged.
    """
    if seed is None:
        return secrets.randbits(64)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer or None")
    return seed


def seed_argument(text):
    """Argparse seed: a decimal integer, or auto/null for OS randomness."""
    if text in ("auto", "null"):
        return None
    value = int(text)
    if value < 0:
        raise ValueError("seed must be nonnegative")
    return value
