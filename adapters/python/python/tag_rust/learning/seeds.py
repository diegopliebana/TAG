"""Versioned independent experiment streams, shared by collection and future training."""
import hashlib
import json
from tag_rust.engine.api import _integer


def episode_seed(master_seed, phase, logical_episode_id, purpose):
    """Hash the versioned UTF-8 JSON bytes to an unsigned little-endian 64-bit seed."""
    master_seed = _integer(master_seed, "master_seed")
    logical_episode_id = _integer(logical_episode_id, "logical_episode_id")
    if not isinstance(phase, str) or not phase or not isinstance(purpose, str) or not purpose:
        raise ValueError("phase and purpose must be nonempty strings")
    encoded = json.dumps(["tag-m7-seed", 1, master_seed, phase, logical_episode_id, purpose],
                         separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
