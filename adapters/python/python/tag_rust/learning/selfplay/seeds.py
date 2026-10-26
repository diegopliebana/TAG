"""Separate deterministic streams for each logical self-play operation."""
import hashlib
import json
import numpy as np


def derive_seed(master, phase, game=0, ply=0, purpose="search", *, version=1):
    if version not in (1, 2):
        raise ValueError("unsupported self-play seed derivation version")
    name = "tag-m8-selfplay-seed" if version == 1 else "tag-mcts-stochastic-selfplay-seed"
    document = [name,version,master,phase,game,ply,purpose]
    encoded = json.dumps(document,separators=(",",":"),ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8],"little")


def seed_version(settings):
    """Placement games retain version 1; stochastic games share version 2."""
    return 2 if settings.game in ("cantstop", "loveletter") else 1


def workflow_seed(settings, phase, game=0, ply=0, purpose="search"):
    """Derive a stream from logical game/action IDs, independently of scheduling."""
    if settings.seed is None:
        raise ValueError("resolve the self-play master seed before deriving streams")
    return derive_seed(settings.seed,phase,game,ply,purpose,version=seed_version(settings))


def game_parameters(settings, game_id, *, phase="selfplay"):
    """Copy rule parameters and supply a recorded independent real-game seed."""
    parameters = dict(settings.parameters)
    if seed_version(settings) == 2:
        parameters["seed"] = workflow_seed(settings,phase,game_id,purpose="game-chance")
    if settings.game == "loveletter":
        # Self-play rotates starters directly. Evaluation rotates the starter
        # independently of candidate seats, covering every pairing in N*N games.
        # An explicit parameter selects a fixed starter in either workflow.
        starter = game_id // settings.players if phase == "evaluation" else game_id
        parameters.setdefault("first_player", starter % settings.players)
    return parameters


def generator(seed):
    return np.random.Generator(np.random.PCG64(seed))
