"""Terminal outcome conventions used by learning reports and value targets.

Rust reports win/draw/loss as codes 1/2/3. Learning targets use rewards
+1/0/-1 independently for each player; no zero-sum assumption is required.
"""
import numpy as np

RESULT_LABELS = {1: "wins", 2: "draws", 3: "losses"}
REWARD_LABELS = {1: "wins", 0: "draws", -1: "losses"}
RESULT_REWARDS = {1: 1., 2: 0., 3: -1.}


def outcome_counts(*, counter="episodes", truncated=False):
    """Create independent counters, retaining each report's existing field names."""
    counts = {counter: 0, "wins": 0, "draws": 0, "losses": 0}
    if truncated:
        counts["truncated"] = 0
    return counts


def seat_outcome_counts(outcomes, player_count):
    """Count completed games per seat and combine all player appearances."""
    per_seat = [outcome_counts(counter="games") for _ in range(player_count)]
    for codes in outcomes:
        if len(codes) != player_count or any(code not in RESULT_LABELS for code in codes):
            raise ValueError("seat summaries require complete player outcomes")
        for seat,code in enumerate(codes):
            per_seat[seat]["games"] += 1
            per_seat[seat][RESULT_LABELS[code]] += 1
    return {"per_seat":per_seat,
            "combined":{key:sum(row[key] for row in per_seat) for key in per_seat[0]}}


def terminal_rewards(codes):
    """Convert a complete player-outcome vector to float32 training targets."""
    if not codes or any(code not in RESULT_REWARDS for code in codes):
        raise ValueError("completed self-play game requires every player's terminal outcome")
    return np.array([RESULT_REWARDS[int(code)] for code in codes], dtype=np.float32)
