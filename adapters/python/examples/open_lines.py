"""Educational Python equivalent of the native open-line heuristic."""
import numpy as np


def evaluate_open_lines(request, lines):
    """Score each candidate for its acting player, using Rust's open_line_score formula.

    lines is a two-dimensional array of cell indices: one row per possible
    winning line. Its order comes from game.spec().heuristic and matches Rust.
    """
    observers = getattr(request, "observers", None)
    values = np.empty(request.action_ids.size, dtype=np.float64)
    for row, cells in enumerate(request.payloads["cells"]):
        # Results use absolute player seats; cells already use the acting player's
        # viewpoint (+1 own piece, -1 opponent, 0 empty).
        actor = request.actor if observers is None else int(observers[row])
        result = int(request.results[row, actor])
        if result != 0:  # Finished: Win=1, Draw=2, Loss=3.
            values[row] = {1: 1.0, 2: 0.0, 3: -1.0}[result]
            continue

        total = 0.0
        for line in lines:
            friendly = 0
            opposing = 0
            for index in line:
                if cells[index] == 1:
                    friendly += 1
                elif cells[index] == -1:
                    opposing += 1
            # Both players in one line block it. Otherwise its contribution is
            # the squared fraction occupied, positive for us, negative for them.
            if len(line) == 0 or (friendly > 0 and opposing > 0):
                contribution = 0.0
            elif friendly > 0:
                fraction = friendly / len(line)
                contribution = fraction * fraction
            else:
                fraction = opposing / len(line)
                contribution = -(fraction * fraction)
            # Keep the same line order and multiply/divide order as Rust so
            # floating-point rounding agrees in the comparison tests.
            total += contribution
        values[row] = 0.9 * total / len(lines) if len(lines) else 0.0
    return values


