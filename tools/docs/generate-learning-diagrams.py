"""Draw the documentation's board/vector/tensor examples using only Python's standard library.

Run from any directory with: python tools/docs/generate-learning-diagrams.py
These are explanatory figures, not game artwork or a runtime image encoder.
"""

from html import escape
from pathlib import Path


DESTINATION = Path(__file__).resolve().parents[2] / "docs" / "images"
SIZE = 3
# One legal two-move position, observed by Player 0 (X).
PIECES = {(1, 2): "X", (2, 0): "O"}
BLUE = "#1d4ed8"
ORANGE = "#9a3412"
TEAL = "#0f766e"
INK = "#172033"
MUTED = "#526178"
BORDER = "#cbd5e1"


def text(x, y, value, size=18, color=INK, anchor="start", weight="normal"):
    """Place escaped text; x/y are the text baseline, in SVG pixels."""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'
    )


def rectangle(x, y, width, height, fill="white", stroke=BORDER, radius=6):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def arrow(x1, y1, x2, y2):
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{MUTED}" '
        'stroke-width="2.5" marker-end="url(#arrow)"/>'
    )


def board(x, y, cell, show_indices=False):
    """Draw the same board with x across columns and y down rows."""
    parts = [text(x + 1.5 * cell, y - 45, "column x", 16, MUTED, "middle")]
    for coordinate in range(SIZE):
        parts.append(text(x + (coordinate + 0.5) * cell, y - 17, coordinate, 16, MUTED, "middle"))
        parts.append(text(x - 17, y + (coordinate + 0.5) * cell + 6, coordinate, 16, MUTED, "middle"))
    parts.append(text(x - 17, y + 3 * cell + 29, "row y", 16, MUTED))
    for row in range(SIZE):
        for column in range(SIZE):
            piece = PIECES.get((column, row), "")
            color = BLUE if piece == "X" else ORANGE
            fill = "#eff6ff" if piece == "X" else "#fff7ed" if piece == "O" else "white"
            left, top = x + column * cell, y + row * cell
            parts.append(rectangle(left, top, cell, cell, fill))
            parts.append(text(left + cell / 2, top + cell * 0.64, piece, 30, color, "middle", "bold"))
            if show_indices:
                parts.append(text(left + cell - 8, top + cell - 8, column * SIZE + row, 13, MUTED, "end"))
    return "".join(parts)


def document(filename, title, description, height, parts):
    """Write a self-contained vector image with an accessible title/description."""
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}"
    viewBox="0 0 1000 {height}" role="img" aria-labelledby="title description">
  <title id="title">{escape(title)}</title>
  <desc id="description">{escape(description)}</desc>
  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
    <path d="M 0 0 L 8 4 L 0 8 Z" fill="{MUTED}"/>
  </marker></defs>
  <rect width="1000" height="{height}" rx="12" fill="#f8fafc"/>
  <g font-family="Segoe UI, Arial, sans-serif">
    {''.join(parts)}
  </g>
</svg>
'''
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / filename).write_text(content, encoding="utf-8")


def vector_diagram():
    values = [1 if PIECES.get((x, y)) == "X" else -1 if PIECES.get((x, y)) == "O" else 0
              for x in range(SIZE) for y in range(SIZE)]
    assert values == [0, 0, 0, 0, 0, 1, -1, 0, 0]
    parts = [
        text(32, 43, "1. Turn the board into a vector", 28, weight="bold"),
        text(32, 77, "Player 0 is observing: X is own, O is opposing. Coordinates start at zero.", 18),
        text(62, 121, "The board", 20, weight="bold"),
        board(65, 190, 64, show_indices=True),
        text(380, 151, "cells", 22, weight="bold"),
        text(955, 151, "shape [9]", 20, MUTED, "end"),
        arrow(286, 275, 350, 275),
        text(380, 199, "index", 15, MUTED),
    ]
    for index, value in enumerate(values):
        x = 380 + index * 64
        fill = "#dbeafe" if value == 1 else "#ffedd5" if value == -1 else "white"
        color = BLUE if value == 1 else ORANGE if value == -1 else INK
        parts.extend([
            text(x + 32, 225, index, 17, MUTED, "middle"),
            rectangle(x, 240, 64, 62, fill),
            text(x + 32, 280, "+1" if value == 1 else value, 24, color, "middle", "bold"),
        ])
    for column in range(SIZE):
        x = 380 + column * 192
        parts.append(f'<path d="M {x + 5} 315 V 325 H {x + 187} V 315" fill="none" stroke="{MUTED}"/>')
        parts.append(text(x + 96, 350, f"column {column}", 17, MUTED, "middle"))
    parts.extend([
        text(380, 388, "Read DOWN column 0, then column 1, then column 2.", 18),
        rectangle(32, 457, 936, 126, "white"),
        text(52, 490, "index = x * height + y", 22, weight="bold"),
        text(52, 526, "X at (1, 2):  index = 1 * 3 + 2 = 5;  cells[5] = +1", 19, BLUE),
        text(52, 558, "O at (2, 0):  index = 2 * 3 + 0 = 6;  cells[6] = -1", 19, ORANGE),
        text(32, 618, "Small board numbers show vector indices. Empty cells become 0.", 18, MUTED),
    ])
    document(
        "learning-board-vector.svg", "Board to x-major cell vector",
        "Player 0 sees X at column 1 row 2 and O at column 2 row 0. Reading down each column gives "
        "[0,0,0,0,0,1,-1,0,0]. X maps to index 5 and O to index 6 using x times height plus y.",
        650, parts,
    )


def tensor_diagram():
    parts = [
        text(32, 43, "2. Represent the same board as three grids", 28, weight="bold"),
        text(32, 77, "Each grid is a channel: 1 marks a matching cell; 0 means no match.", 18),
        text(61, 129, "Same observation", 20, weight="bold"),
        board(65, 195, 57),
        arrow(264, 276, 309, 276),
    ]
    for channel, (label, color, fill) in enumerate([
        ("Own pieces (X)", BLUE, "#dbeafe"),
        ("Opposing pieces (O)", ORANGE, "#ffedd5"),
        ("Empty cells", TEAL, "#ccfbf1"),
    ]):
        x, y, cell = 335 + channel * 214, 195, 57
        parts.append(text(x + 85.5, 125, f"channel {channel}", 17, MUTED, "middle"))
        parts.append(text(x + 85.5, 153, label, 18, color, "middle", "bold"))
        for coordinate in range(SIZE):
            parts.append(text(x + (coordinate + 0.5) * cell, y - 12, coordinate, 15, MUTED, "middle"))
            parts.append(text(x - 12, y + (coordinate + 0.5) * cell + 5, coordinate, 15, MUTED, "middle"))
        for row in range(SIZE):
            for column in range(SIZE):
                piece = PIECES.get((column, row))
                value = int(piece == "X" if channel == 0 else piece == "O" if channel == 1 else piece is None)
                left, top = x + column * cell, y + row * cell
                parts.append(rectangle(left, top, cell, cell, fill if value else "white"))
                parts.append(text(left + cell / 2, top + 38, value, 25, color if value else MUTED, "middle", "bold" if value else "normal"))
        parts.append(text(x + 85.5, 398, "3 rows by 3 columns", 16, MUTED, "middle"))
    parts.extend([
        rectangle(32, 447, 936, 149, "white"),
        text(52, 483, "board shape = [channels, height, width] = [3, 3, 3]", 23, weight="bold"),
        text(52, 523, "Read a value as board[channel, y, x].", 20),
        text(52, 560, "For X at (1, 2):  board[0, 2, 1] = 1;  board[1, 2, 1] = 0;  board[2, 2, 1] = 0.", 19, BLUE),
        text(32, 633, "Channel order: own / opposing / empty. Read across each grid row, then down.", 18, MUTED),
        text(32, 665, "This is a new encoding of the board, not a direct reshape of the cells vector.", 18, MUTED),
    ])
    document(
        "learning-board-tensor.svg", "Board to own, opposing and empty tensor channels",
        "The same observation becomes three 3 by 3 channels. Own has a 1 at row 2 column 1; "
        "opposing has a 1 at row 0 column 2; empty has 1 at every other cell. "
        "The tensor has shape [3,3,3], indexed channel, row y, column x.",
        700, parts,
    )


if __name__ == "__main__":
    vector_diagram()
    tensor_diagram()
    print(f"Wrote two diagrams to {DESTINATION}")
