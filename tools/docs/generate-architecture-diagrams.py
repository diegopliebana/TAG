"""Generate architecture figures as SVG and vector PDF with the standard library.

Run from any directory; --check verifies exports without rewriting them.
The diagrams() function is the shared source for both formats.
"""

import argparse
from dataclasses import dataclass, field
from html import escape
import math
from pathlib import Path

DESTINATION = Path(__file__).resolve().parents[2] / "docs/images/architecture"
INK, MUTED, BORDER = "#172b46", "#526780", "#dce5ef"
BLUE, TEAL, AMBER, PURPLE = "#3669ca", "#148579", "#a46c18", "#7656b5"
PALE_BLUE, PALE_TEAL, PALE_AMBER = "#edf3fd", "#eaf6f3", "#fcf4e6"
BACKGROUND = "#f6f8fc"

# Standard Helvetica character widths, in thousandths of the font size.
WIDTHS = [278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,
          556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,
          1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,
          667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,
          333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,
          556,556,333,500,278,556,500,722,500,500,500,334,260,334,584]


def text_width(label, size, bold=False):
    return sum(WIDTHS[ord(char) - 32] for char in label) * size / 1000 * (1.04 if bold else 1)


@dataclass
class Figure:
    name: str
    title: str
    description: str
    height: int
    shapes: list = field(default_factory=list)
    width: int = 1240

    def text(self, x, y, label, size=18, color=INK, bold=False, anchor="start"):
        self.shapes.append(("text", x, y, label, size, color, bold, anchor))

    def rect(self, x, y, width, height, fill="#ffffff", stroke=BORDER, radius=14):
        self.shapes.append(("rect", x, y, width, height, fill, stroke, radius))

    def line(self, x1, y1, x2, y2, color=BORDER, dashed=False, thickness=2):
        self.shapes.append(("line", x1, y1, x2, y2, color, dashed, thickness))

    def arrow(self, x1, y1, x2, y2, label=None, color=BLUE):
        self.line(x1, y1, x2, y2, color)
        angle = math.atan2(y2 - y1, x2 - x1)
        for offset in (-0.45, 0.45):
            self.line(x2, y2, x2 - 10 * math.cos(angle + offset),
                      y2 - 10 * math.sin(angle + offset), color)
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 12, label, 15, MUTED, anchor="middle")

    def footer(self, label):
        self.line(44, self.height - 61, 1196, self.height - 61)
        self.text(44, self.height - 30, label, 16, MUTED)

    def card(self, x, y, width, title, lines=(), accent=BLUE, height=112):
        self.rect(x, y + 4, width, height, "#e7edf5", None)
        self.rect(x, y, width, height)
        self.rect(x + 16, y + 19, 4, height - 38, accent, None, 2)
        title_size = 21 if text_width(title, 21, True) <= width - 52 else 19
        self.text(x + 33, y + 34, title, title_size, bold=True)
        for row, label in enumerate(lines):
            self.text(x + 33, y + 61 + row * 23, label, 16, MUTED)

    def group(self, x, y, width, height, title, fill=PALE_BLUE, accent=BLUE):
        self.rect(x, y, width, height, fill, None, 20)
        self.text(x + 22, y + 32, title, 13, accent, True)


def sequence(name, title, description, actors, messages, footer):
    """Three participant lanes; arrows show time from top to bottom."""
    figure = Figure(name, title, description, 308 + len(messages) * 83)
    positions = [220, 620, 1020]
    for x, (title, note, accent) in zip(positions, actors):
        figure.rect(x - 170, 144, 340, figure.height - 233, "#eef2f8", None, 18)
        figure.line(x, 238, x, figure.height - 82, BORDER, True)
        figure.card(x - 160, 154, 320, title, [note], accent, 84)
    for row, (source, target, label, note) in enumerate(messages):
        y = 292 + row * 83
        left, right = positions[source], positions[target]
        figure.rect(min(left, right) + 16, y - 32, abs(right - left) - 32, 58,
                    BACKGROUND, None, 10)
        figure.text((left + right) / 2, y - 12, label, 18, INK, True, "middle")
        figure.arrow(left, y + 1, right, y + 1)
        figure.text((left + right) / 2, y + 24, note, 14, MUTED, anchor="middle")
    figure.footer(footer)
    return figure


def diagrams():
    figures = []
    figure = Figure("system", "One engine, three ways to run it",
                    "Native experiments, Python through PyO3, and the browser through Wasm use "
                    "the same Rust engine. The engine groups rules/state, agents/search and "
                    "runners/sessions; optional persistence and inference remain separate from rules.", 760)
    for x, title, lines, adapter, detail, accent in [
        (44, "Native experiments", ["Matches and tournaments", "Command-line programs"], "Rust entry points", "Configuration + reports", BLUE),
        (444, "Python workflows", ["Scripts, training and evaluation", "PyTorch policy/value models"], "Compiled Python adapter", "PyO3 / owned NumPy arrays", PURPLE),
        (844, "Browser interface", ["Human and AI play", "Review, branches and saves"], "Worker + Wasm adapter", "Versioned commands + views", TEAL),
    ]:
        figure.card(x, 149, 352, title, lines, accent, 118)
        figure.arrow(x + 176, 271, x + 176, 307, color=accent)
        figure.card(x, 310, 352, adapter, [detail], accent, 89)
        figure.arrow(x + 176, 403, x + 176, 455, color=accent)
    figure.group(44, 455, 1152, 213, "RUST ENGINE", PALE_TEAL, TEAL)
    for x, title, lines in [
        (66, "Rules + state", ["Observations, actions, chance", "Four games share these contracts"]),
        (442, "Agents + search", ["Random, OSLA and MCTS", "Independent simulated worlds"]),
        (818, "Runners + sessions", ["Apply accepted decisions", "Results, history and branches"]),
    ]:
        figure.card(x, 508, 356, title, lines, TEAL, 116)
    figure.text(66, 650, "Optional engine features: replay files, JSON search configuration, local model inference", 16, MUTED)
    figure.footer("Arrows mean uses. Python and browser adapters share the engine; game rules stay in Rust.")
    figures.append(figure)

    figures.append(sequence(
        "native-decision", "From observation to accepted move",
        "The runner projects permitted input, asks the agent for an index, then applies "
        "the selected action through the forward model. Agents also receive shared rules; OSLA "
        "and MCTS use them on independent simulation states. The caller retains actual state and controllers.",
        [("Agent", "Own memory + choice RNG", PURPLE),
         ("Runner", "Borrows state + controllers", BLUE),
         ("Forward model", "Shared rules operating on states", TEAL)],
        [(1, 2, "List actions + observe", "separate queries on live state"),
         (2, 1, "Observation + choices", "owned player view and action list"),
         (1, 0, "choose(...)", "rules + player observation + choices"),
         (0, 2, "Build / advance simulations", "OSLA / MCTS use independent states"),
         (2, 0, "Simulated successors", "repeat within the search budget"),
         (0, 1, "Selected list index", "runner checks the index"),
         (1, 2, "Apply selected action", "rules resolve effects and chance"),
         (2, 1, "Transition complete", "next player, results or completion")],
        "Agents use shared rules for simulation; the runner updates the authoritative game state."))

    figure = Figure("information", "Keep actual state separate from hypotheses",
                    "Love Letter's rules project a permitted observation from actual state. "
                    "Search samples a consistent possible world using the observation and "
                    "a separate planning RNG; actual hidden cards and game RNG do not enter sampling.", 680)
    for x, title, fill, accent in [(44, "ACTUAL GAME", PALE_AMBER, AMBER),
                                  (444, "PLAYER INPUT", PALE_TEAL, TEAL),
                                  (844, "SEARCH SIMULATION", PALE_BLUE, BLUE)]:
        figure.group(x, 163, 352, 400, title, fill, accent)
    figure.card(62, 222, 316, "Full rules state", ["Hands, deck, turn and effects", "Includes hidden information"], AMBER, 121)
    figure.card(462, 222, 316, "Player observation", ["Own card and public facts", "Permitted retained knowledge"], TEAL, 121)
    figure.card(862, 222, 316, "Sampled world", ["Unknown cards filled consistently", "One hypothesis for simulation"], BLUE, 121)
    figure.arrow(378, 279, 462, 279, "observe")
    figure.arrow(778, 279, 862, 279, "sample")
    figure.card(62, 412, 316, "Game RNG", ["Advances real chance outcomes"], AMBER, 90)
    figure.arrow(220, 412, 220, 347, color=AMBER)
    figure.card(862, 412, 316, "Planning RNG", ["Sampling + simulated future chance"], BLUE, 90)
    figure.arrow(1020, 412, 1020, 347)
    figure.text(620, 418, "Used by agents and encoders", 17, TEAL, True, "middle")
    figure.text(620, 447, "Permitted facts only", 16, MUTED, anchor="middle")
    figure.footer("Separate RNG streams preserve real-game progress while search explores possible continuations.")
    figures.append(figure)

    figures.append(sequence(
        "neural-evaluation", "Neural MCTS: evaluate and resume",
        "Rust retains independent search trees and planning RNGs. Python batches compatible "
        "observation requests, evaluates a frozen PyTorch model, then resumes the searches. "
        "Each tree has at most one pending request; applying the final action is a separate step.",
        [("Rust searches", "Independent trees + planning RNGs", BLUE),
         ("Python coordinator", "MctsBatch / owned request arrays", PURPLE),
         ("PyTorch model", "Frozen policy/value network", TEAL)],
        [(0, 1, "Observation requests", "one pending request per tree"),
         (1, 2, "Evaluate compatible batch", "observations + legal action mapping"),
         (2, 1, "Policy / player values", "explicit requested player perspectives"),
         (1, 0, "Validate replies + resume", "Rust updates tree statistics"),
         (0, 1, "Actions + search reports", "the real game is still unchanged")],
        "Repeat request / evaluation / resume until the budget ends, then apply the selected action separately."))

    figures.append(sequence(
        "browser-move", "A move across the browser boundary",
        "A player sends an action through the page to the worker's Wasm session. Rust "
        "checks its position and applies it. The page displays permitted effects and "
        "conditionally shows Love Letter's action information screen before continuing.",
        [("Player", "Human input + result review", PURPLE),
         ("Page", "Main thread / display and input", BLUE),
         ("Worker + Wasm", "Rules, session and AI controllers", TEAL)],
        [(0, 1, "Choose an action", "from the displayed legal choices"),
         (1, 2, "Versioned command", "Rust checks position and applies move"),
         (2, 1, "View + accepted effects", "projected for the chosen observer"),
         (1, 0, "Action information", "Love Letter screen, when enabled"),
         (0, 1, "Continue presentation", "rules have already resolved the move")],
        "AI path: prepare / poll a proposal, then commit only if current. Reveal mode changes presentation only."))

    figure = Figure("experiments", "Turn a configuration into evidence",
                    "Experiment configuration creates a seat and seed schedule. Games execute "
                    "with independent RNG streams, then reports retain per-seat and combined "
                    "outcomes. Optional action records support later analysis.", 630)
    for x, title, lines, accent in [
        (44, "Configure", ["Game rules + agent settings", "Budgets, seeds and limits"], BLUE),
        (344, "Schedule", ["Rotate or fix player seats", "Per-game RNG streams"], PURPLE),
        (644, "Run games", ["Apply accepted decisions", "Outcomes + truncations"], TEAL),
        (944, "Report", ["Per-seat + combined results", "Chosen ranking convention"], AMBER),
    ]:
        figure.card(x, 181, 252, title, lines, accent, 136)
    for left, right in [(296, 344), (596, 644), (896, 944)]:
        figure.arrow(left, 247, right, 247)
    figure.card(580, 415, 394, "Optional action records", ["Accepted moves, in their recorded order", "Selected with reporting.include_actions"], TEAL, 120)
    figure.arrow(770, 321, 770, 415, "trace")
    figure.text(66, 421, "REPRODUCIBILITY", 13, BLUE, True)
    figure.text(66, 453, "Keep resolved seeds and configuration", 18, bold=True)
    figure.text(66, 480, "alongside model identity and results.", 17, MUTED)
    figure.footer("Keep settings, seat assignments, seeds and outcome definitions with the recorded experiment results.")
    figures.append(figure)

    figure = Figure("artifacts", "Choose the artifact for the job",
                    "A session exports complete rules state or a replay with history and "
                    "controller checkpoints. Training exports a portable inference model or "
                    "a training checkpoint with optimization and RNG progress.", 720)
    figure.group(44, 154, 554, 461, "GAME CONTINUATION", PALE_BLUE, BLUE)
    figure.group(626, 154, 570, 461, "LEARNING", PALE_TEAL, TEAL)
    figure.card(174, 211, 294, "Live session", ["State + controllers + history"], BLUE, 88)
    figure.card(766, 211, 294, "Training run", ["Model + optimizer + progress"], TEAL, 88)
    for center, left, right in [(321, 190, 451), (913, 774, 1052)]:
        figure.line(center, 303, center, 329, MUTED)
        figure.line(left, 329, right, 329, MUTED)
        figure.arrow(left, 329, left, 365)
        figure.arrow(right, 329, right, 365)
    for x, title, lines, accent in [
        (66, "State file", ["One rules position", "Chance RNG, where used", "Choose controllers on load"], BLUE),
        (327, "Replay file", ["Moves + branches", "Controller checkpoints", "Review / fork / continue"], BLUE),
        (648, "Portable model", ["Inference weights + spec", "For compatible agents", "Load locally or in browser"], TEAL),
        (926, "Training checkpoint", ["Model + optimizer", "Training RNG + progress", "Resume optimization"], TEAL),
    ]:
        figure.card(x, 365, 246, title, lines, accent, 161)
    figure.text(66, 579, "Game saves may contain hidden information.", 16, MUTED)
    figure.text(648, 579, "Models evaluate; checkpoints resume training.", 16, MUTED)
    figure.footer("Exact continuation needs saved progress, not just the original seed. File compatibility is validated on load.")
    figures.append(figure)
    for figure in figures:
        figure.height -= 120
        shifted = []
        for kind, *args in figure.shapes:
            args[1] -= 120
            if kind == "line":
                args[3] -= 120
            shifted.append((kind, *args))
        figure.shapes = shifted
    return figures


def svg(figure):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{figure.width}" '
             f'height="{figure.height}" viewBox="0 0 {figure.width} {figure.height}" '
             'role="img" aria-labelledby="title description">',
             f'<title id="title">{escape(figure.title)}</title>',
             f'<desc id="description">{escape(figure.description)}</desc>',
             f'<rect width="{figure.width}" height="{figure.height}" fill="{BACKGROUND}"/>',
             '<g font-family="Arial, Helvetica, sans-serif">']
    for kind, *args in figure.shapes:
        if kind == "text":
            x, y, label, size, color, bold, anchor = args
            parts.append(f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
                         f'font-size="{size}" font-weight="{700 if bold else 400}" '
                         f'fill="{color}">{escape(label)}</text>')
        elif kind == "rect":
            x, y, width, height, fill, stroke, radius = args
            parts.append(f'<rect x="{x:g}" y="{y:g}" width="{width}" height="{height}" '
                         f'rx="{radius}" fill="{fill}" stroke="{stroke or "none"}"/>')
        else:
            x1, y1, x2, y2, color, dashed, thickness = args
            dash = ' stroke-dasharray="5 7"' if dashed else ''
            parts.append(f'<path d="M{x1:g} {y1:g} L{x2:g} {y2:g}" '
                         f'fill="none" stroke="{color}" stroke-width="{thickness}" '
                         f'stroke-linecap="round"{dash}/>')
    return ("\n".join(parts) + "\n</g>\n</svg>\n").encode()


def pdf(figure):
    """Write one vector page with standard fonts, using the SVG scene geometry."""
    def color(value):
        return " ".join(f"{int(value[index:index + 2], 16) / 255:.4f}" for index in (1, 3, 5))

    def literal(value):
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def rounded(x, y, width, height, r):
        # PDF's origin is bottom-left; use cubic quarter circles for rounded cards.
        y = figure.height - y - height
        k = 0.55228475 * r
        return (f"{x+r:g} {y:g} m {x+width-r:g} {y:g} l "
                f"{x+width-r+k:g} {y:g} {x+width:g} {y+r-k:g} {x+width:g} {y+r:g} c "
                f"{x+width:g} {y+height-r:g} l "
                f"{x+width:g} {y+height-r+k:g} {x+width-r+k:g} {y+height:g} {x+width-r:g} {y+height:g} c "
                f"{x+r:g} {y+height:g} l "
                f"{x+r-k:g} {y+height:g} {x:g} {y+height-r+k:g} {x:g} {y+height-r:g} c "
                f"{x:g} {y+r:g} l {x:g} {y+r-k:g} {x+r-k:g} {y:g} {x+r:g} {y:g} c h")

    commands = [f"{color(BACKGROUND)} rg", f"0 0 {figure.width} {figure.height} re f", "1 J"]
    for kind, *args in figure.shapes:
        if kind == "text":
            x, y, label, size, fill, bold, anchor = args
            if anchor == "middle":
                x -= text_width(label, size, bold) / 2
            commands.append(f"BT /{'F2' if bold else 'F1'} {size} Tf {color(fill)} rg "
                            f"1 0 0 1 {x:.3f} {figure.height-y:.3f} Tm ({literal(label)}) Tj ET")
        elif kind == "rect":
            x, y, width, height, fill, stroke, radius = args
            commands.append(f"{color(fill)} rg {color(stroke or fill)} RG 1 w [] 0 d "
                            f"{rounded(x,y,width,height,radius)} {'B' if stroke else 'f'}")
        else:
            x1, y1, x2, y2, fill, dashed, thickness = args
            commands.append(f"{color(fill)} RG {thickness} w {'[5 7]' if dashed else '[]'} 0 d "
                            f"{x1:g} {figure.height-y1:g} m {x2:g} {figure.height-y2:g} l S")
    stream = "\n".join(commands).encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {figure.width} {figure.height}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>").encode(),
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
               f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
               (f"<< /Title ({literal(figure.title)}) /Subject ({literal(figure.description)}) >>").encode("ascii")]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend((f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R /Info 7 0 R >>\n"
                   f"startxref\n{start}\n%%EOF\n").encode())
    return bytes(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check exports without changing files")
    args = parser.parse_args()
    failures = []
    figures = diagrams()
    for figure in figures:
        for suffix, render in [("svg", svg), ("pdf", pdf)]:
            path = DESTINATION / f"{figure.name}.{suffix}"
            content = render(figure)
            if args.check:
                if not path.exists() or path.read_bytes() != content:
                    failures.append(path.name)
            else:
                DESTINATION.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if failures:
        parser.exit(1, "Missing or outdated exports: " + ", ".join(failures) + "\n")
    print(f"{'Checked' if args.check else 'Generated'} {len(figures)} figures in SVG and PDF.")


if __name__ == "__main__":
    main()
