"""Export normalized game and prepared-state figures, without visible titles."""
import argparse
import json
import os
from pathlib import Path
import statistics
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".scratch/research-bench/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run import PROFILES
from figure_layout import layout_with_legend

COLORS = {"rust": "#188477", "java": "#7c5db3"}
LABELS = {"tictactoe": "Tic-Tac-Toe", "connect4": "Connect 4", "cantstop": "Can't Stop", "loveletter": "Love Letter"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args(); root = a.directory
    read = lambda name: json.loads((root / name).read_text())
    games = {(r["engine"], r["game"], r["players"], r["operation"]): r for r in read("game-summary.json")}
    components = {(r["engine"], r["game"], r["players"], r["operation"], r["prefix"]): r for r in read("component-summary.json")}
    plots = root / "plots/workflows"; plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "axes.axisbelow": True, "pdf.fonttype": 42, "svg.fonttype": "none"})
    def save(fig, name):
        layout_with_legend(fig)
        for ext in ("svg", "pdf", "png"): fig.savefig(plots / f"{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    labels = [f"{LABELS[g]}\n{n}p" for g, n in PROFILES]
    x = np.arange(len(PROFILES))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric, scale, label, log in zip(axes.flat,
        ["ns_per_game", "actions_per_game", "turns_per_game", "ns_per_action"],
        [1000, 1, 1, 1], ["Complete game (µs)", "Accepted actions / game", "Logical turns / game", "Elapsed ns / accepted action"], [True, False, False, True]):
        for offset, engine in [(-.18, "rust"), (.18, "java")]:
            selected = [games[engine, g, n, "game"] for g, n in PROFILES]
            y = np.array([r[metric] / scale for r in selected])
            lo = np.array([min(f[metric] for f in r["forks"]) / scale for r in selected])
            hi = np.array([max(f[metric] for f in r["forks"]) / scale for r in selected])
            ax.bar(x + offset, y, .34, color=COLORS[engine], label=engine.capitalize(), yerr=[y-lo, hi-y], capsize=2)
        ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel(label)
        if log: ax.set_yscale("log")
    axes[0, 0].legend(frameon=False); save(fig, "random-game-work")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, engine in zip(axes, ("rust", "java")):
        for offset, mode, alpha, label in [(-.18, "game_direct", .5, "Direct-choice diagnostic"), (.18, "game", 1, "Observation + Random agent")]:
            y = [games[engine, g, n, mode]["ns_per_action"] for g, n in PROFILES]
            ax.bar(x + offset, y, .34, color=COLORS[engine], alpha=alpha, label=label)
        ax.set_xticks(x, labels, fontsize=9); ax.set_ylabel(f"{engine.capitalize()} ns / action"); ax.set_yscale("log"); ax.legend(frameon=False, ncol=2)
    save(fig, "observation-workflows")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    selected = [games["rust", "loveletter", 2, "game"], games["java", "loveletter", 2, "game"], games["rust", "loveletter", 2, "game_aligned"], games["java", "loveletter", 2, "game_aligned"]]
    for ax, metric, scale, label in zip(axes, ["ns_per_game", "actions_per_game", "ns_per_action"], [1000, 1, 1], ["Complete game (µs)", "Accepted actions / game", "Elapsed ns / action"]):
        values=np.array([r[metric]/scale for r in selected])
        lo=np.array([min(f[metric] for f in r['forks'])/scale for r in selected])
        hi=np.array([max(f[metric] for f in r['forks'])/scale for r in selected])
        ax.bar(range(4), values, color=[COLORS['rust'], COLORS['java'], '#72ada5', '#aa91ca'], yerr=[values-lo,hi-values],capsize=2)
        ax.set_xticks(range(4), ['Rust 6\ndefault','Java 7\ndefault','Rust 6\naligned run','Java 6\naligned run'],fontsize=8); ax.set_ylabel(label)
    save(fig, "loveletter-token-target")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    board_games = [(g, 2) for g in LABELS]
    for ax, op, label in zip(axes.flat, ["copy", "observe", "validate", "apply"], ["Full-state copy (µs)", "Player observation/copy (µs)", "Legal list + membership (µs)", "Checked in-place transition (µs)"]):
        for index, (engine, prefix) in enumerate((e, prefix) for e in ("rust", "java") for prefix in (0, 8, 128)):
            values = [components[engine, g, n, op, prefix]["median_ns"]/1000 for g, n in board_games]
            ax.bar(np.arange(4)+(index-2.5)*.12, values, .11, color=COLORS[engine], alpha={0:.4, 8:.7, 128:1}[prefix], label=f"{engine.capitalize()} prefix {prefix}")
        ax.set_xticks(range(4), [LABELS[g] for g, _ in board_games], fontsize=9); ax.set_ylabel(label); ax.set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "prepared-components")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, engine in enumerate(("rust", "java")):
        for col, op in enumerate(("copy", "history_copy")):
            ax = axes[row, col]
            for n, color in [(2, "#547fbd"), (3, "#188477"), (4, "#bd853d")]:
                values = [components[engine, "loveletter", n, op, prefix]["median_ns"]/1000 for prefix in (0, 8, 128)]
                ax.plot(range(3), values, marker="o", color=color, label=f"{n} players")
            ax.set_xticks(range(3), ["Opening", "8 additional\nactions", "128 or last\nnonterminal"])
            ax.set_ylabel(f"{engine.capitalize()} {'state copy' if op == 'copy' else 'history buffers'} (µs)")
    axes[0, 0].legend(frameon=False); save(fig, "loveletter-history-copy")

    print("Exported five workflow figure sets (SVG/PDF/PNG).")


if __name__ == "__main__": main()
