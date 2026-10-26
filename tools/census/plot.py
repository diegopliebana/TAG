"""Export compact census figures; requires Matplotlib."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/benchmarks/framework-size/data"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/framework-size/plots"))
    args = parser.parse_args()
    rows = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    source = {"Rust", "Python", "Python interface", "JavaScript", "Java", "PowerShell", "HTML", "CSS"}
    for name, field, selected, unit in (
        ("source-languages", "language", [r for r in rows if r["language"] in source], "Physical source lines (including comments)"),
        ("source-components", "category", [r for r in rows if r["language"] in source], "Physical source lines (including comments)"),
        ("files", "category", rows, "Physical files"),
    ):
        counts = Counter()
        for row in selected:
            counts[row[field]] += 1 if name == "files" else row["lines"] or 0
        items = sorted(counts.items(), key=lambda item: item[1])
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(10, max(3.6, len(items) * .38)))
            bars = ax.barh([k for k, _ in items], [v for _, v in items], color="#367b9b", height=.65)
            ax.bar_label(bars, labels=[f"{v:,}" for _, v in items], padding=4, fontsize=9)
            ax.set_xlabel(unit)
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=.18)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_xlim(0, max(counts.values()) * 1.18)
            fig.tight_layout()
            for extension in ("svg", "pdf", "png"):
                fig.savefig(args.output / f"{name}.{extension}", dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    main()
