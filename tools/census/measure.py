"""Dependency-free physical-file census; run from the repository root."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

VERSION = "1.0"
LANGUAGES = {
    ".rs": "Rust", ".py": "Python", ".pyi": "Python interface",
    ".js": "JavaScript", ".mjs": "JavaScript", ".java": "Java",
    ".ps1": "PowerShell", ".md": "Markdown", ".html": "HTML",
    ".css": "CSS", ".svg": "SVG", ".json": "JSON", ".jsonl": "JSON Lines",
    ".toml": "TOML", ".lock": "Lockfile", ".csv": "CSV", ".tsv": "TSV",
    ".txt": "Text",
}
BINARY = {".png", ".pdf", ".ogg", ".pt", ".wasm", ".pyd"}
PRIVATE = ("docs/project/", "docs/milestones/", "docs/reviews/", "tools/release/")
PRODUCTS = ("docs/benchmarks/framework-size/data/", "docs/benchmarks/framework-size/plots/")


def public(path):
    return path != "AGENTS.md" and not path.startswith(PRIVATE)


def category(path):
    """Assign each physical file exactly once; ordered rules are intentional."""
    suffix = Path(path).suffix
    if "/results/" in path or path.startswith("results/"):
        return "Training outputs"
    if path.startswith("docs/"):
        if suffix in {".svg", ".pdf", ".png", ".jsonl", ".tsv", ".csv"}:
            return "Documentation exports and measurement data"
        if path.startswith("docs/benchmarks/") and suffix == ".json":
            return "Documentation exports and measurement data"
        return "Documentation"
    if path.startswith("web/assets/"):
        return "Browser artwork and audio"
    if (path.startswith("tests/") or "/tests/" in path
            or Path(path).name.startswith("test_") or Path(path).name.endswith("tests.rs")):
        return "Tests"
    if path.startswith(("tools/benchmarks/", "tools/java-comparison/")):
        return "Benchmark tools"
    if path.startswith("tools/") or path.endswith(".ps1"):
        return "Development and execution tools"
    if "/examples/" in path or path.startswith("examples/"):
        return "Examples and fixtures"
    if path.startswith("src/"):
        return "Core runtime"
    if path.startswith("adapters/python/") and suffix in {".rs", ".py", ".pyi"}:
        return "Python adapter and library"
    if path.startswith("adapters/browser/") and suffix == ".rs":
        return "Browser adapter"
    if path.startswith("web/") and suffix in {".js", ".mjs", ".html", ".css"}:
        return "Browser frontend"
    return "Configuration, metadata and other assets"


def line_counts(data):
    """Count physical lines, including an unterminated final line."""
    text = data.decode("utf-8-sig")
    lines = text.splitlines()
    blank = sum(not line.strip() for line in lines)
    return len(lines), blank, len(lines) - blank


def summarize(rows):
    grouped = defaultdict(lambda: dict(files=0, bytes=0, lines=0, blank=0, nonblank=0))
    for row in rows:
        target = grouped[(row["category"], row["language"])]
        target["files"] += 1
        for key in ("bytes", "lines", "blank", "nonblank"):
            target[key] += row[key] or 0
    return [dict(category=k[0], language=k[1], **v) for k, v in sorted(grouped.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).decode("utf-8").split("\0")
    rows = []
    for name in sorted(set(filter(None, names))):
        path = Path(name)
        if any(part in {"target", "node_modules", "__pycache__", ".scratch", ".git"}
               or part.startswith(".venv") for part in path.parts):
            continue
        if name.startswith("web/pkg/") or (name.startswith("adapters/python/python/tag_rust/")
                                              and path.suffix in {".pyd", ".pdb", ".so"}):
            continue
        if name.startswith(PRODUCTS) or name.startswith("docs/milestones/m11/evidence/stage7/"):
            continue  # Census products cannot recursively measure themselves.
        if any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents)):
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        language = LANGUAGES.get(path.suffix, "Other text")
        counts = (None, None, None)
        if path.suffix in BINARY:
            language = "Binary " + path.suffix[1:].upper()
        else:
            counts = line_counts(data)
        rows.append(dict(path=name, category=category(name), language=language, sha256=hashlib.sha256(data).hexdigest(),
                         bytes=len(data), lines=counts[0], blank=counts[1], nonblank=counts[2]))
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    for directory, selected in ((args.output, [r for r in rows if public(r["path"])]),
                                (args.private_output, rows)):
        (directory / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        summary = summarize(selected)
        (directory / "summary.json").write_text(json.dumps(dict(tool_version=VERSION, groups=summary), indent=2) + "\n", encoding="utf-8")
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    projection = [r for r in rows if public(r["path"])]
    def table(field, selected):
        values = defaultdict(lambda: [0, 0, 0, 0])
        for row in selected:
            value = values[row[field]]
            value[0] += 1
            value[1] += row["lines"] or 0
            value[2] += row["blank"] or 0
            value[3] += row["nonblank"] or 0
        lines = ["| Group | Files | Physical lines | Blank | Nonblank |",
                 "|---|---:|---:|---:|---:|"]
        for key, value in sorted(values.items()):
            lines.append("| " + key + " | " + " | ".join(f"{v:,}" for v in value) + " |")
        totals = [sum(v[i] for v in values.values()) for i in range(4)]
        lines.append("| **Total** | " + " | ".join(f"{v:,}" for v in totals) + " |")
        return "\n".join(lines)
    text = "# Census tables\n\nGenerated by census version " + VERSION + ". Binary files have no line count.\n\n"
    text += "## Public-scope projection by language\n\n" + table("language", projection)
    text += "\n\n## Public-scope projection by component\n\n" + table("category", projection)
    source = {"Rust", "Python", "Python interface", "JavaScript", "Java", "PowerShell", "HTML", "CSS"}
    source_rows = [r for r in projection if r["language"] in source]
    text += "\n\n## Public source files by component\n\n" + table("category", source_rows)
    text += "\n\n## Working checkout by language\n\n" + table("language", rows) + "\n"
    (args.output / "tables.md").write_text(text, encoding="utf-8")
    bindings = []
    candidates = list(Path("web/pkg").glob("*"))
    for extension in ("*.pyd", "*.so"):
        candidates.extend(Path("adapters/python/python/tag_rust").glob(extension))
    for path in sorted(candidates):
        if path.is_file() and not any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
            data = path.read_bytes()
            bindings.append(dict(path=path.as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (args.output / "generated-bindings.json").write_text(json.dumps(bindings, indent=2) + "\n", encoding="utf-8")
    products = []
    for prefix in PRODUCTS:
        for path in sorted(Path(prefix).rglob("*")):
            if path.is_file() and path.name != "products.json":
                data = path.read_bytes()
                products.append(dict(path=path.as_posix(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (args.output / "products.json").write_text(json.dumps(products, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(working_files=len(rows), public_files=sum(public(r["path"]) for r in rows))))


if __name__ == "__main__":
    main()
