"""Check local Markdown links and keep public documentation self-contained.

Run from any directory. No packages, network access or Git metadata are required.
Use --all-links to check links in every documentation record as well.
"""

import argparse
import html
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = {"docs/project", "docs/milestones", "docs/reviews", "tools/release"}
IGNORED = {".git", ".scratch", "target", "node_modules", "pkg", "__pycache__"}
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")


def markdown_files():
    """Walk authored directories, excluding environments and generated output."""
    pending = [ROOT]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or path.name in IGNORED or path.name.startswith(".venv"):
                continue
            if path.is_dir():
                pending.append(path)
            elif path.suffix == ".md":
                yield path


def is_private(path):
    name = path.relative_to(ROOT).as_posix()
    return name == "AGENTS.md" or any(
        name == prefix or name.startswith(prefix + "/") for prefix in PRIVATE
    )


def prose(text):
    return re.sub(r"^\s*(```|~~~).*?^\s*\1\s*$", "", text, flags=re.M | re.S)


def anchors(text):
    result = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    used = {}
    for line in prose(text).splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)(?:\s+#+)?\s*$", line)
        if not match:
            continue
        label = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", match[1])
        label = re.sub(r"<[^>]+>", "", html.unescape(label)).lower()
        slug = re.sub(r"[^\w\s-]", "", label).replace(" ", "-")
        count = used.get(slug, 0)
        used[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-links", action="store_true")
    args = parser.parse_args()
    errors = []
    checked = 0
    pages = 0
    anchor_cache = {}
    for path in markdown_files():
        private = is_private(path)
        if private and not args.all_links:
            continue
        pages += 1
        text = prose(path.read_text(encoding="utf-8"))
        relative = path.relative_to(ROOT).as_posix()
        for match in LINK.finditer(text):
            target = match[1].strip("<>")
            if re.match(r"^[a-zA-Z][\w+.-]*:", target) or target.startswith("//"):
                continue
            file, _, fragment = unquote(target).partition("#")
            destination = (path.parent / file).resolve() if file else path
            if not destination.is_relative_to(ROOT):
                errors.append(f"{relative}: local link leaves repository: {target}")
                continue
            checked += 1
            if not destination.exists():
                errors.append(f"{relative}: missing target: {target}")
                continue
            if not private and is_private(destination):
                errors.append(f"{relative}: public link depends on development records: {target}")
            if fragment and destination.suffix == ".md":
                if destination not in anchor_cache:
                    anchor_cache[destination] = anchors(destination.read_text(encoding="utf-8"))
                if fragment not in anchor_cache[destination]:
                    errors.append(f"{relative}: missing anchor: {target}")
        if not private:
            readable = LINK.sub(lambda match: match[0].split("](")[0] + "]", text)
            readable = re.sub(r"`[^`\n]+`", "", readable)
            forbidden = re.search(
                r"\b(?:milestones?|M\d+|S\d+[a-d]?|stage\s+\d+|later\s+stages?|this\s+stage)\b",
                readable, flags=re.I,
            )
            if forbidden:
                errors.append(f"{relative}: development-progress wording: {forbidden[0]}")
            if re.search(r"\b(?:prepared|written|designed)\s+for\s+students\b", readable, re.I):
                errors.append(f"{relative}: use neutral audience wording")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {pages} Markdown pages and {checked} local links; {len(errors)} errors.")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
