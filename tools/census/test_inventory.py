"""Summarise captured standard test-runner output without counting assertions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import unittest


def rust_inventory(text):
    cases = defaultdict(list)
    context = "unknown"
    for line in text.splitlines():
        if "Running " in line:
            context = line.strip()
        match = re.fullmatch(r"test (.+) \.\.\. (ok|FAILED|ignored)", line)
        if not match:
            continue
        name, status = match.groups()
        if name.startswith(("matches::", "tournaments::")):
            identity = "shared-runner::" + name
        else:
            crate = "browser" if "tag_browser-" in context else "core"
            if "_engine-" in context:
                crate = "python-adapter"
            identity = crate + "::" + name
        # Paths to executables are context, never a second logical test.
        cases[identity].append(dict(context=context.split(" (")[0], status=status))
    return [dict(id=key, executions=value) for key, value in sorted(cases.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--discover-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.discover_python:
        def flatten(suite):
            for test in suite:
                if isinstance(test, unittest.TestSuite):
                    yield from flatten(test)
                else:
                    yield test.id()
        suite = unittest.defaultTestLoader.discover(str(args.discover_python))
        identities = sorted(flatten(suite))
        assert not any("_FailedTest" in name for name in identities)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(identities, indent=2) + "\n", encoding="utf-8")
        print(len(identities))
        return
    if not args.logs:
        parser.error("provide --logs or --discover-python")
    read = lambda name: (args.logs / name).read_text(encoding="utf-8-sig")
    rust = rust_inventory(read("rust-all-run.log"))
    discovered = re.findall(r"^(.+): test$", read("rust-all-list.log"), re.M)
    assert len(discovered) == sum(len(r["executions"]) for r in rust)
    assert set(discovered) == {r["id"].split("::", 1)[1] for r in rust}
    python_rust = rust_inventory(read("python-rust.log"))
    assert all(e["status"] == "ok" for r in rust + python_rust for e in r["executions"])
    python = json.loads(read("python-discovered.json"))
    log = read("python.log")
    assert f"Ran {len(python)} tests" in log and "\nOK\n" in log
    assert len(re.findall(r"^test_.+ \.\.\. ok$", log, re.M)) == len(python)
    assert set(re.findall(r"^test_\S+ \(([^)]+)\) \.\.\. ok$", log, re.M)) == set(python)
    node = re.findall(r"^ok \d+ - (.+)$", read("node.log"), re.M)
    assert len(node) == len(set(node)) and f"# pass {len(node)}" in read("node.log")
    tools = []
    for suite in ("census-unit", "statistics-unit"):
        content = read(suite + ".log")
        names = re.findall(r"^(test_\S+) \(([^)]+)\) \.\.\. ok$", content, re.M)
        assert f"Ran {len(names)} tests" in content and "\nOK\n" in content
        tools.extend(dict(suite=suite, id=full) for _, full in names)
    doclog = read("rust-doctests.log").split("Doc-tests tag_rust", 1)[1]
    doctests = re.findall(r"^test (.+) \.\.\. ok$", doclog, re.M)
    assert len(doctests) == 1 and "1 passed; 0 failed" in doclog
    checks = json.loads(read("extra-runs.json"))
    assert checks["match-cli"]["exit_code"] == checks["tournament-cli"]["exit_code"] == 0
    assert "GPU process exited unexpectedly: exit_code=-1073741790" in read("browser.log")
    data = dict(
        rust=rust, python_adapter_rust=python_rust, rust_doctests=doctests,
        python_methods=python, javascript_tests=node, tool_unittests=tools,
        summary=dict(rust_unique=len(rust), rust_executions=len(discovered),
                     python_adapter_rust=len(python_rust), rust_doctests=len(doctests),
                     python_methods=len(python), javascript_tests=len(node),
                     tool_unittests=len(tools), failed=0, ignored_or_skipped=0),
        operational_checks=dict(match_cli="passed", tournament_cli="passed",
                                graphical_browser="unavailable: GPU process exit -1073741790 before page checks"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["summary"]))


if __name__ == "__main__":
    main()
