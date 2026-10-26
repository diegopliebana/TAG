"""Compare complete seeded games and isolated heuristic scoring. Run outside a debugger."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import tag_rust as tag
from workloads import play, play_owned, verify_owned, evaluate_open_lines
from report import write_tables

ADAPTER = Path(__file__).resolve().parents[3] / "adapters/python"


def measure(operation):
    """Return seven elapsed times per operation, with a warm-up before timing."""
    operation()
    return [sample(operation) for _ in range(7)]


def sample(operation):
    start = time.perf_counter()
    repeats = 0
    while time.perf_counter() - start < 0.5:
        operation()
        repeats += 1
    return (time.perf_counter() - start) / repeats


def benchmark_case(case):
    """Verify and measure one board configuration across the Python scoring routes."""
    print(f"Measuring {case['game']} {case['parameters']}...", flush=True)
    lines = np.array(case["line_indices"], dtype=np.int64).reshape(-1, case["window_length"])
    with tag.create(case["game"], case["parameters"]) as game:
        identity = game.spec().identity
        rust_scorer = game.open_lines_evaluator()
    corpus = []
    for batch in case.pop("batches"):
        request = SimpleNamespace(
            identity=identity, actor=batch["actor"], action_ids=np.arange(len(batch["cells"]), dtype=np.int64),
            payloads={"cells": np.array(batch["cells"], dtype=np.float32)},
            results=np.array(batch["results"], dtype=np.int8),
        )
        np.testing.assert_array_equal(evaluate_open_lines(request, lines), batch["values"])
        np.testing.assert_array_equal(rust_scorer.evaluate(request), batch["values"])
        corpus.append(request)
    case["python_score_seconds"] = [s / case["candidates"] for s in measure(
        lambda: [evaluate_open_lines(request, lines) for request in corpus])]
    case["python_calls_rust_score_seconds"] = [s / case["candidates"] for s in measure(
        lambda: [rust_scorer.evaluate(request) for request in corpus])]
    routes = {"python": "python", "python_calls_rust": "rust"}
    for route, scoring in routes.items():
        case[route] = {}
        for rows in (1, 3, 1024):
            traces = [play(case, 100 + i * 2, rows, scoring=scoring)[0] for i in range(16)]
            assert traces == case["games"], "Python-coordinated and native game traces differ"
            case[route][str(rows)] = dict(game_seconds=[])
    case["python_starts_rust_runs"] = {}
    for rows in (1,3,1024):
        for i, expected in enumerate(case["games"]):
            verify_owned(case,100+i*2,rows,expected)
        case["python_starts_rust_runs"][str(rows)] = dict(game_seconds=[])
    # Interleave batch settings to reduce bias from changes in CPU frequency
    # or other system activity during a run.
    order_rng = random.Random(123)
    for _ in range(7):
        order = [(route, rows) for route in (*routes,"python_starts_rust_runs") for rows in (1, 3, 1024)]
        order_rng.shuffle(order)
        for route, rows in order:
            if route == "python_starts_rust_runs":
                operation = lambda: [play_owned(case,100+i*2,rows) for i in range(16)]
            else:
                operation = lambda: [play(case,100+i*2,rows,scoring=routes[route],record_moves=False)[0] for i in range(16)]
            elapsed = sample(operation)
            case[route][str(rows)]["game_seconds"].append(elapsed / 16)
    for route, rows in [(route, rows) for route in routes for rows in (1, 3, 1024)]:
        # Time the individual calls separately, so their timer overhead is
        # excluded from the main complete-game measurements above.
        profile = dict(prepare=0.0, evaluate=0.0, finish=0.0, commit=0.0, requests=0)
        for i in range(16):
            _, totals = play(case, 100 + i * 2, rows, profile=True, scoring=routes[route])
            for key, value in totals.items():
                profile[key] += value / 16
        case[route][str(rows)]["profile_per_game"] = profile
    case["mean_moves"] = statistics.mean(map(len, case["games"]))
    case["mean_candidates"] = case["candidates"] / 16
    native_ms = statistics.mean(case["native_game_seconds"]) * 1000
    python_ms = statistics.mean(case["python"]["1024"]["game_seconds"]) * 1000
    batch_ms = statistics.mean(case["rust_batch"]["1024"]["game_seconds"]) * 1000
    crossing_ms = statistics.mean(case["python_calls_rust"]["1024"]["game_seconds"]) * 1000
    owned_ms = statistics.mean(case["python_starts_rust_runs"]["1024"]["game_seconds"]) * 1000
    print(f"  full game: original Rust {native_ms:.4f} ms; Rust-only batched {batch_ms:.4f} ms; Python calls Rust {crossing_ms:.4f} ms; Python scoring {python_ms:.4f} ms; Python starts/Rust runs {owned_ms:.4f} ms", flush=True)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Compilation and process startup are outside every reported timing.
    native = subprocess.run(
        ["cargo", "run", "--offline", "--release", "--quiet", "--manifest-path",
         str(ADAPTER / "Cargo.toml"), "--example", "benchmark_osla"],
        check=True, capture_output=True, text=True,
    )
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), numpy=np.__version__,
                  platform=platform.platform(), cpu=os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
                  logical_cpus=os.cpu_count(), rust=subprocess.check_output(["rustc", "--version"], text=True).strip(),
                  method=dict(seeds="(100+2*i,101+2*i), i=0..15", samples=7,
                              minimum_sample_seconds=0.5, batch_sizes=[1, 3, 1024],
                              record_moves_during_timing=False,
                              batch_order="shuffled each round, random.Random(123)",
                              extension_build="maturin develop --release --offline"), cases=[])
    report["cases"] = [benchmark_case(case) for case in json.loads(native.stdout)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_tables(report, args.output.with_suffix(".md"))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
