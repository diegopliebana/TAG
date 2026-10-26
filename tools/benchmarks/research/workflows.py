"""Collect full-game and prepared-position diagnostics for observation/validation/copy costs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run import ROOT, PROFILES, java_binary


def cases():
    for game, players in PROFILES:
        yield game, players, "game_direct", 0
        if (game, players) == ("loveletter", 2):
            yield game, players, "game_aligned", 0
            yield game, players, "game_aligned_direct", 0
        yield game, players, "apply_control", 0
        for prefix in (0, 8, 128):
            for op in ("copy", "observe", "actions", "validate", "apply", "save", "load"):
                yield game, players, op, prefix
            if game == "loveletter":
                yield game, players, "history_copy", prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("timing", "allocations"), default="timing")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / f"workflows-{a.phase}.jsonl"
    if path.exists(): p.error("choose a fresh output directory or inspect the existing phase")
    java = java_binary()
    jroot = ROOT / ".scratch/java-comparison"
    cp = str(jroot / "target/classes") + ";" + (jroot / "target/comparison-classpath.txt").read_text().strip()
    forks = 3 if a.phase == "timing" else 1
    binary = ROOT / ".scratch/research-bench" / ("rust-timing.exe" if a.phase == "timing" else "rust-memory.exe")
    total = 0
    with path.open("w", encoding="utf-8") as stream:
        for fork in range(1, forks + 1):
            for game, players, op, prefix in cases():
                if a.phase == "allocations" and op == "apply_control": continue
                for engine in (["rust", "java"] if fork % 2 else ["java", "rust"]):
                    if engine == "java" and op in ("save", "load"): continue
                    command = [java, "-Xms64m", "-Xmx512m", "-cp", cp, "BenchmarkResearch"] if engine == "java" else [str(binary)]
                    warm = 1000 if engine == "java" else (150 if a.phase == "timing" else 20)
                    sample = 0 if op == "apply_control" else (80 if a.phase == "timing" else 5)
                    count = 32_000 if op == "apply_control" else 1
                    command += [game, str(players), f"{op}@{prefix}", str(count), "42", "32", str(warm), str(sample)]
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
                    if result.returncode: raise RuntimeError(f"{command}: {result.stderr}")
                    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                    assert len(rows) == 6, (command, result.stdout)
                    record = dict(engine=engine, game=game, players=players, operation=op, prefix=prefix,
                                  fork=fork, seed=42, warmup_ms=warm, sample_min_ms=sample, samples=rows)
                    stream.write(json.dumps(record) + "\n"); stream.flush(); total += 1
                if prefix == 128 and op == ("history_copy" if game == "loveletter" else "load"):
                    print(f"Workflow {a.phase}: fork {fork}, {game}/{players} complete ({total} processes)", flush=True)
    print(f"Saved {total} workflow process records", flush=True)


if __name__ == "__main__": main()
