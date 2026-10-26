"""Measure independent-game grouping on CPU; no neural model is involved."""
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import json
import argparse
import platform
import statistics
import sys
import time
import numpy as np
import tag_rust as tag

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "adapters/python/examples"))
from open_lines import evaluate_open_lines


def workload(name, grouped, count=16, *, constant=False):
    with ExitStack() as stack:
        games = [stack.enter_context(tag.create(name)) for _ in range(count)]
        for i,g in enumerate(games):
            for seat in range(2):
                g.set_controller(seat,tag.OSLAPython(evaluator_id="benchmark",seed=100+2*i+seat))
        support = games[0].spec().heuristic
        lines = support.indices.reshape(-1,support.window_length)
        calls = 0
        rows = 0
        def evaluate(request):
            nonlocal calls, rows
            calls += 1
            rows += len(request.action_ids)
            return np.zeros(len(request.action_ids),dtype=np.float64) if constant else evaluate_open_lines(request,lines)
        if grouped:
            batch = stack.enter_context(tag.GameBatch(games))
            queue = deque(batch.advance().groups)
            while queue:
                request = queue.popleft()
                result = batch.resume(request,evaluate(request))
                if any(s.error for s in result.stops):
                    raise RuntimeError(result.stops)
                queue.extend(result.groups)
        else:
            for game in games:
                result = game.advance()
                while result.reason == "evaluation":
                    result = game.resume(result.request,evaluate(result.request))
        # Capturing final facts/checkpoints costs the same in both timed routes.
        facts = [(g.observe(0,payloads=("cells",)).payloads["cells"].tolist(),
                  g.observe(0,payloads=()).results.tolist(),
                  [g.controller_checkpoint(s) for s in range(2)]) for g in games]
        return calls,rows,facts


def measure(name, count, constant, samples, seconds):
    baseline = workload(name,False,count,constant=constant)
    combined = workload(name,True,count,constant=constant)
    assert baseline[1:] == combined[1:], "seeded results or checkpoints differ"
    times = {"individual":[],"grouped":[]}
    repetitions = {"individual":[],"grouped":[]}
    for sample in range(samples):
        for grouped in ((False,True) if sample % 2 == 0 else (True,False)):
            elapsed = 0.0
            repetitions_done = 0
            while elapsed < seconds:
                start = time.perf_counter()
                result = workload(name,grouped,count,constant=constant)
                elapsed += time.perf_counter()-start
                assert result == (combined if grouped else baseline)
                repetitions_done += 1
            route = "grouped" if grouped else "individual"
            times[route].append(1000*elapsed/repetitions_done)
            repetitions[route].append(repetitions_done)
    return dict(game=name,evaluator="constant" if constant else "python-open-lines",
        games=count,rows=baseline[1],individual_calls=baseline[0],grouped_calls=combined[0],
        milliseconds=times,repetitions=repetitions,minimum_sample_seconds=seconds,
        means={k:statistics.mean(v) for k,v in times.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling",action="store_true",help="Measure 1 through 128 games in powers of two.")
    args = parser.parse_args()
    counts = (1,2,4,8,16,32,64,128) if args.scaling else (16,)
    records = []
    for name in ("tictactoe","connect4"):
        for constant in (False,True):
            for count in counts:
                record = measure(name,count,constant,7,0.2)
                records.append(record)
                print(f"{name} / {record['evaluator']} / {count} games: calls "
                      f"{record['individual_calls']} -> {record['grouped_calls']}; ms "
                      f"{record['means']['individual']:.3f} -> {record['means']['grouped']:.3f}",flush=True)
    folder = ROOT / "docs/benchmarks/osla-python"
    if args.scaling:
        value = dict(environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                    processor=platform.processor()),
                    settings=dict(max_rows=1024,max_group_rows=4096,max_bytes=67108864,
                                  payloads=["cells"],samples=7,minimum_sample_seconds=0.2),records=records)
        (folder / "scaling-results.json").write_text(json.dumps(value,indent=2)+"\n")
    else:
        (folder / "grouped-results.json").write_text(json.dumps(records,indent=2)+"\n")


if __name__ == "__main__":
    main()
