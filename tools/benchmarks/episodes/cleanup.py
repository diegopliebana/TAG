"""Record selected whole-workload samples for the installed cleanup build."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before-cleanup", "after-cleanup"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for game in ("tictactoe", "connect4"):
        for count in (1, 128):
            result = measure.measure(game, "random", count, 7, 0.2)
            records.append(result)
            print(game, count, result["means_ms"], flush=True)
    value = dict(
        recorded=datetime.now().astimezone().isoformat(), phase=args.phase,
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=measure.ROOT, text=True).strip(),
        environment=dict(python=sys.version, numpy=measure.np.__version__,
                         gymnasium=measure.gymnasium.__version__, platform=platform.platform()),
        records=records,
    )
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
