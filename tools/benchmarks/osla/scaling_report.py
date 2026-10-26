"""Render saved independent-game scaling measurements without running games."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "docs/benchmarks/osla-python"


def main():
    data = json.loads((FOLDER / "scaling-results.json").read_text())
    lines = ["# Independent-game grouping: CPU scaling experiment", "",
        "Measured 16-10-2026 (Europe/Paris). [Raw samples and environment](scaling-results.json).", "",
        "These samples precede the final external-evaluation cleanup. The [cleanup measurement](cleanup-results.md)",
        "records a targeted before/after comparison; saved scaling samples are retained.", "",
        "The number of games varies through 1, 2, 4, 8, 16, 32, 64 and 128. All other",
        "settings stay fixed: default boards, cells inputs, controller max_rows=1,024,",
        "coordinator max_group_rows=4,096 and max_bytes=64 MiB. Each game has independently",
        "seeded controllers: game i uses seeds 100+2*i and 101+2*i. Larger workloads",
        "retain the same first games and add more. Rust runs games sequentially; the",
        "experiment measures grouping efficiency on CPU, without worker threads or a neural model.", "",
        "The individual route runs complete games one after another with advance/resume.",
        "The grouped route uses GameBatch to advance all games and combine compatible requests.",
        "Both evaluate the same candidates and produce identical final boards, results and",
        "complete controller/RNG checkpoints at every repetition.", "",
        "## How to read the tables", "",
        "Each row compares two ways of completing the same number of games. The time",
        "columns show mean milliseconds per completed game: total workload time divided",
        "by the number of games. The time ratio is individual time divided by grouped time.",
        "It compares the routes at that row's fixed game count; it describes no linear",
        "growth rate as more games are added. A ratio of 1.22 means the individual route",
        "took 1.22 times as long: grouping used about 82% of that time, saving about 18%.",
        "A ratio below 1 means the grouped route took longer.", "",
        "The last two columns count calls to the Python evaluator while completing the",
        "entire set of games once. They count function calls, with many candidate states",
        "potentially evaluated in each call. They exclude repetitions used for timing.", "",
        "For example, the 128-game Tic-Tac-Toe heuristic row has 1,152 individual calls",
        "and 9 grouped calls. Individually, each of 128 nine-move games requests evaluation",
        "nine times: 128 * 9 = 1,152. With grouping, the first call scores candidates for",
        "the first move of all 128 games, the second handles their second moves, and so",
        "on for nine calls. The first grouped call contains 128 * 9 = 1,152 candidate",
        "states; later calls contain fewer candidates as the boards fill. Both routes",
        "evaluate the same total candidate states. The saving comes from handling them",
        "in fewer calls. The summary notation 1,152 -> 9 means those two call counts.", ""]
    for game in ("tictactoe", "connect4"):
        for evaluator in ("python-open-lines", "constant"):
            lines += [f"## {game}: {evaluator}", "",
                "| Games | Individual ms/game | Grouped ms/game | Time ratio: individual / grouped | Python calls, individual (all games) | Python calls, grouped (all games) |",
                "|---:|---:|---:|---:|---:|---:|"]
            for r in data["records"]:
                if r["game"] != game or r["evaluator"] != evaluator:
                    continue
                n = r["games"]
                a,b = r["means"]["individual"],r["means"]["grouped"]
                lines.append(f"| {n} | {a/n:.4f} | {b/n:.4f} | {a/b:.2f} | {r['individual_calls']} | {r['grouped_calls']} |")
            lines.append("")
    lines += ["At 128 games, the first combined requests contain 1,152 candidate rows for",
        "Tic-Tac-Toe and 896 for Connect 4, both below the fixed group cap of 4,096.",
        "This experiment therefore varies game count while keeping the row cap inactive;",
        "it does not measure the effect of splitting groups at a smaller cap.", "",
        "## Method and limits of the result", "",
        "Seven samples follow validation/warm-up runs. Route order alternates within each",
        "game-count case. Each sample repeats complete workloads until at least 200 ms of",
        "measured time accumulates; long workloads can exceed that minimum in one run.",
        "The table averages the seven sample means. Timed work includes setup, complete play,",
        "final observation/checkpoint extraction and cleanup. Equality comparisons happen",
        "outside timing. Raw samples retain repetition counts and variation.", "",
        "Game counts run in ascending order, with the real heuristic before the constant",
        "evaluator. CPU load, temperature and interpreter behavior can vary during the run.",
        "Treat small timing differences cautiously, particularly when locating a crossover",
        "between routes. Call counts and exact-result comparisons are deterministic for",
        "these seeds and versions. These timings describe whole workloads rather than",
        "isolating copying, Rust preparation or NumPy conversion.", "",
        "The real Python heuristic still loops through each candidate and winning line.",
        "The constant NumPy evaluator returns zero, giving a cheap-scoring comparison.",
        "Compare routes within an evaluator: its scores affect chosen moves and game length.",
        "Larger counts amortize coordination calls over more games, but also allocate larger",
        "arrays. Fewer calls alone do not guarantee improved total runtime. There is no",
        "measurement of peak memory, GPU transfer, PyTorch inference or within-tree MCTS.", "",
        "Reproduce with:", "", "```powershell",
        ".\\.venv-m5\\Scripts\\python.exe tools/benchmarks/osla/grouped.py --scaling",
        ".\\.venv-m5\\Scripts\\python.exe tools/benchmarks/osla/scaling_report.py", "```", "",
        "The earlier [16-game measurement](grouped-results.md) is retained. Compare the",
        "scaling cases within this fresh run; earlier measurements can have different load.", ""]
    (FOLDER / "scaling-results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
