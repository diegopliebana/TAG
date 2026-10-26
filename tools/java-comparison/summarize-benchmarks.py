# Read repeated Java/Rust timing samples and check that both timed the same work.
# Summaries use medians; this script writes reports, not expected game-rule results.
from pathlib import Path
import statistics
import hashlib
import argparse
parser = argparse.ArgumentParser(description="Check matched work and summarize Java/Rust timings.")
parser.add_argument("directory", nargs="?", default="docs/benchmarks/java-rust")
parser.add_argument("--forks", type=int, default=3)
arguments = parser.parse_args()
if arguments.forks < 1:
    parser.error("forks must be positive")
root = Path(arguments.directory)
rows = []
corpora = {}
for engine in ("rust", "java"):
    for fork in range(1, arguments.forks + 1):
        path = root / f"{engine}-fork-{fork}.tsv"
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        samples = [line.split("\t") for line in lines if line.startswith("sample\t")]
        assert len(samples) == 50, (path, len(samples))
        for line in lines:
            if line.startswith("corpus\t"):
                parts = line.split("\t")
                signature = tuple(parts[2:])
                if parts[1] in corpora:
                    assert corpora[parts[1]] == signature, (path, line)
                corpora[parts[1]] = signature
        for fields in samples:
            assert fields[1] == engine
            rows.append((engine, fields[2], fields[3], fork, int(fields[4]), int(fields[5]), float(fields[6])))
summaries = ["game\toperation\tunit\trust_median_ns\tjava_median_ns\tjava_over_rust\trust_min_ns\trust_max_ns\tjava_min_ns\tjava_max_ns"]
for game in ("ttt", "c4"):
    for operation in ("init", "actions", "copy", "transition", "replay"):
        selected = [[row for row in rows if row[0] == engine and row[1] == game and row[2] == operation] for engine in ("rust", "java")]
        assert all(len(group) == 5 * arguments.forks for group in selected)
        assert len({row[5] for group in selected for row in group}) == 1
        rust, java = [[row[6] for row in group] for group in selected]
        r, j = statistics.median(rust), statistics.median(java)
        summaries.append(f"{game}\t{operation}\t{'game' if operation == 'replay' else 'call'}\t{r:.3f}\t{j:.3f}\t{j/r:.3f}\t{min(rust):.3f}\t{max(rust):.3f}\t{min(java):.3f}\t{max(java):.3f}")
out = root / "summary.tsv"
out.write_text("\n".join(summaries)+"\n", encoding="utf-8")
print("\n".join(summaries))
print("Matching corpus counts:", corpora)
print("Corpus SHA256:", hashlib.sha256(Path("tools/java-comparison/benchmark-corpus.tsv").read_bytes()).hexdigest())
