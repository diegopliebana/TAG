"""Normalize complete-game measurements and prepared-position component probes."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import argparse
import csv


def game_metrics(sample):
    games, actions, turns, rounds, observations = sample["game_work"]
    if games != sample["calls"] or min(games, actions, turns) <= 0:
        raise ValueError("complete-game work counters do not match measured calls")
    return dict(ns_per_game=sample["elapsed_ns"] / games, actions_per_game=actions / games,
                turns_per_game=turns / games, rounds_per_game=rounds / games,
                observations_per_game=observations / games, ns_per_action=sample["elapsed_ns"] / actions,
                ns_per_turn=sample["elapsed_ns"] / turns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    read = lambda name: [json.loads(line) for line in (a.directory / name).read_text().splitlines() if line.strip()]
    all_rows = read("timing.jsonl") + read("workflows-timing.jsonl")
    games = defaultdict(list)
    components = defaultdict(list)
    for row in all_rows:
        if row["operation"] in ("game", "game_direct", "game_aligned", "game_aligned_direct"):
            samples = [game_metrics(s) for s in row["samples"][:-1]]
            games[row["engine"], row["game"], row["players"], row["operation"]].append(
                dict(fork=row["fork"], **{key: statistics.median(s[key] for s in samples) for key in samples[0]}))
        elif "prefix" in row:
            components[row["engine"], row["game"], row["players"], row["operation"], row["prefix"]].append(
                dict(fork=row["fork"], median_ns=statistics.median(s["elapsed_ns"] / s["calls"] for s in row["samples"][:-1]), metadata=row["samples"][-1]))
    output = []
    for key, rows in sorted(games.items()):
        assert len(rows) == 3 and len({r["fork"] for r in rows}) == 3
        output.append(dict(zip(("engine", "game", "players", "operation"), key), forks=rows,
                           **{metric: statistics.median(r[metric] for r in rows) for metric in rows[0] if metric != "fork"}))
    (a.directory / "game-summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (a.directory / "game-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields=[key for key in output[0] if key != "forks"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:r[key] for key in fields} for r in output)
    output = []
    for key, rows in sorted(components.items()):
        assert len(rows) == 3 and len({r["fork"] for r in rows}) == 3
        values = [r["median_ns"] for r in rows]
        output.append(dict(zip(("engine", "game", "players", "operation", "prefix"), key), forks=rows,
                           median_ns=statistics.median(values), min_fork_ns=min(values), max_fork_ns=max(values)))
    (a.directory / "component-summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (a.directory / "component-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields=[key for key in output[0] if key != "forks"]+['actual_prepared_actions']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in output:
            values={key:row[key] for key in fields if key!='actual_prepared_actions'}
            values['actual_prepared_actions']='|'.join(map(str,sorted({f['metadata']['prepared_actions'] for f in row['forks']})))
            writer.writerow(values)
    print("Wrote measured game-work and prepared-position component summaries")


if __name__ == "__main__": main()
