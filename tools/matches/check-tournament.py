"""Check tournament CLI results and independently recalculate both ranking metrics.

Build the release tournament example with features mcts-json,replay first.
Only Python's standard library is required.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "target/release/examples" / ("tournament.exe" if os.name == "nt" else "tournament")


def verify(report):
    """Calculate rates directly from each recorded seat outcome, independently of Rust."""
    per_game = []
    for entry, game in enumerate(report["game_entries"]):
        counts = [[0, 0, 0] for _ in report["agents"]]
        for record in report["games"]:
            if record["entry_index"] != entry or record["outcome"] is None:
                continue
            assert record["points"] == [{"win": 1.0, "draw": .5, "loss": 0.0}[r]
                                        for r in record["outcome"]["results"]]
            for agent, result in zip(record["agents_by_seat"], record["outcome"]["results"]):
                counts[agent][{"win": 0, "draw": 1, "loss": 2}[result]] += 1
        rates = []
        for agent, (wins, draws, losses) in zip(report["agents"], counts):
            played = wins + draws + losses
            expected = (wins / played, (wins + .5 * draws) / played) if played else None
            row = next(row for row in report["rankings"]["per_game"][entry]["rows"] if row["agent"] == agent["name"])
            assert row["appearances"] == played
            if expected is None:
                assert row["rank"] is None and row["win_rate"] is None
            else:
                assert math.isclose(row["win_rate"], expected[0], abs_tol=1e-12)
                assert math.isclose(row["points_rate"], expected[1], abs_tol=1e-12)
            rates.append(expected)
        per_game.append(rates)
        assert report["rankings"]["per_game"][entry]["game_id"] == game["id"]
    for index, agent in enumerate(report["agents"]):
        observed = [rates[index] for rates in per_game if rates[index] is not None]
        row = next(row for row in report["rankings"]["global"] if row["agent"] == agent["name"])
        assert row["games_covered"] == len(observed)
        if observed:
            for column, key in enumerate(("win_rate", "points_rate")):
                assert math.isclose(row[key], sum(r[column] for r in observed) / len(observed), abs_tol=1e-12)
    key = report["rankings"]["sort_by"]
    for rows in [r["rows"] for r in report["rankings"]["per_game"]] + [report["rankings"]["global"]]:
        values = [row[key] for row in rows if row[key] is not None]
        assert values == sorted(values, reverse=True)
        previous, rank = None, None
        for position, row in enumerate(rows, 1):
            if row[key] is None:
                assert row["rank"] is None
            else:
                if row[key] != previous:
                    rank = position
                assert row["rank"] == rank
                previous = row[key]


def main():
    with tempfile.TemporaryDirectory(prefix="tournament-cli-", dir=ROOT / "target") as temporary:
        folder = Path(temporary)
        (folder / "elsewhere").mkdir()
        (folder / "agent.json").write_text('{"schema_version":1,"budgets":{"iterations":5}}', encoding="utf-8")
        config = {"schema_version": 1, "agents": [
            {"name": "random", "agent": {"type": "random"}},
            {"name": "osla", "agent": {"type": "osla"}},
            {"name": "mcts", "agent": {"type": "mcts", "config_file": "agent.json"}}],
            "games": [{"id": "ttt", "name": "tictactoe"}, {"id": "c4", "name": "connect4"}],
            "mode": "random", "matchups": 5, "seed": 42, "game_logs": {"mode": "inline"},
            "results_file": "results.json"}
        path = folder / "tournament.json"
        for mode, seats, expected in [("random", "permutations", 20), ("random", "random", 10),
                                      ("exhaustive", "permutations", 12), ("exhaustive", "random", 6)]:
            config["mode"], config["seat_allocation"] = mode, seats
            if mode == "random":
                config["matchups"] = 5
            else:
                config.pop("matchups", None)
            for metric in ("win_rate", "points_rate"):
                config["ranking_metric"] = metric
                destination = folder / f"{mode}-{seats}-{metric}.json"
                config["results_file"] = destination.name
                path.write_text(json.dumps(config), encoding="utf-8")
                result = subprocess.run([str(BINARY), str(path)], cwd=folder / "elsewhere", capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                report = json.loads(destination.read_text())
                assert report["completed_games"] == expected == len(report["games"])
                verify(report)
                assert "Combined results across 2 game entries" in result.stdout
                assert "Game entry 'ttt': tictactoe, 2 players" in result.stdout
                assert "Game entry 'c4': connect4, 2 players" in result.stdout
                assert "Results:" in result.stdout
                original = destination.read_bytes()
                refused = subprocess.run([str(BINARY), str(path)], capture_output=True, text=True)
                assert refused.returncode != 0 and not refused.stdout
                assert destination.read_bytes() == original
        config["mode"] = "exhaustive"
        config["agents"][2]["agent"] = {"type": "mcts", "config": {
            "schema_version": 1, "tree_policy": "puct", "budgets": {"iterations": 30},
            "puct_preferences": {"source": "probabilities", "evaluator": {"name": "uniform_actions"}}}}
        config["seat_allocation"] = "permutations"
        config["results_file"] = "partial.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        failed = subprocess.run([str(BINARY), str(path)], cwd=folder, capture_output=True, text=True)
        assert failed.returncode != 0
        partial = json.loads((folder / "partial.json").read_text())
        assert partial["completed_games"] == 2 and len(partial["games"]) == 3
        assert partial["games"][-1]["outcome"] is None and partial["failure"]
        verify(partial)
        # Exercise the supplied examples with their real MCTS files, keeping only
        # generated result destinations and temporary configs outside examples/.
        for name, expected in (("exhaustive", 12), ("random", 20)):
            original = ROOT / "examples/tournaments" / f"{name}.json"
            supplied = json.loads(original.read_text())
            for agent in supplied["agents"]:
                if "config_file" in agent["agent"]:
                    agent["agent"]["config_file"] = str((original.parent / agent["agent"]["config_file"]).resolve())
            destination = folder / f"supplied-{name}.json"
            supplied["results_file"] = destination.name
            path.write_text(json.dumps(supplied), encoding="utf-8")
            result = subprocess.run([str(BINARY), str(path)], cwd=folder, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            report = json.loads(destination.read_text())
            assert report["completed_games"] == expected
            verify(report)
            if name == "random":
                assert "Outcome points=" in result.stdout
                assert "games played=" in result.stdout
                assert "Combined results across 2 game entries" in result.stdout
                assert report["tournament_id"]
                for record in report["games"]:
                    assert "actions" not in record["outcome"]
                    metadata_path = destination.parent / record["game_log"]
                    metadata = json.loads(metadata_path.read_text())
                    replay_path = destination.parent / record["replay_file"]
                    assert replay_path.resolve() == (metadata_path.parent / metadata["replay_file"]).resolve()
                    replay = json.loads(replay_path.read_text())
                    assert metadata["tournament_id"] == report["tournament_id"]
                    assert metadata["record"]["index"] == record["index"]
                    assert len(replay["branches"][0]["moves"]) == record["outcome"]["accepted_moves"]
        print("Tournament CLI checks pass: eight schedule/metric combinations, independent rankings, "
              "required files, raw results, another cwd, overwrite protection, saved partial failures "
              "and both current examples, including separate browser replay files.")


if __name__ == "__main__":
    main()
