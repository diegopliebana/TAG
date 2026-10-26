"""Check the compiled match CLI, including exit codes, files and another cwd.

Build first with cargo build --offline --release --features mcts-json,replay
--example match_play. This script uses only Python's standard library.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "target" / "release" / "examples" / (
    "match_play.exe" if os.name == "nt" else "match_play")


def invoke(path, cwd):
    return subprocess.run([str(BINARY), str(path)], cwd=cwd, capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory(prefix="match-cli-", dir=ROOT / "target") as temporary:
        folder = Path(temporary)
        (folder / "agents").mkdir()
        (folder / "another-cwd").mkdir()
        (folder / "agents" / "mcts.json").write_text(json.dumps({
            "schema_version": 1, "budgets": {"iterations": 8}, "rollout_length": 2}), encoding="utf-8")
        basic = {"type": "random"}
        osla = {"type": "osla"}
        first = {"type": "first_available"}
        search = {"type": "mcts", "config_file": "agents/mcts.json"}
        configurations = [basic, osla, first, search]
        executed = 0
        for game in ("tictactoe", "connect4"):
            for agent in configurations:
                value = {"schema_version": 1, "game": {"name": game}, "games": 3, "seed": 42,
                         "players": [{"name": "one", "agent": agent}, {"name": "two", "agent": osla}],
                         "reporting": {"format": "json", "include_actions": True}}
                path = folder / "match.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                completed = invoke(path, folder / "another-cwd")
                assert completed.returncode == 0, completed.stderr
                report = json.loads(completed.stdout)
                assert report["completed_games"] == 3 and report["failure"] is None
                assert [row["seats"] for row in report["games"]] == [[0, 1], [1, 0], [0, 1]]
                repeat = invoke(path, folder)
                assert repeat.returncode == 0 and json.loads(repeat.stdout) == report
                executed += 1

        value["players"][0]["agent"] = basic
        value["reporting"]["path"] = "report.json"
        value["reporting"]["recordings"] = "replays"
        path.write_text(json.dumps(value), encoding="utf-8")
        completed = invoke(path, folder)
        assert completed.returncode == 0, completed.stderr
        assert json.loads((folder / "report.json").read_text()) == json.loads(completed.stdout)
        assert len(list((folder / "replays").glob("game-*.json"))) == 3
        original = (folder / "report.json").read_bytes()
        refused = invoke(path, folder)
        assert refused.returncode != 0 and not refused.stdout
        assert (folder / "report.json").read_bytes() == original

        value["reporting"] = {"format": "json"}
        value["players"][0]["agent"] = {"type": "mcts", "config": {
            "schema_version": 1, "tree_policy": "puct", "budgets": {"iterations": 20},
            "puct_preferences": {"source": "probabilities", "evaluator": {"name": "uniform_actions"}}}}
        path.write_text(json.dumps(value), encoding="utf-8")
        failed = invoke(path, folder)
        assert failed.returncode != 0, failed.stdout
        report = json.loads(failed.stdout)
        assert report["completed_games"] == 0 and report["failure"]["game_index"] == 0
        assert all(agent["combined"]["losses"] == 0 for agent in report["agents"])

        value["players"][0]["agent"] = {"type": "unknown"}
        path.write_text(json.dumps(value), encoding="utf-8")
        invalid = invoke(path, folder)
        assert invalid.returncode != 0 and not invalid.stdout
        usage = subprocess.run([str(BINARY)], cwd=folder, capture_output=True, text=True)
        assert usage.returncode != 0 and "usage:" in usage.stderr
        print(f"CLI checks pass: {executed} repeated game/agent configurations, relative paths, "
              "report/replay files, overwrite rejection and runtime/configuration/usage errors.")


if __name__ == "__main__":
    main()
