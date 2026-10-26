"""Regenerate small CPU reference models and predictions for local MCTS tests."""
import json
from pathlib import Path
import torch
from tag_rust.engine.api import create
from tag_rust.learning.artifacts import model_document
from tag_rust.learning.inputs import ModelContract, ValueSpec, pack_observations
from tag_rust.learning.network import PolicyValueNetwork


def main():
    destination = Path(__file__).resolve().parents[3] / "web/tests/fixtures/mcts"
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    torch.set_num_threads(1)
    for game_name in ("tictactoe", "connect4"):
        for order in (None, (1, 0)):
            torch.manual_seed(51)
            with create(game_name) as game:
                contract = ModelContract.from_spec(game.spec(), payloads=("cells",),
                    value=ValueSpec(players=order, gamma=1., discount_clock="environment-action"))
                model = PolicyValueNetwork(contract, (4,))
                context = {"observers": [0, 1], "turn_context": "any_turn" if order is None else "own_turn"}
                document = model_document(model, provenance={"algorithm": "mcts-selfplay", "version": 1,
                    "model_context": context})
                filename = f"{game_name}-{'scalar' if order is None else 'vector'}.json"
                (destination / filename).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                rows = []
                for prefix in ([], [0], [0, 1], [0, 1, 3]):
                    game.reset()
                    for action in prefix:
                        observation = game.observe(game.next_player, ())
                        game.apply(action, position=observation.position)
                    for observer in (0, 1):
                        batch = pack_observations(contract, [game.observe(observer, ("cells",))])
                        with torch.no_grad():
                            logits, values = model({name: torch.from_numpy(a.copy()) for name, a in batch.payloads.items()},
                                torch.from_numpy(batch.context.copy()))
                        rows.append({"prefix": prefix, "observer": observer, "inputs": batch.inputs[0].tolist(),
                            "logits": logits[0].tolist(), "values": values[0].reshape(-1).tolist()})
                cases.append({"game": game_name, "model_file": filename, "rows": rows})
    (destination / "predictions.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    export_cantstop(destination)


def export_cantstop(destination):
    """Reference vectors for each supported player count and every game phase.

    Short columns keep complete browser matches small; default-rule model
    compatibility is covered by the separately recorded self-play export.
    """
    cases = []
    for players in (2, 3, 4):
        torch.manual_seed(51)
        parameters = {"seed": 18446744073709551615,
                      "rules": {"column_heights": [3] * 11, "columns_to_win": 1}}
        with create("cantstop", parameters, players=players) as game:
            payloads = ("permanent", "temporary", "owners", "dice", "context")
            contract = ModelContract.from_spec(game.spec(), payloads=payloads,
                value=ValueSpec(players=tuple(range(players)), gamma=1., discount_clock="environment-action"))
            model = PolicyValueNetwork(contract, (4,))
            document = model_document(model, provenance={"algorithm": "mcts-selfplay", "version": 1,
                "model_context": {"observers": list(range(players)), "turn_context": "own_turn"}})
            filename = f"cantstop-{players}.json"
            (destination / filename).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            rows = []
            prefix = []
            # Initial, rolled, allocated, stopped, then another player's Roll.
            for action_id in (None, 0, "first_allocation", 1, 0):
                if action_id is not None:
                    observation = game.observe(game.next_player, ())
                    selected = observation.offered_action_ids[0] if action_id == "first_allocation" else action_id
                    game.apply(int(selected), position=observation.position)
                    prefix.append(int(selected))
                observer = game.next_player
                batch = pack_observations(contract, [game.observe(observer, payloads)])
                with torch.no_grad():
                    logits, values = model({name: torch.from_numpy(a.copy()) for name, a in batch.payloads.items()},
                                          torch.from_numpy(batch.context.copy()))
                rows.append({"prefix": list(prefix), "observer": observer, "inputs": batch.inputs[0].tolist(),
                             "logits": logits[0].tolist(), "values": values[0].tolist()})
            cases.append({"players": players, "parameters": parameters, "model_file": filename, "rows": rows})
    (destination / "cantstop-predictions.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
