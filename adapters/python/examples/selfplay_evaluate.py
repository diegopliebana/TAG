"""Evaluate a saved frozen self-play model against configured native opponents."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tag_rust import _engine
from tag_rust.learning.artifacts import load_model,_atomic_write
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.selfplay import SelfPlaySettings,evaluate_model
from tag_rust.learning.selfplay.settings import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True,help="portable policy/value model JSON")
    parser.add_argument("--config",type=Path,required=True,help="evaluation configuration JSON")
    args = parser.parse_args()
    try:
        model = load_model(args.model)
        snapshot = FrozenModel(model)
        configuration = read_json(args.config)
        if not isinstance(configuration,dict):
            raise ValueError("evaluation configuration must be an object")
        # The saved model already declares its inputs and value output; don't ask
        # the experiment file to duplicate those numeric descriptions.
        configuration.setdefault("game",model.contract.identity.game)
        configuration.setdefault("players",model.contract.identity.player_count)
        configuration.setdefault("model",{"payloads":[t.name for t in model.contract.tensors],
            "hidden_sizes":list(model.hidden_sizes),
            "value_output":"observer" if model.contract.value.players is None else "players",
            "value_players":model.contract.value.players})
        settings = SelfPlaySettings.from_dict(configuration,directory=args.config.parent)
        if not settings.evaluation.enabled:
            raise ValueError("standalone evaluation requires evaluation.enabled=true")
        if settings.output.model_file is not None or settings.output.checkpoint_file is not None:
            raise ValueError("standalone evaluation writes only output.report_file")
        if settings.output.report_file is not None and Path(settings.output.report_file).resolve() in (
            args.model.resolve(),args.config.resolve()
        ):
            raise ValueError("evaluation report must differ from model and configuration files")
        if settings.seed is None:
            settings = replace(settings,seed=_engine.automatic_search_seed())
        print(json.dumps({"master_seed":settings.seed,"model_identity":snapshot.identity}),flush=True)
        result = evaluate_model(settings,snapshot)
        document = {"format":"tag-mcts-model-evaluation","version":1,"settings":settings.document(),"evaluation":result}
        if settings.output.report_file is not None:
            encoded = json.dumps(document,indent=2,allow_nan=False).encode("utf-8")
            _atomic_write(settings.output.report_file,lambda stream:stream.write(encoded))
        print(json.dumps(result,allow_nan=False))
        if result.get("failures"):
            raise SystemExit(1)
    except (ValueError,RuntimeError,OSError,MemoryError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
