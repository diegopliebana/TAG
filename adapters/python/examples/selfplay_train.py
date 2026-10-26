"""Generate neural-MCTS self-play, train policy/value targets, save and evaluate."""
import argparse
import json
from pathlib import Path
from tag_rust.learning.selfplay import SelfPlaySettings,SelfPlaySession,load_checkpoint,save_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,help="self-play JSON; default examples/selfplay/tictactoe.json")
    parser.add_argument("--resume",type=Path,help="completed-cycle self-play checkpoint")
    parser.add_argument("--model",type=Path,help="compatible gamma=1 evaluation model for initialization")
    parser.add_argument("--max-cycles",type=int,help="stop after this many additional complete cycles")
    parser.add_argument("--checkpoint-limit-mib",type=int,default=512,help="maximum checkpoint size accepted when resuming")
    args = parser.parse_args()
    try:
        if args.resume:
            if args.config is not None or args.model is not None:
                raise ValueError("resume uses checkpoint settings/model; omit --config and --model")
            session = load_checkpoint(args.resume,max_bytes=args.checkpoint_limit_mib*1024*1024)
        else:
            path = args.config or Path(__file__).resolve().parents[3]/"examples/selfplay/tictactoe.json"
            settings = SelfPlaySettings.load(path)
            initial_model = None
            if args.model:
                from tag_rust.learning.artifacts import load_model
                initial_model = load_model(args.model)
            session = SelfPlaySession(settings,initial_model=initial_model)
        if session.settings.game in ("cantstop", "loveletter"):
            print(json.dumps({"master_seed":session.settings.seed,"seed_derivation_version":2,
                              "completed_cycles":session.completed_cycles}),flush=True)
        for report in session.run(max_cycles=args.max_cycles):
            save_outputs(session)
            losses = report["optimization"]
            summary = {"cycle":report["cycle"],"model_identity":report["model_identity"],
                "buffer_positions":report["buffer_positions"],"loss":sum(row["loss"] for row in losses)/len(losses),
                "evaluation":report["evaluation"]}
            if "per_seat" in report:
                summary.update({name:report[name] for name in ("games","per_seat","combined")})
            print(json.dumps(summary,allow_nan=False),flush=True)
        save_outputs(session)
        print(json.dumps({"completed_cycles":session.completed_cycles,"master_seed":session.settings.seed,
                          "output":session.settings.document()["output"]},indent=2))
    except (ValueError,RuntimeError,OSError,MemoryError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
