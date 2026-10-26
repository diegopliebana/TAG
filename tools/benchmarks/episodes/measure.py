"""Measure complete individual/bulk learner episodes on CPU using the release extension."""
import argparse
import json
import platform
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import gymnasium
from tag_rust.gym import GameEnv
from tag_rust.gym_batch import GameEnvBatch

ROOT=Path(__file__).resolve().parents[3]


def facts(observation,info):
    return (observation["cells"].tolist(),info["results"].tolist(),tuple(info["opponent_rng"]),
            info["learner_steps"],info["underlying_moves"],info["opponent_seed"])


def workload(game,opponent,count,bulk,*,trace=False):
    envs=[GameEnv(game,learner=i%2,opponent=opponent,payloads=("cells",)) for i in range(count)]
    batch=None
    calls=0
    traces=[[] for _ in envs]
    finals=[None]*count
    try:
        if bulk:
            batch=GameEnvBatch(envs)
            result=batch.reset(seeds=[700+i for i in range(count)])
            if not result.valid.all() or any(result.errors):raise RuntimeError(result.errors)
            active=[True]*count
            if trace:
                for i in range(count):traces[i].append(facts({"cells":result.observations["cells"][i]},result.infos[i]))
            while any(active):
                actions=[int(np.flatnonzero(result.observations["action_mask"][i])[0]) if active[i] else None for i in range(count)]
                result=batch.step(actions);calls+=1
                if any(result.errors):raise RuntimeError(result.errors)
                for i in range(count):
                    if not result.valid[i]:continue
                    observation={"cells":result.observations["cells"][i]}
                    info=result.infos[i]
                    if trace:traces[i].append(facts(observation,info))
                    if result.terminated[i] or result.truncated[i]:
                        active[i]=False;finals[i]=facts(observation,info)
        else:
            for i,env in enumerate(envs):
                observation,info=env.reset(seed=700+i)
                if trace:traces[i].append(facts(observation,info))
                while True:
                    action=int(np.flatnonzero(observation["action_mask"])[0])
                    observation,reward,terminated,truncated,info=env.step(action);calls+=1
                    if trace:traces[i].append(facts(observation,info))
                    if terminated or truncated:
                        finals[i]=facts(observation,info);break
        return calls,finals,traces if trace else None
    finally:
        if batch is not None:batch.close()
        for env in envs:env.close()


def measure(game,opponent,count,samples,seconds):
    a=workload(game,opponent,count,False,trace=True)
    b=workload(game,opponent,count,True,trace=True)
    assert a[1:]==b[1:],"per-transition facts or exact RNG differ"
    baseline=a[1]
    times={"individual":[],"bulk":[]};repetitions={"individual":[],"bulk":[]}
    for sample in range(samples):
        for bulk in ((False,True) if sample%2==0 else (True,False)):
            elapsed=0.;reps=0
            while elapsed<seconds:
                start=time.perf_counter()
                result=workload(game,opponent,count,bulk)
                elapsed+=time.perf_counter()-start;reps+=1
                assert result[1]==baseline,"timed trajectory/checkpoint differs"
            route="bulk" if bulk else "individual"
            times[route].append(elapsed*1000/reps);repetitions[route].append(reps)
    means={k:statistics.mean(v) for k,v in times.items()}
    return dict(game=game,opponent=opponent,environments=count,samples_ms=times,repetitions=repetitions,
                minimum_sample_seconds=seconds,means_ms=means,
                ratio_individual_over_bulk=means["individual"]/means["bulk"],
                step_calls=dict(individual=a[0],bulk=b[0]),
                learner_steps=sum(f[3] for f in baseline),underlying_moves=sum(f[4] for f in baseline))


def render(value):
    from datetime import datetime
    recorded = datetime.fromisoformat(value["recorded"]).strftime("%d-%m-%Y %H:%M")
    lines=["# Individual and bulk learner episodes: CPU measurements","",
           f"Recorded {recorded} (Europe/Paris). Raw samples: [results.json](results.json).","",
           "Times are mean milliseconds per complete workload, including environment construction,",
           "seeded reset, learner action selection, native opponent play, output handling, final",
           "facts/checkpoint extraction and close. Each workload completes the listed number of",
           "independent games. A ratio above 1 means bulk took less time in this run.","",
           "| Game | Rust opponent | Environments | Individual ms | Bulk ms | Individual/bulk | Step calls individual/bulk |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for r in value["records"]:
        lines.append(f"| {r['game']} | {r['opponent']} | {r['environments']} | {r['means_ms']['individual']:.3f} | {r['means_ms']['bulk']:.3f} | {r['ratio_individual_over_bulk']:.2f} | {r['step_calls']['individual']}/{r['step_calls']['bulk']} |")
    lines.extend(["","Seven warmed samples per route normally accumulate at least 200 ms each; raw",
                  "settings/samples record the exact run. Route order alternates. Exact per-transition",
                  "boards, results, counters and opponent RNG checkpoints are compared before timing;",
                  "every timed repetition also checks final facts outside its timed interval.","",
                  "Step calls count collector step entry points, excluding reset/creation/close and",
                  "internal implementation calls. Bulk executes members sequentially in Rust, without",
                  "threads, GPU inference or automatic reset. Scalar-info construction and Python seed",
                  "coordination remain per member. Stacking copies numeric payloads; no zero-copy or",
                  "universal speedup is claimed. Peak memory and component timings are unmeasured.","",
                  "Learners alternate seats, always choose the first legal ID, and use reset seeds",
                  "700+member_index. Payload is cells. Each individual/bulk pair uses the same settings",
                  "and seed pairs; the raw records include total learner steps and underlying moves.",""])
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples",type=int,default=7)
    parser.add_argument("--seconds",type=float,default=.2)
    parser.add_argument("--render",action="store_true",help="Render the saved JSON without rerunning measurements")
    args=parser.parse_args()
    folder=ROOT/"docs/benchmarks/episodes";folder.mkdir(parents=True,exist_ok=True)
    if args.render:
        value=json.loads((folder/"results.json").read_text())
    else:
        if args.samples<1 or args.seconds<=0:parser.error("samples and seconds must be positive")
        from datetime import datetime
        from zoneinfo import ZoneInfo
        # Windows may lack a timezone database; Europe/Paris offset is recorded explicitly for this run.
        try:recorded=datetime.now(ZoneInfo("Europe/Paris")).isoformat()
        except Exception:recorded=datetime.now().astimezone().isoformat()
        value=dict(recorded=recorded,environment=dict(python=sys.version,numpy=np.__version__,gymnasium=gymnasium.__version__,
                   platform=platform.platform(),processor=platform.processor()),
                   settings=dict(samples=args.samples,minimum_sample_seconds=args.seconds,payloads=["cells"],
                                 seed_base=700,learner="member_index % 2",action="first legal ID",max_bytes=67108864),records=[])
        for game in ("tictactoe","connect4"):
            for opponent in ("random","osla_python"):
                for count in (1,2,4,8,16,32,64,128):
                    result=measure(game,opponent,count,args.samples,args.seconds)
                    value["records"].append(result)
                    print(f"{game}/{opponent}/{count}: individual={result['means_ms']['individual']:.3f} ms bulk={result['means_ms']['bulk']:.3f} ms ratio={result['ratio_individual_over_bulk']:.2f}",flush=True)
        (folder/"results.json").write_text(json.dumps(value,indent=2)+"\n")
    (folder/"results.md").write_text(render(value))


if __name__=="__main__":main()
