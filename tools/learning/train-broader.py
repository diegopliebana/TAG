"""Compare equal-budget OSLA and Random opposition using existing training APIs.

Runs default Tic-Tac-Toe with three training seeds. Existing artifacts are left
alone. Each finished run records its own model, settings and held-out outcomes.
This is an experiment driver, not a new production training algorithm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import torch
from tag_rust.learning.artifacts import load_checkpoint, save_checkpoint, save_model
from tag_rust.learning.evaluation import evaluate_model
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.tabular import QTable, run
from tag_rust.learning.training import TrainingSession, TrainingSettings

diagnostics = runpy.run_path(str(Path(__file__).with_name('evaluate-strength.py')))


def write(path, value):
    """Save progress after a completed run; full training checkpoints are separate."""
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def q_evaluation(table, seed, opponent, episodes):
    original=table.choose
    coverage=dict(decisions=0,unseen=0)
    def choose(observation,info,rng,*,epsilon):
        coverage['decisions']+=1
        coverage['unseen']+=table._key(observation) not in table.values
        return original(observation,info,rng,epsilon=epsilon)
    table.choose=choose
    try:
        counts=run(table,rounds=episodes,members=1,seed=seed,epsilon=0,learning=False,opponent=opponent)
    finally:
        table.choose=original
    return dict(seed=seed,opponent=opponent,coverage=coverage,**counts)


def select_candidates(manifest, destination):
    """Select using validation only and copy artifacts without replacing old files."""
    destination.mkdir(parents=True,exist_ok=True)
    selected={}
    for kind in ('neural','q'):
        def rank(report):
            rows=[r for r in report['evaluations'] if kind=='q' or r.get('mode')=='osla']
            counts=[r.get('combined') or r for r in rows]
            score=sum(r['wins']+.5*r['draws'] for r in counts)/sum(r['episodes'] for r in counts)
            tie=report['rows'] if kind=='q' else -report['position_diagnostics']['missed_forced_blocks']
            return score,tie
        best=max((r for r in manifest['runs'] if r['kind']==kind),key=rank)
        source=Path(best['model'])
        model=destination/f'tictactoe-{kind}-candidate.json'
        copies=[(source,model)]
        if kind=='neural':copies.append((source.parent/'training.pt',model.with_suffix('.pt')))
        for original,target in copies:
            data=original.read_bytes()
            if target.exists() and target.read_bytes()!=data:
                raise ValueError(f'{target} already contains a different artifact; choose --models-output')
            target.write_bytes(data)
        selected[kind]=dict(run=best['name'],model=str(model.resolve()),
            validation_score=rank(best)[0],sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('target/m7-broader-training'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--models-output',type=Path,default=Path('target/models'))
    args=parser.parse_args()
    if any(seed<0 for seed in args.seeds) or len(set(args.seeds))!=len(args.seeds):
        parser.error('training seeds must be distinct and nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    validation_seeds=(90011,90012)
    manifest=dict(game='tictactoe',training_seeds=args.seeds,validation_seeds=validation_seeds,
        neural=dict(teacher_episodes=512,actor_episodes=4096,group_size=16,hidden_sizes=[64,64],
                    entropy_weight=.01,learning_rate=.001),
        q=dict(episodes=16384,chunk_episodes=2048,epsilon_start=.4,epsilon_end=.1,alpha=.2,gamma=.99),
        opposition=['osla','random'],runs=[])
    torch.set_num_threads(1)
    for kind in ('neural','q'):
        for opponent in ('osla','random'):
            for seed in args.seeds:
                name=f'{kind}-{opponent}-seed{seed}'
                folder=args.output/name
                folder.mkdir(exist_ok=True)
                report_path=folder/'report.json'
                if report_path.exists():
                    report=json.loads(report_path.read_text(encoding='utf-8'))
                    print(f'Reusing completed run {name}',flush=True)
                elif kind=='neural':
                    settings=TrainingSettings(teacher_episodes=512,actor_episodes=4096,
                        master_seed=seed,teacher_opponent=opponent,actor_opponent=opponent)
                    checkpoint=folder/'training.pt'
                    session=load_checkpoint(checkpoint) if checkpoint.exists() else TrainingSession(settings)
                    if session.settings!=settings:
                        raise ValueError(f'checkpoint settings differ for {name}; choose a new output directory')
                    print(f'Training {name}',flush=True)
                    for update in session.run():
                        if update['update']%64==0:
                            save_checkpoint(checkpoint,session)
                            print(f'{name}: {session.next_episode_id}/4608 episodes',flush=True)
                    save_checkpoint(checkpoint,session)
                    model_path=folder/'model.json'
                    save_model(model_path,session.model,provenance=session.provenance())
                    snapshot=FrozenModel(session.model)
                    evaluations=[]
                    for evaluation_seed in validation_seeds:
                        for adversary in ('random','osla'):
                            evaluations.append(evaluate_model(session.model,episodes=100,seed=evaluation_seed,
                                                              mode='policy',opponent=adversary))
                            osla,_=diagnostics['evaluate_osla'](snapshot,adversary,evaluation_seed,100)
                            evaluations.append(osla)
                    tactics,_=diagnostics['inspect_positions'](snapshot)
                    report=dict(name=name,kind=kind,training_opponent=opponent,training_seed=seed,
                        model=str(model_path.resolve()),model_identity=snapshot.identity,
                        provenance=session.provenance(),evaluations=evaluations,position_diagnostics=tactics)
                    write(report_path,report)
                else:
                    task=dict(game='tictactoe',parameters={},learner=None,opponent=opponent,
                              opponent_settings={},max_episode_steps=None)
                    table=QTable(task)
                    training=[]
                    for chunk in range(8):
                        epsilon=.4-.3*chunk/7
                        stream_seed=1_000_000+seed*10_000+chunk
                        counts=run(table,rounds=128,members=16,seed=stream_seed,epsilon=epsilon,learning=True)
                        training.append(dict(chunk=chunk,seed=stream_seed,epsilon=epsilon,**counts))
                        print(f'{name}: {(chunk+1)*2048}/16384 episodes; {len(table.values)} rows',flush=True)
                    model_path=folder/'model.json'
                    table.save(model_path)
                    reloaded=QTable.load(model_path)
                    evaluations=[q_evaluation(reloaded,s,adversary,100)
                                 for s in validation_seeds for adversary in ('random','osla')]
                    report=dict(name=name,kind=kind,training_opponent=opponent,training_seed=seed,
                        model=str(model_path.resolve()),rows=len(table.values),task=table.task,
                        training=training,evaluations=evaluations)
                    write(report_path,report)
                manifest['runs'].append(report)
                write(args.output/'experiment.json',manifest)
                print(f'Completed {name}',flush=True)
    manifest['selection']=select_candidates(manifest,args.models_output)
    write(args.output/'experiment.json',manifest)
    print(json.dumps(manifest['selection'],indent=2),flush=True)
    print(f'Completed comparison: {args.output / "experiment.json"}',flush=True)


if __name__=='__main__':main()
