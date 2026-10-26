"""Measure neural-pipeline CPU components and complete workloads using the release Python adapter.

Run from the repository root. Five warmed samples each run for at least 0.08 s.
Components overlap and must not be added to predict complete-workload timings.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import tag_rust as tag
from tag_rust.learning.collection import collect_episode_group
from tag_rust.learning.inputs import pack_observations
from tag_rust.learning.network import _tensor
from tag_rust.learning.play import NeuralValueEvaluator
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.training import TrainingSession, TrainingSettings, make_targets, loss_terms, runtime_identity


def measure(operation, samples, seconds):
    operation()  # Warm dispatch, allocation and kernels before recording.
    rows=[]
    for _ in range(samples):
        start=time.perf_counter_ns()
        count=0
        while (time.perf_counter_ns()-start)/1e9<seconds:
            operation()
            count+=1
        rows.append(dict(seconds=(time.perf_counter_ns()-start)/1e9/count,iterations=count))
    values=[r['seconds'] for r in rows]
    return dict(median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values),samples=rows)


def play(evaluator, *, neural):
    """Profile the public manual request path, or the Rust-owned heuristic baseline."""
    totals=dict(prepare_seconds=0.,evaluate_seconds=0.,finish_seconds=0.,commit_seconds=0.,requests=0,candidate_rows=0,moves=0)
    with tag.create(evaluator.snapshot.contract.identity.game) as game:
        for seat in (0,1):
            game.set_controller(seat,evaluator.controller(seed=700+seat) if neural else tag.OSLAPython.rust_heuristic(seed=700+seat))
        if neural:
            while game.next_player is not None:
                start=time.perf_counter();request=game.prepare_evaluation()
                totals['prepare_seconds']+=time.perf_counter()-start
                while isinstance(request,tag.EvaluationRequest):
                    totals['requests']+=1;totals['candidate_rows']+=len(request.action_ids)
                    start=time.perf_counter();values=evaluator.evaluate(request)
                    totals['evaluate_seconds']+=time.perf_counter()-start
                    start=time.perf_counter();request=game.finish_evaluation(request,values)
                    totals['finish_seconds']+=time.perf_counter()-start
                start=time.perf_counter();game.commit_evaluation(request)
                totals['commit_seconds']+=time.perf_counter()-start
                totals['moves']+=1
        else:
            stop=game.advance()
            assert stop.reason=='finished'
            totals['moves']=stop.moves
        totals['results']=game.observe(0,payloads=()).results.tolist()
    return totals


def full_training(game):
    session=TrainingSession(TrainingSettings(game=game,teacher_episodes=16,actor_episodes=16,master_seed=700))
    list(session.run())
    assert session.phase=='complete' and session.updates==2
    return dict(episodes=session.next_episode_id,updates=session.updates,
                outcomes=[report['outcomes'] for report in session.history])


def benchmark(game_name,args):
    session=TrainingSession(TrainingSettings(game=game_name,teacher_episodes=16,actor_episodes=16,master_seed=700))
    model=session.model.eval();snapshot=FrozenModel(model);evaluator=NeuralValueEvaluator(snapshot)
    result=dict(game=game_name,model_identity=snapshot.identity,parameter_count=sum(p.numel() for p in model.parameters()),batches=[])
    timing=lambda f:measure(f,args.samples,args.seconds)
    with tag.create(game_name) as game:
        for actor,action in enumerate((0,1)):
            game.apply(action,position=game.observe(actor,payloads=()).position,actor=actor)
        observation=game.observe(0,payloads=('cells',))
        result['prefix']=[0,1]
        def simulate():
            with game.clone() as branch:
                current=branch.observe(0,payloads=())
                branch.apply(int(current.offered_action_ids[0]),position=current.position,actor=0)
        result['clone_observe_apply_python_bridge']=timing(simulate)
        result['frozen_snapshot_create']=timing(lambda:FrozenModel(model))
        for count in (1,16,64):
            observations=[observation]*count
            batch=pack_observations(snapshot.contract,observations)
            payloads={k:_tensor(v) for k,v in batch.payloads.items()};context=_tensor(batch.context)
            with torch.inference_mode():logits,values=model(payloads,context)
            probabilities=torch.softmax(logits,dim=1)
            def forward():
                with torch.inference_mode():return model(payloads,context)
            def input_copy():return {k:_tensor(v) for k,v in batch.payloads.items()},_tensor(batch.context)
            def outputs():return logits.numpy().copy(),values.numpy().copy(),probabilities.numpy().copy()
            row=dict(rows=count,
                owned_batch_bytes=sum(v.nbytes for v in batch.payloads.values())+batch.context.nbytes+batch.legal_mask.nbytes+batch.action_ids.nbytes,
                tensor_input_copy_bytes=sum(v.nbytes for v in batch.payloads.values())+batch.context.nbytes,
                prediction_float_copy_bytes=4*count*(2*snapshot.contract.actions.size+1),
                float64_reply_bytes=8*count)
            row['encode_export_observations']=timing(lambda:[game.observe(0,payloads=('cells',)) for _ in range(count)])
            row['pack_owned_observations']=timing(lambda:pack_observations(snapshot.contract,observations))
            row['copy_inputs_to_cpu_tensors']=timing(input_copy)
            row['forward_prepared_cpu_tensors']=timing(forward)
            row['copy_float_outputs_to_numpy']=timing(outputs)
            row['copy_values_float64']=timing(lambda:values.numpy().astype(np.float64,copy=True))
            row['checked_frozen_predict']=timing(lambda:snapshot.predict(batch))
            assert not np.shares_memory(batch.payloads['cells'],observation.payloads['cells'])
            assert not np.shares_memory(payloads['cells'].numpy(),batch.payloads['cells'])
            prediction=snapshot.predict(batch)
            assert not np.shares_memory(prediction.logits,logits.numpy())
            result['batches'].append(row)
            print(f'{game_name}: measured batch {count}',flush=True)
    group=collect_episode_group(game_name,episode_count=16,master_seed=700,phase='teacher',payloads=('cells',))
    targets=make_targets(group,snapshot)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
    def update():
        optimizer.zero_grad(set_to_none=True)
        terms=loss_terms(model,targets,phase='teacher',settings=session.settings)
        assert torch.isfinite(terms['total'])
        terms['total'].backward();optimizer.step()
    result['teacher_loss_backward_adam_fixed_group']=timing(update)
    result['teacher_target_decisions']=len(targets.actions)
    result['full_neural_game']=timing(lambda:play(evaluator,neural=True))
    result['profiled_neural_game']=play(evaluator,neural=True)
    result['full_rust_heuristic_game']=timing(lambda:play(evaluator,neural=False))
    result['rust_heuristic_game_counts']=play(evaluator,neural=False)
    def collect():
        return collect_episode_group(game_name,episode_count=16,master_seed=700,phase='evaluation',
                                     snapshot=snapshot,opponent='random',payloads=('cells',))
    result['policy_collection_16']=timing(collect)
    result['policy_collection_counts']=collect().outcomes
    result['fresh_training_32']=timing(lambda:full_training(game_name))
    result['fresh_training_counts']=full_training(game_name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('target/m7-s5/measurements.json'))
    parser.add_argument('--samples',type=int,default=5)
    parser.add_argument('--seconds',type=float,default=.08)
    args=parser.parse_args()
    if args.samples<3 or not np.isfinite(args.seconds) or args.seconds<=0:parser.error('use at least three samples and positive seconds')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    result=dict(runtime=runtime_identity(),executable=sys.executable,cpu=platform.processor(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
        cuda_available=torch.cuda.is_available(),device='cpu',gpu_transfers='none; not measured',
        clock='perf_counter_ns',samples=args.samples,minimum_sample_seconds=args.seconds,games=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for game in ('tictactoe','connect4'):
        result['games'].append(benchmark(game,args))
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {args.output}',flush=True)


if __name__=='__main__':main()
