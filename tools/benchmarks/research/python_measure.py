"""Measure Python/Rust transfers, independent-tree search and CPU model batches.

Uses deterministic untrained networks: timings describe execution, not strength.
Each invocation is one fresh-process repetition. Run after other timing probes.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import tag_rust as tag
from tag_rust.learning.inputs import ModelContract,ValueSpec,pack_observations
from tag_rust.learning.network import PolicyValueNetwork
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.artifacts import save_model
from tag_rust.search.neural_mcts import NeuralMctsEvaluator
from run import PROFILES,process_memory
from collections import deque
from tag_rust.learning.play import NeuralValueEvaluator
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"adapters/python/examples"))
from open_lines import evaluate_open_lines

PAYLOADS={'tictactoe':('cells',),'connect4':('cells',),'cantstop':('permanent','temporary','owners','dice','context'),'loveletter':('hands','players','reserve','discards','context')}

def measure(fn, *, minimum=.05):
    fn() # Trigger dispatch/import/cache preparation before recorded samples.
    samples=[]
    for _ in range(5):
        start=time.perf_counter_ns();calls=0;last=None
        while calls==0 or (time.perf_counter_ns()-start)/1e9<minimum:
            last=fn();calls+=1
        samples.append(dict(calls=calls,elapsed_ns=time.perf_counter_ns()-start))
    return samples,last

def create(game,players,seat=0):
    params={'seed':42} if game in ('cantstop','loveletter') else {}
    if game=='loveletter':params['first_player']=seat
    result=tag.create(game,params,players=players)
    if game!='loveletter':
        for _ in range(256):
            if result.next_player==seat:break
            o=result.observe(result.next_player,());result.apply(int(o.offered_action_ids[0]),position=o.position)
        assert result.next_player==seat
    if game=='cantstop':
        o=result.observe(seat,());result.apply(int(o.offered_action_ids[0]),position=o.position)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork',type=int,required=True)
    parser.add_argument('--game',choices=['tictactoe','connect4','cantstop','loveletter'])
    parser.add_argument('--players',type=int,choices=[2,3,4])
    parser.add_argument('--output',type=Path,default=Path('docs/benchmarks/research'))
    parser.add_argument('--models',type=Path,default=Path('.scratch/research-bench/models'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.models.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    path=args.output/f'python-{args.fork}.jsonl'
    if path.exists():parser.error('output exists')
    with path.open('w',encoding='utf-8') as stream:
        for game,players in PROFILES:
            if args.game and game!=args.game:continue
            if args.players and players!=args.players:continue
            with create(game,players) as root:
                torch.manual_seed(42)
                contract=ModelContract.from_spec(root.spec(),payloads=PAYLOADS[game],value=ValueSpec(players=tuple(range(players)),gamma=1,discount_clock='environment-action'))
                network=PolicyValueNetwork(contract,(16,));snapshot=FrozenModel(network);evaluator=NeuralMctsEvaluator(snapshot,turn_context="own_turn")
                model_path=args.models/f'{game}-{players}.json'
                save_model(model_path,network,provenance={'scope':'deterministic untrained performance fixture','seed':42,'model_context':{'observers':list(range(players)),'turn_context':'own_turn'}})
                common=dict(game=game,players=players,fork=args.fork,seed=42,model_identity=snapshot.identity,parameter_count=sum(p.numel() for p in network.parameters()),model_weight_bytes=sum(p.numel()*p.element_size() for p in network.parameters()),payloads=PAYLOADS[game])
                def record(op,fn,**settings):
                    samples,last=measure(fn);stream.write(json.dumps(dict(common,operation=op,samples=samples,**settings))+'\n');stream.flush();return last
                observation=root.observe(root.next_player,PAYLOADS[game]);seat=root.next_player
                def clone():
                    with root.clone() as child:return child.next_player
                record('python_clone_handle',clone)
                record('python_observe_owned_arrays',lambda:root.observe(seat,PAYLOADS[game]))
                record('python_pack_observation',lambda:pack_observations(contract,[observation]))
                for rows in [1,16,64]:
                    batch=pack_observations(contract,[observation]*rows)
                    record('python_checked_predict',lambda:snapshot.predict(batch),rows=rows,owned_input_bytes=sum(a.nbytes for a in batch.payloads.values())+batch.context.nbytes+batch.legal_mask.nbytes)
                for neural,iterations,parallel in [(False,32,1),(True,8,1),(True,32,1),(True,128,1),(True,32,4),(True,32,16)]:
                    def search():
                        with ExitStack() as stack:
                            games=[stack.enter_context(create(game,players,index%players)) for index in range(parallel)]
                            config={'schema_version':1,'seed':42,'rollout_length':0 if neural else 4,'budgets':{'iterations':iterations}}
                            bindings=None
                            if neural:
                                config.update(tree_policy='puct',state_evaluator={'name':'network_value'},puct_preferences={'source':'probabilities','evaluator':{'name':'network_policy'}},value_completion='require_full_vector')
                                bindings={'network_value':evaluator,'network_policy':evaluator}
                            searches=[stack.enter_context(tag.MctsSearch(g,dict(config,seed=42+index),bindings=bindings,value_observers='deciding_player')) for index,g in enumerate(games)]
                            with tag.MctsBatch(searches,max_group_rows=4096) as group:
                                reports=group.run();assert not group.errors,group.errors
                            assert len(reports)==parallel
                            for report in reports:assert report['counters']['completed_iterations']==iterations,report
                            return reports
                    reports=record('python_neural_mcts' if neural else 'python_native_mcts',search,iterations=iterations,parallel=parallel,includes_game_and_search_setup=True)
                    stream.write(json.dumps(dict(common,operation='search_structure',neural=neural,iterations=iterations,parallel=parallel,reports=reports))+'\n');stream.flush()
                if players==2 and game in ('tictactoe','connect4'):
                    torch.manual_seed(42)
                    scalar=PolicyValueNetwork(ModelContract.from_spec(root.spec(),payloads=('cells',)),(16,))
                    scalar_snapshot=FrozenModel(scalar)
                    scalar_path=args.models/f'{game}-{players}-osla.json'
                    if not scalar_path.exists():save_model(scalar_path,scalar,provenance={'scope':'untrained discounted observer benchmark fixture','seed':42})
                    neural_osla=NeuralValueEvaluator(scalar_snapshot)
                    def play_osla(backend,parallel=1):
                        with ExitStack() as stack:
                            games=[stack.enter_context(tag.create(game)) for _ in range(parallel)]
                            for index,g in enumerate(games):
                                for actor in range(2):
                                    seed=42+2*index+actor
                                    controller=tag.OSLAPython.rust_heuristic(seed=seed) if backend=='rust' else neural_osla.controller(seed=seed) if backend=='neural' else tag.OSLAPython(evaluator_id='python-open-lines',seed=seed)
                                    g.set_controller(actor,controller)
                            spec=games[0].spec();lines=spec.heuristic.indices.reshape(-1,spec.heuristic.window_length)
                            if parallel==1:
                                g=games[0];stop=g.advance()
                                while stop.reason=='evaluation':
                                    request=stop.request
                                    values=neural_osla.evaluate(request) if backend=='neural' else evaluate_open_lines(request,lines)
                                    stop=g.resume(request,values)
                                assert stop.reason=='finished'
                            else:
                                with tag.GameBatch(games) as group:
                                    pending=deque(group.advance().groups)
                                    while pending:
                                        request=pending.popleft();result=group.resume(request,evaluate_open_lines(request,lines))
                                        assert all(stop.error is None for stop in result.stops)
                                        pending.extend(result.groups)
                            assert all(g.next_player is None for g in games)
                    for backend in ['rust','python','neural']:
                        record('external_osla_'+backend,lambda:play_osla(backend),parallel=1,model_identity=scalar_snapshot.identity if backend=='neural' else None,parameter_count=sum(p.numel() for p in scalar.parameters()) if backend=='neural' else None,model_weight_bytes=sum(p.numel()*p.element_size() for p in scalar.parameters()) if backend=='neural' else None)
                    record('grouped_external_osla',lambda:play_osla('python',4),parallel=4,model_identity=None,parameter_count=None,model_weight_bytes=None)
                baseline=process_memory(__import__('os').getpid())
                clones=[root.clone() for _ in range(128)];loaded=process_memory(__import__('os').getpid())
                for item in clones:item.close()
                stream.write(json.dumps(dict(common,operation='python_retained_handles',count=128,idle=baseline,loaded=loaded))+'\n');stream.flush()
            print(f'Python fork {args.fork}: {game}/{players} complete',flush=True)

if __name__=='__main__':main()
