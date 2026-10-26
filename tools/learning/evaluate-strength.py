"""Evaluate an existing Tic-Tac-Toe model without changing its weights.

Writes outcomes, exact native OSLA traces and prediction rows for the companion
evaluate-strength.mjs Wasm check. Run from the repository root.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import tag_rust as tag
from tag_rust.learning.artifacts import load_model
from tag_rust.learning.evaluation import evaluate_model, empty_outcomes
from tag_rust.learning.inputs import pack_observations
from tag_rust.learning.play import NeuralValueEvaluator
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.tabular import QTable, run


LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))


def winner(board):
    return next((board[a] for a,b,c in LINES if board[a] and board[a] == board[b] == board[c]), 0)


@lru_cache(None)
def optimal(board, actor):
    """Exact win/draw/loss under best play, measured for the next actor."""
    won = winner(board)
    if won:
        return 1 if won == actor else -1
    legal = [i for i,x in enumerate(board) if not x]
    if not legal:
        return 0
    return max(-optimal(board[:a]+(actor,)+board[a+1:], -actor) for a in legal)


def add_result(outcomes, seat, code):
    for counts in (outcomes['combined'], outcomes['per_seat'][seat]):
        counts['episodes'] += 1
        counts[{1:'wins',2:'draws',3:'losses'}[code]] += 1


def evaluate_osla(snapshot, opponent, seed, episodes):
    evaluator = NeuralValueEvaluator(snapshot)
    outcomes, traces = empty_outcomes(), []
    for logical in range(episodes):
        seat, learner_seed, opponent_seed = logical % 2, seed+2*logical, seed+2*logical+1
        rng = np.random.default_rng(opponent_seed)
        with tag.create('tictactoe') as game:
            game.set_controller(seat, evaluator.controller(seed=learner_seed))
            if opponent == 'osla':
                game.set_controller(1-seat, tag.OSLAPython.rust_heuristic(seed=opponent_seed))
            actions = []
            while not game.observe(0, payloads=()).done:
                before = game.observe(0, payloads=('cells',)).payloads['cells'].copy()
                observation = game.observe(1-seat, payloads=())
                if opponent == 'random' and observation.next_player == 1-seat:
                    action = int(rng.choice(observation.offered_action_ids))
                    game.apply(action, position=observation.position, actor=1-seat)
                else:
                    stop = game.advance(max_moves=1)
                    while stop.reason == 'evaluation':
                        stop = game.resume(stop.request, evaluator.evaluate(stop.request), max_moves=1)
                    if stop.reason not in ('move_limit','finished','human'):
                        raise RuntimeError(stop.reason)
                after = game.observe(0, payloads=('cells',))
                changed = np.flatnonzero(before != after.payloads['cells'])
                assert len(changed) == 1
                actions.append(int(changed[0]))
            code = int(game.observe(seat,payloads=()).results[seat])
            add_result(outcomes, seat, code)
            traces.append(dict(seat=seat,learner_seed=learner_seed,opponent_seed=opponent_seed,
                               actions=actions,result=code))
    return dict(mode='osla',opponent=opponent,seed=seed,**outcomes), traces


def inspect_positions(snapshot, table=None):
    """Visit every reachable ongoing position once; assess choices with exact play."""
    assert optimal((0,)*9,1)==0, 'default Tic-Tac-Toe must draw under best play'
    seen, rows, totals = set(), [], dict(positions=0,suboptimal=0,forced_losses=0,
        immediate_win_opportunities=0,missed_immediate_wins=0,forced_block_opportunities=0,
        missed_forced_blocks=0,block_examples=[],q_covered_positions=0)
    def visit(prefix):
        with tag.create('tictactoe') as game:
            for turn,action in enumerate(prefix):
                observation=game.observe(turn%2,payloads=())
                game.apply(action,position=observation.position,actor=turn%2)
            observation=game.observe(0,payloads=('cells',))
            key=tuple(int(x) for x in observation.payloads['cells'])
            if key in seen or observation.done:
                return
            seen.add(key)
            actor=observation.next_player
            if table is not None:
                relative=tuple(int(x) for x in game.observe(actor,payloads=('cells',)).payloads['cells'])
                totals['q_covered_positions']+=relative in table.values
            legal=game.observe(actor,payloads=()).offered_action_ids.tolist()
            predictions, exact, immediate = [], [], []
            for action in legal:
                with game.clone() as successor:
                    successor.apply(action,position=successor.observe(actor,payloads=()).position,actor=actor)
                    obs=successor.observe(actor,payloads=tuple(t.name for t in snapshot.contract.tensors))
                    output=snapshot.predict(pack_observations(snapshot.contract,[obs]))
                    code=int(obs.results[actor])
                    predictions.append({1:1.,2:0.,3:-1.}[code] if code else float(output.values[0]))
                following=key[:action]+((1 if actor==0 else -1),)+key[action+1:]
                exact.append(-optimal(following, -1 if actor==0 else 1))
                immediate.append(winner(following) != 0)
            best=max(predictions)
            # Match native OSLA's near-best selection, including its value scale.
            selected=[i for i,v in enumerate(predictions) if best-v<=.001]
            totals['positions']+=1
            totals['suboptimal']+=sum(exact[i]<max(exact) for i in selected)/len(selected)
            totals['forced_losses']+=max(exact)==-1
            if any(immediate):
                totals['immediate_win_opportunities']+=1
                totals['missed_immediate_wins']+=sum(not immediate[i] for i in selected)/len(selected)
            threats=[]
            opponent_mark=-1 if actor==0 else 1
            for action in legal:
                following=key[:action]+(opponent_mark,)+key[action+1:]
                if winner(following):threats.append(action)
            if len(threats)==1 and not any(immediate) and max(exact)>=0:
                totals['forced_block_opportunities']+=1
                totals['missed_forced_blocks']+=sum(legal[i]!=threats[0] for i in selected)/len(selected)
                if legal[selected[0]]!=threats[0] and len(totals['block_examples'])<3:
                    totals['block_examples'].append(dict(prefix=prefix,actor=actor,required_action=threats[0],
                        chosen_actions=[legal[i] for i in selected],legal=legal,predicted_values=predictions,
                        exact_values=exact))
            if len(rows)<256:
                obs=game.observe(actor,payloads=tuple(t.name for t in snapshot.contract.tensors))
                batch=pack_observations(snapshot.contract,[obs]); prediction=snapshot.predict(batch)
                rows.append(dict(prefix=prefix,observer=actor,inputs=batch.inputs[0].tolist(),
                                 logits=prediction.logits[0].tolist(),value=float(prediction.values[0])))
            for action in legal:visit(prefix+[action])
    visit([])
    assert totals['positions']==4520, 'expected every reachable ongoing default Tic-Tac-Toe position'
    return totals,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('target/models/tictactoe.json'))
    parser.add_argument('--q-model',type=Path)
    parser.add_argument('--episodes',type=int,default=100,help='games per seed and matchup; must be even')
    parser.add_argument('--seeds',type=int,nargs='+',default=[90001,90002])
    parser.add_argument('--output',type=Path,default=Path('target/strength-evaluation'))
    args=parser.parse_args()
    if args.episodes<2 or args.episodes%2:parser.error('episodes must be positive and even')
    if any(seed<0 for seed in args.seeds) or len(set(args.seeds))!=len(args.seeds):
        parser.error('evaluation seeds must be distinct and nonnegative')
    torch.set_num_threads(1)
    model=load_model(args.model); snapshot=FrozenModel(model)
    if snapshot.contract.identity.game!='tictactoe':parser.error('this diagnostic currently covers default Tic-Tac-Toe')
    args.output.mkdir(parents=True,exist_ok=True)
    result=dict(model=str(args.model.resolve()),model_identity=snapshot.identity,seeds=args.seeds,
                episodes_per_seed=args.episodes,torch=str(torch.__version__),numpy=np.__version__,evaluations=[],traces=[])
    result['provenance']=json.loads(args.model.read_text(encoding='utf-8'))['provenance']
    table=QTable.load(args.q_model) if args.q_model else None
    for seed in result['seeds']:
        for opponent in ('random','osla'):
            policy=evaluate_model(model,episodes=args.episodes,seed=seed,mode='policy',opponent=opponent)
            result['evaluations'].append(policy)
            osla,traces=evaluate_osla(snapshot,opponent,seed,args.episodes)
            result['evaluations'].append(osla)
            result['traces'].append(dict(opponent=opponent,seed=seed,games=traces))
            print(json.dumps(dict(policy=policy['combined'],osla=osla['combined'],opponent=opponent,seed=seed)),flush=True)
    result['position_diagnostics'],result['prediction_rows']=inspect_positions(snapshot,table)
    if args.q_model:
        result['q_model']=str(args.q_model.resolve());result['q_rows']=len(table.values);result['q_evaluations']=[]
        result['q_sha256']=hashlib.sha256(args.q_model.read_bytes()).hexdigest()
        result['q_task']=table.task
        choose=table.choose
        for seed in result['seeds']:
            for opponent in ('random','osla'):
                coverage=dict(decisions=0,unseen=0,all_legal_values_equal=0)
                def tracked(observation,info,rng,*,epsilon):
                    coverage['decisions']+=1
                    row=table.values.get(table._key(observation))
                    if row is None:coverage['unseen']+=1
                    if row is None or np.ptp(row[table._legal(observation)])==0:coverage['all_legal_values_equal']+=1
                    return choose(observation,info,rng,epsilon=epsilon)
                table.choose=tracked
                counts=run(table,rounds=args.episodes,members=1,seed=seed,epsilon=0,learning=False,opponent=opponent)
                result['q_evaluations'].append(dict(seed=seed,opponent=opponent,coverage=coverage,**counts))
                print(json.dumps(dict(mode='q_learning',seed=seed,opponent=opponent,coverage=coverage,
                                      wins=counts['wins'],draws=counts['draws'],losses=counts['losses'])),flush=True)
    (args.output/'python.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['position_diagnostics']),flush=True)


if __name__=='__main__':main()
