"""Regenerate small trained-model and PyTorch prediction fixtures for Rust/Wasm checks.

Run from the repository root using the configured Python interpreter. Outputs are test
data, not playing-strength claims; no browser dependency or service is required.
"""
from pathlib import Path
import json
import numpy as np
import torch
import tag_rust as tag
from tag_rust.learning.training import TrainingSettings, TrainingSession
from tag_rust.learning.artifacts import model_document
from tag_rust.learning.inputs import ModelContract, pack_observations
from tag_rust.learning.network import PolicyValueNetwork
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.play import NeuralValueEvaluator
from tag_rust.learning.tabular import QTable, run


def main():
    destination=Path('web/tests/fixtures/learned')
    destination.mkdir(parents=True,exist_ok=True)
    fixtures=[]
    for name in ('tictactoe','connect4'):
        session=TrainingSession(TrainingSettings(game=name,master_seed=42,hidden_sizes=(4,),teacher_episodes=8,actor_episodes=8,group_size=4))
        list(session.run())
        models=[('trained',session.model)]
        zero=PolicyValueNetwork(session.model.contract,(4,))
        with torch.no_grad():
            for parameter in zero.parameters():parameter.zero_()
        models.append(('zero',zero))
        with tag.create(name) as game:
            torch.manual_seed(19)
            combined=PolicyValueNetwork(ModelContract.from_spec(game.spec(),payloads=('board','cells')), (3,))
            models.append(('combined',combined))
            torch.manual_seed(23)
            wide=PolicyValueNetwork(session.model.contract)
            models.append(('wide',wide))
        for variant,model in models:
            snapshot=FrozenModel(model)
            document=model_document(model,provenance=dict(purpose='browser numerical fixture'))
            filename=f'{name}-{variant}.json'
            (destination/filename).write_text(json.dumps(document,sort_keys=True,separators=(',',':')),encoding='utf-8')
            moves=[0,3,1,4,2] if name=='tictactoe' else [0,1,0,1,0,1,0]
            rows=[]
            with tag.create(name) as game:
                for count in range(len(moves)+1):
                    for observer in (0,1):
                        observation=game.observe(observer,payloads=tuple(t.name for t in model.contract.tensors))
                        batch=pack_observations(model.contract,[observation])
                        output=snapshot.predict(batch)
                        rows.append(dict(prefix=moves[:count],observer=observer,inputs=batch.inputs[0].tolist(),
                                         logits=output.logits[0].tolist(),value=float(output.values[0])))
                    if count<len(moves):
                        observation=game.observe(count%2,payloads=())
                        game.apply(moves[count],position=observation.position,actor=count%2)
            traces=[]
            if variant in ('trained','zero'):
                evaluator=NeuralValueEvaluator(snapshot)
                for seat in (0,1):
                    with tag.create(name) as game:
                        game.set_controller(seat,evaluator.controller(seed=7))
                        game.set_controller(1-seat,tag.OSLAPython.rust_heuristic(seed=13))
                        # Bound each execution call to one accepted move so all actions are retained.
                        actions=[]
                        before=np.zeros(9 if name=='tictactoe' else 42,dtype=np.float32)
                        stop=game.advance(max_moves=1)
                        while True:
                            if stop.reason=='evaluation':
                                stop=game.resume(stop.request,evaluator.evaluate(stop.request),max_moves=1)
                                continue
                            observation=game.observe(0,payloads=('cells',))
                            cells=observation.payloads['cells']
                            changed=np.flatnonzero(cells!=before)
                            if len(changed):
                                index=int(changed[0]); actions.append(index if name=='tictactoe' else index//6)
                                before=cells.copy()
                            if stop.reason=='finished':break
                            if stop.reason!='move_limit':raise RuntimeError(stop.reason)
                            stop=game.advance(max_moves=1)
                        traces.append(dict(seat=seat,actions=actions))
            fixtures.append(dict(game=name,model_file=filename,rows=rows,traces=traces))
    task=dict(game='tictactoe',parameters={},learner=None,opponent='random',opponent_settings={},max_episode_steps=None)
    table=QTable(task)
    run(table,rounds=16,members=4,seed=42,epsilon=.2,learning=True)
    table.save(destination/'q-table.json')
    (destination/'fixtures.json').write_text(json.dumps(dict(torch=str(torch.__version__),numpy=np.__version__,
        absolute_tolerance=2e-6,relative_tolerance=2e-5,models=fixtures),indent=2)+'\n',encoding='utf-8')
    print('Wrote trained/combined model and Q-table fixtures to',destination)


if __name__=='__main__':main()
