"""Correct historical tactical counts to include OSLA's 0.001 near-best tolerance.

No training or weights change. Historical outcome reports remain intact; this
writes explicitly corrected diagnostics and checks a real native near-tie case.
"""
import json
from pathlib import Path
import runpy

import torch
import tag_rust as tag
from tag_rust.learning.artifacts import load_model
from tag_rust.learning.play import NeuralValueEvaluator
from tag_rust.learning.snapshot import FrozenModel


def main():
    torch.set_num_threads(1)
    helper=runpy.run_path('tools/learning/evaluate-strength.py')
    manifest=json.loads(Path('target/m7-broader-training/experiment.json').read_text())
    models=[('original',Path('target/models/tictactoe.json'))]
    models.extend((r['name'],Path(r['model'])) for r in manifest['runs'] if r['kind']=='neural')
    result=dict(tie_tolerance=.001,models=[])
    for name,path in models:
        snapshot=FrozenModel(load_model(path))
        diagnostics,_=helper['inspect_positions'](snapshot)
        result['models'].append(dict(name=name,model_identity=snapshot.identity,diagnostics=diagnostics))
        print(name,diagnostics['missed_forced_blocks'],flush=True)
    snapshot=FrozenModel(load_model(models[0][1]));evaluator=NeuralValueEvaluator(snapshot)
    counts={7:0,8:0}
    for seed in range(64):
        with tag.create('tictactoe') as game:
            for index,action in enumerate((0,1,2,3,4,6,5)):
                game.apply(action,position=game.observe(index%2,payloads=()).position,actor=index%2)
            game.set_controller(1,evaluator.controller(seed=seed))
            request=game.prepare_evaluation()
            while isinstance(request,tag.EvaluationRequest):
                request=game.finish_evaluation(request,evaluator.evaluate(request))
            counts[request.action_id]+=1
    assert all(counts.values()),'both near-best moves should be eligible in the native controller'
    result['native_near_tie_choice_counts']=counts
    output=Path('target/m7-s5/corrected-diagnostics.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
