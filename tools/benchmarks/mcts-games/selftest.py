"""Check that measurement probes preserve seeded trajectories and work accounting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import run

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('run.py')
    for stride in (0,1,1024):
        subprocess.run([sys.executable,str(script),'--output',str(root/f'stride-{stride}'),'--games','2','--shards','1',
            '--iterations','32','--warmup','1','--stride',str(stride),'--aligned','true','--seed-copies','true','--skip-forced','true'],check=True)
    checks=0
    for engine in ('rust','java'):
        for game,players in run.PROFILES:
            cases=[json.loads((root/f'stride-{s}'/f'{engine}-{game}-{players}-shard0.json').read_text()) for s in (0,1,1024)]
            traces=[[(g['seed'],g['trace_hash'],g['results'],g['decisions'],g['searched_decisions']) for g in case['games']] for case in cases]
            assert traces[0]==traces[1]==traces[2],(engine,game,players,'instrumentation changed play')
            for case in cases[1:]:
                for g in case['games']:
                    metrics={(m['scope'],m['operation']):m for m in g['metrics']}
                    assert metrics['real','init']['calls']==1
                    assert metrics['real','apply']['calls']==g['decisions']
                    assert metrics['real','copy_apply']['samples']==g['decisions']
                    assert metrics['search','actions']['calls']>0
                    assert g['searched_decisions']<=g['decisions']
                    assert all(g['memory_peak'][key]>0 for key in ('working_set','private_bytes'))
            checks+=1
    (root/'validation.json').write_text(json.dumps({'profiles_checked':checks,'games_per_profile_mode':2,
        'strides':[0,1,1024],'trajectories_preserved':True,'accounting_valid':True},indent=2)+'\n')
    print('Validated 16 engine/profile combinations with three probe modes and two games each.')

if __name__=='__main__':main()
