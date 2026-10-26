"""Validate extra coverage, work counters and frozen runtime sources in the rerun."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from run import ROOT
from workflows import cases
from workflow_summary import game_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    read=lambda name:[json.loads(line) for line in (root/name).read_text().splitlines() if line.strip()]
    checks={}
    for phase,forks in [('timing',3),('allocations',1)]:
        expected={(engine,g,n,op,prefix,f) for f in range(1,forks+1) for g,n,op,prefix in cases()
                  if not(phase=='allocations' and op=='apply_control') for engine in ['rust','java']
                  if not(engine=='java' and op in ['save','load'])}
        rows=read(f'workflows-{phase}.jsonl')
        keys=[(r['engine'],r['game'],r['players'],r['operation'],r['prefix'],r['fork']) for r in rows]
        assert len(keys)==len(set(keys)) and set(keys)==expected
        for r in rows:
            assert r['seed']==42 and len(r['samples'])==6
            assert 0<=r['samples'][-1]['prepared_actions']<=r['prefix']
            for s in r['samples'][:-1]:
                assert s['calls']>0 and s['calls']%32==0 and s['elapsed_ns']>0
                assert s.get('allocated_bytes',0)>=0
                if phase=='allocations' and r['engine']=='rust':assert s['instrumented']
                if r['operation'].startswith('game'):
                    metrics=game_metrics(s)
                    assert metrics['turns_per_game']<=metrics['actions_per_game']
                    assert s['game_work'][4]==(0 if r['operation'].endswith('direct') else s['game_work'][1])
                if r['operation']=='apply_control':assert s['calls']==32000
        checks[f'workflow_{phase}_processes']=len(rows)
    primary=read('timing.jsonl');extra=read('workflows-timing.jsonl')
    by_mode=defaultdict(set)
    for r in primary+extra:
        if r['operation'] not in ['game','game_direct','game_aligned','game_aligned_direct']:continue
        for s in r['samples'][:-1]:
            metrics=game_metrics(s)
            by_mode[r['engine'],r['game'],r['players'],r['operation']].add(tuple(metrics[k] for k in ['actions_per_game','turns_per_game','rounds_per_game']))
    for (engine,g,n,mode),counts in by_mode.items():
        assert len(counts)==1,(engine,g,n,mode,'Fixture work varied between repetitions',counts)
        if mode in ['game','game_aligned']:
            assert counts==by_mode[engine,g,n,mode+'_direct'],(engine,g,n,mode,'Observation path changed measured work')
    checks['paired_game_work_profiles']=sum(mode in ['game','game_aligned'] for engine,g,n,mode in by_mode)
    sources=json.loads((root/'source-hashes.json').read_text())
    for path,digest in sources.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    checks['frozen_source_files']=len(sources)
    figures=list((root/'plots/workflows').glob('*.svg'));assert len(figures)==5
    for path in figures:ET.parse(path)
    for ext,magic in [('pdf',b'%PDF'),('png',b'\x89PNG\r\n\x1a\n')]:
        files=list((root/'plots/workflows').glob('*.'+ext));assert len(files)==5
        assert all(path.read_bytes().startswith(magic) for path in files)
    checks['extra_standalone_exports']=15
    (root/'workflow-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
