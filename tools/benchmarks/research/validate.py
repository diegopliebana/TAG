"""Validate complete workload coverage and numerical/protocol integrity of accepted samples."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from run import PROFILES,COMMON,EXTRA

def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path,nargs='?',default=Path('docs/benchmarks/research'));a=p.parse_args();root=a.directory
    checks={}
    for phase,forks in [('timing',3),('allocations',1)]:
        rows=lines(root/(phase+'.jsonl'))
        expected={(engine,g,n,op,f) for g,n in PROFILES for engine in ['rust','java'] for op in (COMMON+EXTRA if engine=='rust' else COMMON) for f in range(1,forks+1)}
        keys=[(r['engine'],r['game'],r['players'],r['operation'],r['fork']) for r in rows];assert len(keys)==len(set(keys)) and set(keys)==expected
        for r in rows:
            assert len(r['samples'])==6 and r['seed']==42
            for s in r['samples'][:-1]:assert s['calls']>0 and s['elapsed_ns']>0 and math.isfinite(s['elapsed_ns']/s['calls'])
            if phase=='timing' and r['engine']=='java' and r['operation']=='mcts':assert r['samples'][-1]['search_visits']==r['iterations']==32
            if phase=='allocations' and r['engine']=='rust':assert all(s['instrumented'] for s in r['samples'][:-1])
        checks[phase+'_processes']=len(rows)
    memory=lines(root/'retention.jsonl');assert len(memory)==144
    keys=[(r['engine'],r['game'],r['players'],r['count'],r['fork']) for r in memory];assert len(set(keys))==144
    for r in memory:
        assert [s['phase'] for s in r['samples']]==['idle','loaded']
        for s in r['samples']:
            m=s['process'];assert m['peak_working_set']>=m['working_set']>0 and m['private_bytes']>0
            if r['engine']=='java':assert m['private_bytes']>=s['heap_used_bytes'],'Sampler must observe the JVM, not a launcher'
        assert r['samples'][1]['count']==r['count']
    checks['retention_processes']=144
    for fork in [1,2,3]:
        rows=lines(root/f'python-{fork}.jsonl');assert len(rows)==160
        assert {(r['game'],r['players']) for r in rows}==set(PROFILES)
        for r in rows:
            if r['operation']=='search_structure':
                assert len(r['reports'])==r['parallel']
                for report in r['reports']:assert report['counters']['completed_iterations']==r['iterations']
        adapters=lines(root/f'adapter-{fork}.jsonl');assert len(adapters)==109
        assert {(r['game'],r['players']) for r in adapters}==set(PROFILES)
    checks['python_processes']=3;checks['native_adapter_processes']=3
    tree=json.loads((root/'tree-memory.json').read_text());assert len(tree)==96
    for r in tree:
        if r['engine']=='java':assert r['loaded']['visits']==r['iterations'] and r['loaded']['process']['private_bytes']>=r['loaded']['heap_used_bytes']
    checks['tree_memory_processes']=96
    cli=json.loads((root/'cli.json').read_text());assert len(cli)==72
    for fork in [1,2,3]:
        for g,n in PROFILES:
            match=[r for r in cli if (r['fork'],r['game'],r['players'],r['operation'])==(fork,g,n,'match_cli')]
            assert len(match)==2 and match[0]['work_checksum']==match[1]['work_checksum']
    checks['cli_game_executions']=sum(r['completed_games'] for r in cli)
    assert len(json.loads((root/'features.json').read_text()))==48
    for fork in [1,2,3]:
        rows=json.loads((root/f'episodes-{fork}.json').read_text());assert len(rows)==8
        for r in rows:assert r['next_episode_id']==r['episodes']
    for ext in ['svg','pdf','png']:
        files=list((root/'plots').glob('*.'+ext));assert len(files)==12,(ext,len(files))
        for file in files:
            if ext=='svg':ET.parse(file)
            if ext=='pdf':assert file.read_bytes().startswith(b'%PDF')
            if ext=='png':assert file.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    checks['standalone_figures']=12;checks['standalone_exports']=36
    (root/'validation.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
