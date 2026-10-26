"""Continue a prepared benchmark run, executing phases sequentially to avoid contention."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run import ROOT,PROFILES,process_memory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='docs/benchmarks/research');a=p.parse_args();out=ROOT/a.output
    assert len((out/'timing.jsonl').read_text().splitlines())==720,'Complete the three-fork timing collection first'
    assert {json.loads(line)['seed'] for line in (out/'timing.jsonl').read_text().splitlines()}=={42},'This full-study fixture uses seed 42; exploratory seed collections are separate.'
    def call(arguments):
        result=subprocess.run(arguments,cwd=ROOT)
        if result.returncode:raise SystemExit(result.returncode)
    for phase,forks,expected in [('allocations',1,240),('retention',3,144)]:
        path=out/(phase+'.jsonl')
        if not path.exists():call([sys.executable,'tools/benchmarks/research/run.py','--phase',phase,'--forks',str(forks),'--output',a.output])
        assert len(path.read_text().splitlines())==expected,(phase,'Incomplete phase; preserve partial results and select a new output directory')
    for fork in [1,2,3]:
        path=out/f'python-{fork}.jsonl'
        if not path.exists():call([sys.executable,'tools/benchmarks/research/python_measure.py','--fork',str(fork),'--output',a.output])
        assert len({(r['game'],r['players']) for r in (json.loads(line) for line in path.read_text().splitlines())})==8
    models=sorted((ROOT/'.scratch/research-bench/models').glob('*.json'))+[ROOT/'web/tests/fixtures/learned/q-table.json']
    assert len(models)==11
    for fork in [1,2,3]:
        path=out/f'adapter-{fork}.jsonl'
        if not path.exists():
            result=subprocess.run([str(ROOT/'target/release/examples/research_adapter.exe'),*map(str,models)],cwd=ROOT,capture_output=True,encoding='utf-8',errors='replace',timeout=240)
            if result.returncode:
                (out/f'adapter-{fork}-failure.txt').write_text(result.stderr,encoding='utf-8');raise RuntimeError(result.stderr)
            path.write_text(result.stdout,encoding='utf-8');print('Native adapter fork',fork,'complete',flush=True)
        assert len(path.read_text().splitlines())==109,'Incomplete or obsolete adapter collection'
    features=[]
    for fork in [1,2,3]:
        for game,players in PROFILES:
            for profile in ['minimal-features','optional-features']:
                proc=subprocess.Popen([str(ROOT/'.scratch/research-bench'/(profile+'.exe')),game,str(players),'42'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    idle=json.loads(proc.stdout.readline());idle['process']=process_memory(proc.pid)
                    proc.stdin.write('continue\n');proc.stdin.flush();loaded=json.loads(proc.stdout.readline());loaded['process']=process_memory(proc.pid)
                    proc.stdin.write('finish\n');proc.stdin.flush();_,error=proc.communicate(timeout=30);assert proc.returncode==0,error
                    features.append(dict(fork=fork,game=game,players=players,profile=profile,binary_bytes=(ROOT/'.scratch/research-bench'/(profile+'.exe')).stat().st_size,idle=idle,loaded=loaded))
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
    (out/'features.json').write_text(json.dumps(features,indent=2)+'\n',encoding='utf-8')
    print('Optional-feature initialized-memory checks complete',flush=True)
    if not (out/'matched').exists():
        call(['powershell','-NoProfile','-File','tools/java-comparison/run-benchmarks.ps1','-Passes','500','-Forks','3','-OutputDirectory',str(Path(a.output)/'matched')])
    call([sys.executable,'tools/java-comparison/summarize-benchmarks.py',str(out/'matched')])
    call([sys.executable,'tools/benchmarks/research/summarize.py',a.output])

if __name__=='__main__':main()
