"""Measure search-memory scaling, distinguishing temporary Rust trees and retained Java trees."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
from run import ROOT,PROFILES,process_memory,java_binary

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'docs/benchmarks/research');args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True);r=ROOT/'.scratch/java-comparison';cp=str(r/'target/classes')+';'+(r/'target/comparison-classpath.txt').read_text().strip();rows=[];java=java_binary()
    for fork in [1,2,3]:
        for game,players in PROFILES:
            for budget in [32,128]:
                command=[java,'-Xms64m','-Xmx512m','-cp',cp,'BenchmarkTreeMemory',game,str(players),str(budget)]
                proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace')
                try:
                    line=proc.stdout.readline()
                    if not line:raise RuntimeError(proc.stderr.read())
                    idle=json.loads(line);idle['process']=process_memory(proc.pid)
                    proc.stdin.write('continue\n');proc.stdin.flush();line=proc.stdout.readline()
                    if not line:raise RuntimeError(proc.stderr.read())
                    loaded=json.loads(line);loaded['process']=process_memory(proc.pid)
                    proc.stdin.write('finish\n');proc.stdin.flush();_,error=proc.communicate(timeout=30);assert proc.returncode==0,error
                    rows.append(dict(engine='java',fork=fork,game=game,players=players,iterations=budget,idle=idle,loaded=loaded))
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
                result=subprocess.run([str(ROOT/'.scratch/research-bench/rust-memory.exe'),game,str(players),'mcts','1','42',str(budget),'20','5'],cwd=ROOT,capture_output=True,text=True,timeout=60);assert result.returncode==0,result.stderr
                rows.append(dict(engine='rust',fork=fork,game=game,players=players,iterations=budget,samples=[json.loads(line) for line in result.stdout.splitlines()],tree_lifetime='temporary; MctsAgent retains the root report, not the full tree'))
        print('Tree memory fork',fork,'complete',flush=True)
    (output/'tree-memory.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
