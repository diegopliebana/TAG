"""Run isolated, repeated Windows research measurements without editing public TAG.

Requires prebuilt disposable Java classes and Rust timing/allocation probes.
See README.md for builds, metric definitions, settings and supported comparisons.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
PROFILES=[('tictactoe',2),('connect4',2)]+[(g,p) for g in ('cantstop','loveletter') for p in (2,3,4)]
COMMON=['control','init','copy','observe','actions','transition','query','fork','first','random','osla','mcts','game']
EXTRA=['offered','encode','save','load']

def java_binary():
    # Oracle's PATH executable can be a launcher process. Measure the JVM itself.
    result=subprocess.run(['java','-XshowSettings:properties','-version'],capture_output=True,text=True,check=True)
    match=re.search(r'^\s*java.home\s*=\s*(.+)$',result.stderr,re.M)
    if not match:raise RuntimeError('Cannot discover java.home')
    binary=Path(match.group(1).strip())/'bin/java.exe'
    if not binary.is_file():raise RuntimeError('The direct JDK executable is unavailable')
    return str(binary)

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ['peak_working_set','working_set','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile','private_bytes']]

def process_memory(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    handle=kernel.OpenProcess(0x410,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        result=Counters();result.cb=ctypes.sizeof(result)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(result),result.cb):raise ctypes.WinError(ctypes.get_last_error())
        return {k:getattr(result,k) for k in ['working_set','peak_working_set','private_bytes','pagefile','peak_pagefile']}
    finally:kernel.CloseHandle(handle)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/benchmarks/research')
    parser.add_argument('--forks',type=int,default=3)
    parser.add_argument('--seed',default='42',help='Integer or auto; resolved once and recorded')
    parser.add_argument('--operation',choices=COMMON+EXTRA,help='Restrict a timing/allocation collection to one operation')
    parser.add_argument('--engine',choices=['rust','java'],help='Restrict an exploratory/replacement run to one engine')
    parser.add_argument('--phase',choices=['timing','allocations','retention'],required=True)
    args=parser.parse_args()
    if os.name!='nt':parser.error('Process memory probe currently requires Windows')
    if not 1<=args.forks<=10:parser.error('forks must be 1..10')
    seed=int.from_bytes(os.urandom(8),'little')%(2**63) if args.seed=='auto' else int(args.seed)
    if not 0<=seed<2**63:parser.error('seed must fit a nonnegative Java long')
    output=(ROOT/args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    java_root=ROOT/'.scratch/java-comparison'
    cp=str(java_root/'target/classes')+';'+(java_root/'target/comparison-classpath.txt').read_text().strip()
    def command(engine,game,players,op,count,iterations,warm,sample):
        base=[java,'-Xms64m','-Xmx512m','-cp',cp,'BenchmarkResearch'] if engine=='java' else [str(ROOT/'.scratch/research-bench'/('rust-memory.exe' if args.phase!='timing' else 'rust-timing.exe'))]
        return base+[game,str(players),op,str(count),str(seed),str(iterations),str(warm),str(sample)]
    java=java_binary()
    records=[];path=output/(args.phase+'.jsonl')
    if path.exists():parser.error(f'{path.name} exists; choose a new output directory')
    with path.open('w',encoding='utf-8') as stream:
        for fork in range(args.forks):
            for game,players in PROFILES:
                for op in (['retain_states'] if args.phase=='retention' else ([args.operation] if args.operation else COMMON+EXTRA)):
                    for count in ([1,128,2048] if args.phase=='retention' else [1]):
                        for engine in (['rust','java'] if fork%2==0 else ['java','rust']):
                            if args.engine and engine!=args.engine:continue
                            if engine=='java' and op in EXTRA:continue
                            cmd=command(engine,game,players,op,count,32,1000 if engine=='java' else 150,80 if args.phase=='timing' else 5)
                            if args.phase=='allocations' and engine=='rust':cmd[-2]='20'
                            if args.phase=='allocations' and engine=='java':cmd[-2]='1000'
                            started=time.perf_counter()
                            if args.phase=='retention':
                                proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace')
                                try:
                                    idle=json.loads(proc.stdout.readline());idle['process']=process_memory(proc.pid)
                                    proc.stdin.write('continue\n');proc.stdin.flush()
                                    loaded=json.loads(proc.stdout.readline());loaded['process']=process_memory(proc.pid)
                                    proc.stdin.write('finish\n');proc.stdin.flush()
                                    _,error=proc.communicate(timeout=30)
                                    if proc.returncode:raise RuntimeError(error)
                                    rows=[idle,loaded]
                                finally:
                                    if proc.poll() is None:proc.kill();proc.wait()
                            else:
                                proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
                                if proc.returncode:raise RuntimeError(f'{engine} {game} {op}: {proc.stderr}')
                                rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
                                assert len(rows)==6,(engine,game,op,rows)
                            record=dict(engine=engine,game=game,players=players,operation=op,count=count,fork=fork+1,seed=seed,iterations=32,wall_seconds=time.perf_counter()-started,samples=rows)
                            if args.phase!='retention':
                                record['warmup_ms']=int(cmd[-2]);record['sample_min_ms']=int(cmd[-1])
                            stream.write(json.dumps(record)+'\n');stream.flush();records.append(record)
                print(f'{args.phase}: fork {fork+1}, {game}/{players} complete ({len(records)} processes)',flush=True)
    print(f'Saved {len(records)} process records to {path.relative_to(ROOT)}')

if __name__=='__main__':main()
