"""Sequential isolated MCTS matches, sampled APIs and comparable Windows process memory."""
import argparse
import datetime
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('research_process_memory',HERE.parent/'research/run.py')
research=importlib.util.module_from_spec(spec);spec.loader.exec_module(research)
PROFILES=research.PROFILES

def measurement_settings(settings):
    """Collection counts may change without changing the measured workload."""
    return {k:v for k,v in settings.items() if k not in ('games_per_profile','shards','shard_sizes')}

def shard_sizes(settings):
    return settings.get('shard_sizes',[settings['games_per_profile']//settings['shards']]*settings['shards'])

def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def interrupted_progress(destination,identity,command):
    checkpoint=destination.with_suffix('.baseline.json')
    progress=destination.with_suffix('.progress.jsonl')
    if not checkpoint.exists():
        if progress.exists() and progress.stat().st_size:
            raise ValueError(f'Partial games have no saved baseline; archive before restarting: {progress}')
        return [],[]
    saved=json.loads(checkpoint.read_text())
    if saved['identity']!=identity or saved['command']!=command:
        raise ValueError('Interrupted process settings/command differ')
    games=[]
    if progress.exists():
        data=progress.read_bytes();offset=0
        for line in data.splitlines(keepends=True):
            # A crash may leave only the final record incomplete. Never discard
            # a malformed complete record or silently accept a gap/duplicate.
            if not line.endswith(b'\n'):
                with progress.open('r+b') as stream:stream.truncate(offset)
                break
            game=json.loads(line)
            if game['id']!=len(games) or game['seed']!=identity['seed']+len(games):
                raise ValueError('Interrupted game coverage/seed mismatch')
            if not 0<=game['process_segment']<len(saved['segments']):
                raise ValueError('Missing process baseline for interrupted game')
            games.append(game);offset+=len(line)
    if len(games)>identity['games']:raise ValueError('Too many interrupted games')
    return games,saved['segments']

def run_process(command,destination,identity):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        saved=json.loads(destination.read_text())
        old=saved['identity']
        if ({k:v for k,v in old.items() if k!='settings'}!={k:v for k,v in identity.items() if k!='settings'}
            or measurement_settings(old['settings'])!=measurement_settings(identity['settings'])):
            raise ValueError(f'Existing result settings differ: {destination}')
        if len(saved['games'])!=identity['games'] or [g['id'] for g in saved['games']]!=list(range(identity['games'])):
            raise ValueError(f'Existing result has incomplete game coverage: {destination}')
        print('Already complete:',destination.name,flush=True);return saved
    games,segments=interrupted_progress(destination,identity,command)
    offset=len(games);segment_index=len(segments)
    if offset and offset==identity['games']:
        result={'identity':identity,'command':command,'baseline':segments[0]['baseline'],
                'ready':segments[0]['ready'],'process_segments':segments,'games':games,'wall_seconds':None}
        atomic_json(destination,result);return result
    resumed_command=list(command)
    if offset:
        # Each executable resets game/agent/simulation RNGs from the game seed.
        # Start a fresh warmed process at the first unrecorded game; keep binary
        # and MCTS settings unchanged, and retain each process's own baseline.
        common=len(command)-(9 if identity['engine']=='rust' else 10)
        resumed_command[common+2]=str(identity['games']-offset)
        resumed_command[common+4]=str(identity['seed']+offset)
        print(f'Resuming {destination.name}: {offset} completed games retained',flush=True)
    error_path=destination.with_suffix('.stderr.log')
    with error_path.open('a' if segments else 'w',encoding='utf-8') as errors:
        process=subprocess.Popen(resumed_command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
        stop=threading.Event();lock=threading.Lock();peak={};samples=0;sampling_errors=[]
        def sample():
            nonlocal samples
            while not stop.is_set():
                try:memory=research.process_memory(process.pid)
                except OSError as error:
                    if process.poll() is None:sampling_errors.append(str(error))
                    return
                with lock:
                    samples+=1
                    for k,v in memory.items():peak[k]=max(peak.get(k,0),v)
                stop.wait(.05)
        thread=None;baseline=None;ready=None;clock=time.perf_counter()
        try:
            for line in process.stdout:
                if not line.strip():continue
                value=json.loads(line)
                if value['phase']=='ready':
                    ready=value
                    baseline=research.process_memory(process.pid)
                    segments.append({'first_game_id':offset,'command':resumed_command,'baseline':baseline,'ready':ready,
                                     'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                    atomic_json(destination.with_suffix('.baseline.json'),
                                {'identity':identity,'command':command,'segments':segments})
                    thread=threading.Thread(target=sample,daemon=True);thread.start()
                    process.stdin.write('\n');process.stdin.flush()
                    print('Ready:',identity,flush=True)
                elif value['phase']=='game':
                    value['id']+=offset
                    if value['id']!=len(games) or value['seed']!=identity['seed']+len(games):
                        raise ValueError('Unexpected game id/seed from process')
                    value['process_segment']=segment_index
                    memory=research.process_memory(process.pid)
                    with lock:
                        value['memory_peak']={k:max(v,peak.get(k,0)) for k,v in memory.items()}
                        value['memory_samples']=samples;peak.clear();samples=0
                    games.append(value)
                    # Persist progress separately; completed shards are atomic and resumable.
                    with destination.with_suffix('.progress.jsonl').open('a',encoding='utf-8') as progress:
                        progress.write(json.dumps(value)+'\n')
                        progress.flush();os.fsync(progress.fileno())
                    if len(games)==1 or len(games)%5==0:
                        print(f"{identity['engine']} {identity['game']} {identity['players']}p shard {identity['shard']}: {len(games)}/{identity['games']} games; last {value['elapsed_ns']/1e9:.2f}s, {value['decisions']} actions",flush=True)
                    process.stdin.write('\n');process.stdin.flush()
                else:raise ValueError(f'Unexpected phase: {value}')
            code=process.wait()
            if code:raise RuntimeError(f'Probe exit {code}; see {error_path}')
            if baseline is None or len(games)!=identity['games']:raise ValueError('Incomplete match collection')
            if sampling_errors:raise RuntimeError(f'Memory sampling errors: {sampling_errors}')
            result={'identity':identity,'command':command,'baseline':segments[0]['baseline'],'ready':segments[0]['ready'],
                    'process_segments':segments,'games':games,'wall_seconds':time.perf_counter()-clock,
                    'wall_seconds_scope':'latest process segment only'}
            atomic_json(destination,result)
            return result
        finally:
            stop.set()
            if process.poll() is None:process.terminate();process.wait()
            if thread:thread.join(timeout=2)
            process.stdin.close();process.stdout.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games',type=int,default=500,help='Complete games per engine/profile, split across shards')
    p.add_argument('--shards',type=int,default=5)
    p.add_argument('--shard-sizes',help='Explicit comma-separated process batch sizes, e.g. 100,100,50')
    p.add_argument('--resize',action='store_true',help='Change collection counts only, preserving existing raw settings/results')
    p.add_argument('--iterations',type=int,default=5000)
    p.add_argument('--warmup',type=int,default=6,help='Untimed search decisions before the process baseline')
    p.add_argument('--stride',type=int,default=1024)
    p.add_argument('--seed',default='42',help='Nonnegative integer, or auto for one recorded OS-random seed')
    for key in ('aligned','seed-copies','skip-forced'):
        p.add_argument('--'+key,choices=('true','false'),required=True)
    p.add_argument('--engine',choices=('rust','java'))
    p.add_argument('--game',choices=('tictactoe','connect4','cantstop','loveletter'))
    p.add_argument('--players',type=int,choices=(2,3,4))
    p.add_argument('--report',action='store_true',help='Validate full coverage and write tables after collection')
    a=p.parse_args()
    sizes=[int(v) for v in a.shard_sizes.split(',')] if a.shard_sizes else None
    if (a.games<1 or a.shards<1 or (sizes is None and a.games%a.shards)
        or (sizes is not None and (len(sizes)!=a.shards or min(sizes)<1 or sum(sizes)!=a.games))
        or a.iterations<1 or a.warmup<0 or a.stride<0):p.error('Invalid counts or shard sizes')
    if os.name!='nt':p.error('The memory sampler requires Windows')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=output/'settings.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text());seed=previous['seed'] if a.seed=='auto' else int(a.seed)
    else:seed=int.from_bytes(os.urandom(8),'little')%(2**63-1000000) if a.seed=='auto' else int(a.seed)
    if not 0<=seed<2**63-1000000:p.error('seed must fit a nonnegative Java long with derivation headroom')
    settings={'games_per_profile':a.games,'shards':a.shards,'iterations':a.iterations,'rollout_length':10,
              'warmup_decisions':a.warmup,'sampling_stride':a.stride,'seed':seed,'aligned':a.aligned,
              'seed_copies':a.seed_copies,'skip_forced':a.skip_forced,'memory_interval_ms':50,'java_heap':'-Xms64m -Xmx512m'}
    settings['rules']={'tictactoe':{'grid_size':3},'connect4':{'width':7,'height':6,'winning_length':4},
        'cantstop':{'column_heights':[3,5,7,9,11,13,11,9,7,5,3],'markers':3,'columns_to_win':3},
        'loveletter':{'classic_cards':16,'winning_tokens':[6,5,4]}}
    settings['agent_seed_derivation']='4 * game_seed + 100000 + absolute_seat (wrapping 64-bit)'
    if sizes is not None:settings['shard_sizes']=sizes
    build_manifest=ROOT/'.scratch/mcts-games/build-manifest.json'
    settings['probe_build_sha256']=hashlib.sha256(build_manifest.read_bytes()).hexdigest()
    if manifest.exists() and json.loads(manifest.read_text())!=settings:
        old=json.loads(manifest.read_text())
        if not a.resize or measurement_settings(old)!=measurement_settings(settings):
            raise ValueError('Run settings changed; use --resize for collection-count changes only')
        original=output/'original-settings.json'
        if not original.exists():original.write_bytes(manifest.read_bytes())
    (output/'build-manifest.json').write_bytes(build_manifest.read_bytes())
    manifest.write_text(json.dumps(settings,indent=2)+'\n')
    (output/'runner-source-hashes.json').write_text(json.dumps(
        {str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    java=research.java_binary();classpath=(ROOT/'.scratch/mcts-games/classpath.txt').read_text()
    offset=0
    for shard,n in enumerate(shard_sizes(settings)):
        for game,players in PROFILES:
            if (a.game and a.game!=game) or (a.players and a.players!=players):continue
            for engine in (('rust','java') if shard%2==0 else ('java','rust')):
                if a.engine and a.engine!=engine:continue
                identity={'engine':engine,'game':game,'players':players,'shard':shard,'games':n,'seed':seed+offset,'settings':settings}
                common=[game,str(players),str(n),str(a.iterations),str(seed+offset),str(a.stride),a.aligned]
                command=([str(ROOT/'target/release/examples/mcts_games_measure.exe'),*common,a.skip_forced,str(a.warmup)] if engine=='rust' else
                         [java,'-Xms64m','-Xmx512m','-cp',classpath,'BenchmarkMctsGames',*common,a.seed_copies,a.skip_forced,str(a.warmup)])
                run_process(command,output/f'{engine}-{game}-{players}-shard{shard}.json',identity)
        offset+=n
    print('Requested profiles complete.',flush=True)
    if a.report:
        subprocess.run([sys.executable,str(HERE/'summarize.py'),str(output)],check=True)

if __name__=='__main__':main()
