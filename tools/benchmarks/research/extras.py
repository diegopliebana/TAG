"""Measure complete CLI schedules and native/Python learner collection workflows.

CLI times include startup, setup and output transfer; they are not per-rule timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from run import ROOT,PROFILES


def cli(output):
    rows=[]
    folder=Path(tempfile.mkdtemp(prefix='cli-',dir=ROOT/'.scratch/research-bench'))
    for fork in [1,2,3]:
        for game,players in PROFILES:
            baseline=None
            for include in ([False,True] if fork%2 else [True,False]):
                config={'schema_version':1,'game':{'name':game},'players':[{'name':str(i),'agent':{'type':'random'}} for i in range(players)],'games':64,'positions':'rotate','seed':42,'reporting':{'format':'json','include_actions':include}}
                path=folder/'match.json';path.write_text(json.dumps(config),encoding='utf-8');start=time.perf_counter_ns()
                result=subprocess.run([str(ROOT/'target/release/examples/match_play.exe'),str(path)],cwd=ROOT,capture_output=True)
                elapsed=time.perf_counter_ns()-start;assert result.returncode==0,result.stderr
                report=json.loads(result.stdout);assert report['completed_games']==64 and report['failure'] is None
                moves=sum(g['accepted_moves'] for g in report['games'])
                for g in report['games']:g.pop('actions',None)
                identity=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
                if baseline is None:baseline=identity
                else:assert identity==baseline,'Action logging changed outcomes/streams'
                rows.append(dict(operation='match_cli',fork=fork,game=game,players=players,include_actions=include,elapsed_ns=elapsed,completed_games=64,accepted_moves=moves,stdout_bytes=len(result.stdout),work_checksum=identity,configuration=config))
            config={'schema_version':1,'agents':[{'name':str(i),'agent':{'type':'random'}} for i in range(players)],'games':[{'id':game,'name':game,'player_count':players}],'mode':'random','matchups':2,'seat_allocation':'permutations','allow_self_play':False,'repetitions':1,'ranking_metric':'points_rate','seed':42,'results_file':f'tournament-{fork}-{game}-{players}.json','game_logs':{'mode':'off'}}
            path=folder/'tournament.json';path.write_text(json.dumps(config),encoding='utf-8');start=time.perf_counter_ns()
            result=subprocess.run([str(ROOT/'target/release/examples/tournament.exe'),str(path)],cwd=ROOT,capture_output=True);elapsed=time.perf_counter_ns()-start;assert result.returncode==0,result.stderr
            report=json.loads((folder/config['results_file']).read_text());assert report['failure'] is None and report['completed_games']>0
            rows.append(dict(operation='tournament_cli',fork=fork,game=game,players=players,elapsed_ns=elapsed,completed_games=report['completed_games'],stdout_bytes=len(result.stdout),report_bytes=(folder/config['results_file']).stat().st_size,configuration=config))
        print('CLI schedule fork',fork,'complete',flush=True)
    (output/'cli.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')


def episodes(output,fork):
    from tag_rust.learning.collection import collect_episode_group
    from tag_rust.learning.artifacts import load_model
    from tag_rust.learning.snapshot import FrozenModel
    from python_measure import measure
    rows=[]
    for game in ['tictactoe','connect4']:
        snapshot=FrozenModel(load_model(ROOT/'.scratch/research-bench/models'/f'{game}-2-osla.json'))
        for phase in ['teacher','actor_critic']:
            for count in [1,16]:
                def collect():
                    return collect_episode_group(game,episode_count=count,master_seed=42,phase=phase,payloads=('cells',),snapshot=None if phase=='teacher' else snapshot)
                samples,group=measure(collect)
                rows.append(dict(operation='learner_collection',game=game,phase=phase,episodes=count,fork=fork,samples=samples,next_episode_id=group.next_episode_id,outcomes=group.outcomes))
        print('Episode collection fork',fork,game,'complete',flush=True)
    (output/f'episodes-{fork}.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('docs/benchmarks/research'));parser.add_argument('--episodes',type=int);args=parser.parse_args()
    if args.episodes:episodes(args.output,args.episodes)
    else:cli(args.output)

if __name__=='__main__':main()
