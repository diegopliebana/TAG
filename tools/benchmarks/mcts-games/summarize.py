"""Validate complete shards and report MCTS-game API timings and process memory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import run

NAMES={'tictactoe':'Tic-Tac-Toe','connect4':'Connect 4','cantstop':"Can't Stop",'loveletter':'Love Letter'}
LABELS={'init':'Initialize','actions':'Actions','copy':'Exact copy probe','observe':'Observation/player copy','copy_apply':'Copy + apply probe','apply':'In-place apply','fork':'Simulation fork','actual_copy':'Actual exact-copy calls'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def exclusion_key(row):
    return tuple(row[k] for k in ('engine','game','players','shard','id'))

def pooled_game_mean(rows,field):
    totals={'mean_game_seconds':('total_elapsed_ns',1_000_000_000),'mean_decisions':('total_decisions',1),
            'mean_searched_decisions':('total_searched_decisions',1)}
    total,scale=totals[field]
    return sum(r[total] for r in rows)/(sum(r['games'] for r in rows)*scale)

def pooled_api_mean(rows):
    samples=sum(r['samples'] for r in rows)
    return sum(r['sampled_ns'] for r in rows)/samples if samples else None

def included_games(result,exclusions):
    identity=result['identity'];kept=[];removed=[]
    for game in result['games']:
        key=exclusion_key({**identity,'id':game['id']})
        if key in exclusions:
            assert exclusions[key]['seed']==game['seed'],'Exclusion seed does not match raw game'
            removed.append(key)
        else:kept.append(game)
    assert kept,'A shard has no included games; its statistics cannot be estimated'
    return kept,removed

def summarize(root):
    settings=json.loads((root/'settings.json').read_text())
    assert all(settings[k]=='true' for k in ('aligned','seed_copies','skip_forced')),'This report describes the selected aligned/seeded/forced-action-skipping setup only'
    n=settings['games_per_profile'];shards=settings['shards'];records={};api=[];work=[]
    assert len(run.shard_sizes(settings))==shards and sum(run.shard_sizes(settings))==n
    exclusion_file=root/'exclusions.json'
    exclusion_document=json.loads(exclusion_file.read_text()) if exclusion_file.exists() else {'version':1,'games':[]}
    assert exclusion_document['version']==1
    exclusions={exclusion_key(g):g for g in exclusion_document['games']}
    assert len(exclusions)==len(exclusion_document['games']),'Duplicate exclusions'
    matched_exclusions=set();memory_records=[]
    expected={(engine,game,p,s) for engine in ('rust','java') for game,p in run.PROFILES for s in range(shards)}
    for path in sorted(root.glob('*-shard*.json')):
        if path.name.endswith('.baseline.json'):continue
        result=json.loads(path.read_text());i=result['identity'];key=(i['engine'],i['game'],i['players'],i['shard'])
        assert key not in records and key in expected,key
        assert run.measurement_settings(i['settings'])==run.measurement_settings(settings)
        assert len(result['games'])==run.shard_sizes(settings)[i['shard']]
        assert [g['id'] for g in result['games']]==list(range(run.shard_sizes(settings)[i['shard']]))
        assert i['seed']==settings['seed']+sum(run.shard_sizes(settings)[:i['shard']])
        for g in result['games']:
            assert g['elapsed_ns']>0 and 0<g['searched_decisions']<=g['decisions']
            assert g['seed']==i['seed']+g['id']
            metrics={(m['scope'],m['operation']):m for m in g['metrics']}
            assert metrics['real','init']['calls']==1
            assert metrics['real','apply']['calls']==g['decisions']
            assert metrics['real','copy_apply']['samples']==g['decisions']
            assert not metrics['search','init']['calls']
            assert all(g['memory_peak'][f]>0 for f in ('working_set','private_bytes'))
        records[key]=result
        included,removed=included_games(result,exclusions)
        segments=result.get('process_segments',[{'baseline':result['baseline']}])
        for index,segment in enumerate(segments):
            members=[g for g in included if g.get('process_segment',0)==index]
            if not members:continue
            memory_records.append({'engine':i['engine'],'game':i['game'],'players':i['players'],'shard':i['shard'],
                'process_segment':index,'included_games':len(members),
                **{f'baseline_{f}':segment['baseline'][f] for f in ('working_set','private_bytes')},
                **{f'peak_{f}':max(g['memory_peak'][f] for g in members) for f in ('working_set','private_bytes')}})
        matched_exclusions.update(removed)
        for scope in ('real','search'):
            for op in LABELS:
                entries=[m for g in included for m in g['metrics'] if (m['scope'],m['operation'])==(scope,op)]
                samples=sum(m['samples'] for m in entries);ns=sum(m['ns'] for m in entries)
                api.append({'engine':i['engine'],'game':i['game'],'players':i['players'],'shard':i['shard'],'scope':scope,'operation':op,
                            'calls':sum(m['calls'] for m in entries),'samples':samples,'sampled_ns':ns,'mean_ns':ns/samples if samples else None})
        work.append({'engine':i['engine'],'game':i['game'],'players':i['players'],'shard':i['shard'],
                     'collected_games':len(result['games']),'excluded_games':len(removed),'games':len(included),
                     'total_elapsed_ns':sum(g['elapsed_ns'] for g in included),
                     'total_decisions':sum(g['decisions'] for g in included),
                     'total_searched_decisions':sum(g['searched_decisions'] for g in included),
                     'total_search_ns':sum(g['search_ns'] for g in included),
                     'total_probe_ns':sum(g['probe_ns'] for g in included),
                     'mean_game_seconds':statistics.mean(g['elapsed_ns']/1e9 for g in included),
                     'mean_decisions':statistics.mean(g['decisions'] for g in included),
                     'mean_searched_decisions':statistics.mean(g['searched_decisions'] for g in included),
                     'ms_per_searched_decision':sum(g['search_ns'] for g in included)/sum(g['searched_decisions'] for g in included)/1e6,
                     'probe_fraction':sum(g['probe_ns'] for g in included)/sum(g['elapsed_ns'] for g in included),
                     'timer_control_ns':result['ready']['timer_control_ns'],
                     **{f'baseline_{f}':result['baseline'][f] for f in ('working_set','private_bytes')},
                     **{f'peak_{f}':max(g['memory_peak'][f] for g in included) for f in ('working_set','private_bytes')}})
    assert set(records)==expected,f'Incomplete: {len(records)}/{len(expected)} shards; no final report written'
    assert matched_exclusions==set(exclusions),'An exclusion does not match a collected game'
    included_total=sum(r['games'] for r in work)
    (root/'api-summary.json').write_text(json.dumps(api,indent=2)+'\n')
    (root/'game-summary.json').write_text(json.dumps(work,indent=2)+'\n')
    (root/'process-memory-summary.json').write_text(json.dumps(memory_records,indent=2)+'\n')
    for name,rows in [('api-summary',api),('game-summary',work)]:
        with (root/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    profiles=[]
    for engine in ('rust','java'):
        for game,p in run.PROFILES:
            rows=[r for r in work if (r['engine'],r['game'],r['players'])==(engine,game,p)]
            profiles.append({'engine':engine,'game':game,'players':p,
                'collected_games':sum(r['collected_games'] for r in rows),'included_games':sum(r['games'] for r in rows),
                'excluded_games':sum(r['excluded_games'] for r in rows),'total_elapsed_ns':sum(r['total_elapsed_ns'] for r in rows),
                **{field:pooled_game_mean(rows,field) for field in ('mean_game_seconds','mean_decisions','mean_searched_decisions')}})
    (root/'profile-summary.json').write_text(json.dumps(profiles,indent=2)+'\n')
    with (root/'profile-summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(profiles[0]));writer.writeheader();writer.writerows(profiles)
    def value(engine,game,p,scope,op):
        rows=[r for r in api if (r['engine'],r['game'],r['players'],r['scope'],r['operation'])==(engine,game,p,scope,op)]
        return pooled_api_mean(rows)
    def summary(engine,game,p,field):
        source=memory_records if field.startswith(('baseline_','peak_')) else work
        rows=[r for r in source if (r['engine'],r['game'],r['players'])==(engine,game,p)]
        if field.startswith('mean_'):return pooled_game_mean(rows,field)
        return statistics.median(r[field] for r in rows)
    text=f'''# Complete all-MCTS games: API costs and process memory

Both implementations completed **{n:,} games per game/player setting**:
{n*16:,} collected games in total, of which **{included_total:,} are included in
statistics**. Explicitly identified interrupted games are retained in the raw
records and excluded from all game/API/memory summaries without replacements.
The per-setting counts below identify the actual sample sizes. Every searched decision completed
{settings['iterations']:,} iterations, with random rollouts limited to 10 actions.
Forced actions execute directly without search. All players use native MCTS,
with UCB exploration sqrt(2), final selection by visits and tree depth 1,000.
Configurable rules are aligned; Java Love Letter uses its information-set mode,
Rust its existing observation-derived forks. Neither game trajectories nor
search algorithms/heuristics are claimed equivalent.

Each engine/profile uses {shards} sequential fresh processes, with batch sizes
{', '.join(map(str,run.shard_sizes(settings)))} games,
with {settings['warmup_decisions']} untimed search decisions per process. Engines
do not run concurrently. Java uses a 64 MiB initial/512 MiB maximum heap.
The recorded master seed is {settings['seed']}. Parameters, seed derivations,
measurement hooks and reproducibility checks are described in the
[tool methodology](../../../../tools/benchmarks/mcts-games/README.md).

## Real execution APIs

Values are microseconds per call: total sampled nanoseconds divided by the number
of timed calls, pooled across batches. Batch means are not given equal weight.
The first column is Java/Rust time; above 1 means Rust was faster in these
encountered positions. Real calls are timed throughout each game. Exact-copy
and copy+apply rows are independent shadow probes at pre-transition positions,
not extra actions or calls attributed to the actual agent. Initialization includes
state construction/setup, excluding parameter construction in both engines.

'''
    counts=[]
    for game,p in run.PROFILES:
        count=lambda e,f:sum(r[f] for r in work if (r['engine'],r['game'],r['players'])==(e,game,p))
        counts.append([NAMES[game],p,count('rust','games'),count('java','games'),count('rust','excluded_games'),count('java','excluded_games')])
    count_table=table(['Game','Players','Rust included','Java included','Rust excluded','Java excluded'],counts)
    text=text.replace('## Real execution APIs','## Included samples\n\n'+count_table+'\n\n'+
        ('[Exclusion records and reasons](exclusions.json).\n\n' if exclusions else 'No games excluded.\n\n')+'## Real execution APIs',1)
    headers=['Java/Rust time','Game','Players','Operation','Rust µs','Java µs']
    for scope,title,ops in [('real',None,('init','actions','copy','observe','copy_apply')),
                            ('search','Search-position APIs',('actions','copy','observe','copy_apply','apply','fork','actual_copy'))]:
        if title:text+='\n## '+title+'\n\nSearch API calls are counted throughout. Latencies sample the first 16 calls\nper game/category, then every '+str(settings['sampling_stride'])+'th call. Copy probes sample\npre-transition search positions; actual copy-call timings are separate. Rust\nuses observation-derived forks rather than exact clones to start simulations.\nJava player copies can include hidden-state sampling. A missing entry means that\nAPI was not executed, not that it costs zero. Search does not initialize games.\n\n'
        rows=[]
        for game,p in run.PROFILES:
            for op in ops:
                r,j=(value(e,game,p,scope,op) for e in ('rust','java'))
                if r is None and j is None:continue
                rows.append([f'{j/r:.2f}' if r and j else '—',NAMES[game],p,LABELS[op],f'{r/1000:.4f}' if r is not None else '—',f'{j/1000:.4f}' if j is not None else '—'])
        text+=table(headers,rows)+'\n'
    text+='\n## Instrumented complete-game execution\n\nThese times include API counters/timers and shadow probes, agent setup and real\nplay. Probe duration is recorded separately; it is not subtracted to invent an\nuninstrumented result. A search decision can contain many simulated actions.\nGame length and search work differ between engines even with aligned settings.\n\n'
    text+='Game means pool all included games: total game duration divided by the\nincluded game count (250 normally, 248 for the setting with two exclusions).\nThe 100/100/50 batches are weighted by their game counts. Per-call API means\nuse timed-call counts instead. [Pooled totals and denominators](profile-summary.csv)\nmake the computation explicit.\n\n'
    rows=[]
    for game,p in run.PROFILES:
        r,j=(summary(e,game,p,'mean_game_seconds') for e in ('rust','java'))
        rows.append([f'{j/r:.2f}',NAMES[game],p,f'{r:.3f}',f'{j:.3f}',f"{summary('rust',game,p,'mean_decisions'):.1f}",f"{summary('java',game,p,'mean_decisions'):.1f}"])
    text+=table(['Java/Rust time','Game','Players','Rust s/game','Java s/game','Rust actions/game','Java actions/game'],rows)+'\n'
    text+='\n## Process memory while playing\n\nEach value is the median across processes. Peak means the largest sampled value\nduring a process\'s measured games, with 50 ms polling. Shorter-lived peaks can be\nmissed. Baselines follow warmup. Private committed memory is storage committed\nto the process; resident working set is currently resident RAM, including shared\npages. Existing available heap space can accommodate objects without increasing\nprocess memory. These values include runtimes, search trees, allocators and\ninstrumentation; they are not state sizes.\n\n'
    rows=[]
    for game,p in run.PROFILES:
        for field,label in [('private_bytes','Private committed'),('working_set','Resident working set')]:
            rb,jb=(summary(e,game,p,'baseline_'+field)/2**20 for e in ('rust','java'))
            rp,jp=(summary(e,game,p,'peak_'+field)/2**20 for e in ('rust','java'))
            rows.append([f'{jp/rp:.2f}',NAMES[game],p,label,f'{rb:.2f}',f'{rp:.2f}',f'{jb:.2f}',f'{jp:.2f}'])
    text+=table(['Java/Rust peak','Game','Players','Metric','Rust baseline MiB','Rust peak MiB','Java baseline MiB','Java peak MiB'],rows)+'\n'
    text+='\n## Data and limits\n\n[Settings](settings.json), [API summaries](api-summary.csv), [game/memory summaries](game-summary.csv)\nand per-shard JSON files retain all measured games. Counts distinguish actual\nAPI calls from shadow-copy samples. The original prepared-state study remains\na separate workload. Tiny per-call timings include timer overhead; process\ntimer controls and probe fractions are in the game summary. Rules-version-2\nRust Can\'t Stop resolves busts within Roll, while Java retains its forced bust\naction and its other documented rule/knowledge differences. No language-wide\nspeed or memory multiplier follows from this experiment.\n'
    if (root/'plots/ratios/README.md').exists():
        text=text.replace('\n## Data and limits',"\n## Ratios by player count\n\n[Speed and memory plots for Can't Stop and Love Letter](plots/ratios/README.md)\nshow Java/Rust ratios against the number of players, with equality at 1.\n\n## Data and limits")
    (root/'README.md').write_text(text,encoding='utf-8')
    validation={'complete':True,'profiles':16,'games_per_profile':n,'games':n*16,'included_games':included_total,
                'excluded_games':len(exclusions),'included_counts':counts,'shards':len(records),'iteration_budget':settings['iterations']}
    (root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    plan_file=root/'collection-plan.json'
    if plan_file.exists():
        plan=json.loads(plan_file.read_text());plan.update(state='complete',collected_games=n*16,included_games=included_total)
        plan_file.write_text(json.dumps(plan,indent=2)+'\n')
    report_sources={str(p.relative_to(run.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).resolve(),Path(__file__).resolve().parent.parent/'research/report_remeasurement.py')}
    if exclusion_file.exists():report_sources['exclusions.json']=hashlib.sha256(exclusion_file.read_bytes()).hexdigest()
    (root/'report-source-hashes.json').write_text(json.dumps(report_sources,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*')) if p.is_file() and p.name!='checksums.json'}
    (root/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(validation))
    if root.parent.name=='research':
        subprocess.run([sys.executable,str(Path(__file__).parent.parent/'research/report_remeasurement.py'),str(root.parent)],check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();summarize(a.directory)

if __name__=='__main__':main()
