"""Aggregate process repetitions and export standalone scientific plots.

Requires NumPy and Matplotlib. Error bars show the range of process medians,
not 15 independent batch measurements. See the accompanying report for scopes.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[3]/'.scratch/research-bench/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import PROFILES,COMMON,EXTRA
from figure_layout import layout_with_legend

COLORS={'rust':'#188477','java':'#7c5db3'}

def read(path):return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def aggregate(records):
    groups=defaultdict(list)
    for row in records:
        samples=[s['elapsed_ns']/s['calls'] for s in row['samples'] if 'elapsed_ns' in s]
        if not samples:continue
        key=(row['engine'],row['game'],row['players'],row['operation'])
        groups[key].append(dict(fork=row['fork'],median_ns=statistics.median(samples)))
    result=[]
    for key,items in sorted(groups.items()):
        assert len(items)==3 and len({x['fork'] for x in items})==3,key
        values=[x['median_ns'] for x in items]
        result.append(dict(zip(('engine','game','players','operation'),key),median_ns=statistics.median(values),min_fork_ns=min(values),max_fork_ns=max(values),forks=items))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',nargs='?',type=Path,default=Path('docs/benchmarks/research'));args=parser.parse_args();root=args.directory
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b7c5d1','axes.labelcolor':'#233d52','text.color':'#233d52','xtick.color':'#526780','ytick.color':'#526780','axes.grid':True,'grid.alpha':.22,'grid.color':'#a6b5c5','pdf.fonttype':42,'savefig.facecolor':'white'})
    plots=root/'plots';plots.mkdir(exist_ok=True)
    def save(fig,name):
        layout_with_legend(fig)
        for ext in ['svg','pdf','png']:fig.savefig(plots/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    records=read(root/'timing.jsonl');assert len(records)==len(PROFILES)*(2*len(COMMON)+len(EXTRA))*3
    summary=aggregate(records)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['engine','game','players','operation','median_ns','min_fork_ns','max_fork_ns']);writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='forks'} for r in summary)
    lookup={(r['engine'],r['game'],r['players'],r['operation']):r for r in summary}
    def bars(ax,game,players,ops):
        x=np.arange(len(ops));width=.36
        for offset,engine in [(-.18,'rust'),(.18,'java')]:
            rows=[lookup[engine,game,players,op] for op in ops];means=np.array([r['median_ns']/1000 for r in rows]);low=[r['min_fork_ns']/1000 for r in rows];high=[r['max_fork_ns']/1000 for r in rows]
            ax.bar(x+offset,means,width,label='Rust' if engine=='rust' else 'Java TAG',color=COLORS[engine],yerr=[means-low,high-means],capsize=3)
        ax.set_xticks(x,ops);ax.set_yscale('log');ax.set_ylabel(f'{game} ({players}p)\nµs / call');ax.set_axisbelow(True)
    fig,axes=plt.subplots(4,1,figsize=(11,12))
    for ax,(game,players) in zip(axes,[('tictactoe',2),('connect4',2),('cantstop',2),('loveletter',2)]):bars(ax,game,players,['init','copy','observe','actions','transition','query','fork'])
    axes[0].legend(frameon=False,ncol=2);save(fig,'core-operations')
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for ax,(game,players) in zip(axes.flat,PROFILES):bars(ax,game,players,['first','random','osla','mcts']);ax.tick_params(axis='x',rotation=0)
    axes.flat[0].legend(frameon=False,ncol=2);save(fig,'agent-decisions')
    allocations=read(root/'allocations.jsonl');assert len(allocations)==len(PROFILES)*(2*len(COMMON)+len(EXTRA))
    allocation_summary=[]
    for r in allocations:
        samples=r['samples'][:-1]
        allocation_summary.append(dict(engine=r['engine'],game=r['game'],players=r['players'],operation=r['operation'],bytes_per_call=statistics.median(s['allocated_bytes']/s['calls'] for s in samples),calls_per_call=statistics.median(s.get('allocation_calls',0)/s['calls'] for s in samples),peak_extra_live_bytes=max(s.get('peak_live_bytes',0)-s.get('live_before',0) for s in samples)))
    (root/'allocation-summary.json').write_text(json.dumps(allocation_summary,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(len(PROFILES))
    for offset,engine in [(-.18,'rust'),(.18,'java')]:
        values=[next(r['bytes_per_call']/1024 for r in allocation_summary if (r['engine'],r['game'],r['players'],r['operation'])==(engine,g,p,'copy')) for g,p in PROFILES]
        ax.bar(x+offset,values,.36,color=COLORS[engine],label='Rust requested bytes' if engine=='rust' else 'Java thread-allocated bytes')
    ax.set_xticks(x,[f'{g}\n{p} players' for g,p in PROFILES]);ax.set_yscale('log');ax.set_ylabel('KiB allocated / state-copy API call');ax.legend(frameon=False);save(fig,'state-copy-allocations')
    retention=read(root/'retention.jsonl');assert len(retention)==len(PROFILES)*2*3*3
    memory=[]
    for r in retention:
        idle,loaded=r['samples'];a=idle['process'];b=loaded['process']
        memory.append(dict(engine=r['engine'],game=r['game'],players=r['players'],fork=r['fork'],count=r['count'],idle=a,loaded=b,private_delta=b['private_bytes']-a['private_bytes'],working_set_delta=b['working_set']-a['working_set'],heap_or_requested_delta=(loaded['heap_used_bytes']-idle['heap_used_bytes']) if r['engine']=='java' else loaded['alloc'][2]-loaded['before'][2]))
    (root/'memory-summary.json').write_text(json.dumps(memory,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    for ax,field in zip(axes,['private_bytes','working_set']):
        for offset,engine in [(-.18,'rust'),(.18,'java')]:
            values=[statistics.median(r['loaded'][field]/2**20 for r in memory if (r['engine'],r['game'],r['players'],r['count'])==(engine,g,p,2048)) for g,p in PROFILES]
            base=[statistics.median(r['idle'][field]/2**20 for r in memory if (r['engine'],r['game'],r['players'],r['count'])==(engine,g,p,2048)) for g,p in PROFILES]
            ax.bar(x+offset,values,.36,color=COLORS[engine],label=engine);ax.scatter(x+offset,base,marker='_',s=180,c='#1d2b3d',zorder=4)
        ax.set_yscale('log');ax.set_xticks(x,[f'{g}\n{p}p' for g,p in PROFILES]);ax.set_ylabel(('Private committed memory' if field=='private_bytes' else 'Resident working set')+' (MiB)');ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    handles,labels=axes[0].get_legend_handles_labels();handles.append(Line2D([],[],color='#1d2b3d',marker='_',linestyle='None',markersize=12));labels.append('Before retaining state copies')
    axes[0].legend(handles,labels,frameon=False,ncol=3);save(fig,'process-memory-2048-states')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(game,players) in zip(axes.flat,[('tictactoe',2),('connect4',2),('cantstop',2),('loveletter',2)]):
        for engine in ['rust','java']:
            counts=[1,128,2048];values=[statistics.median(r['heap_or_requested_delta'] for r in memory if (r['engine'],r['game'],r['players'],r['count'])==(engine,game,players,n))/1024 for n in counts]
            ax.plot(counts,values,'o-',color=COLORS[engine],label='Rust live requested' if engine=='rust' else 'Java post-GC heap delta')
        ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1);ax.set_xlabel(f'{game}: retained state count');ax.set_ylabel('KiB retained');ax.set_xticks([1,128,2048],['1','128','2048'])
    axes.flat[0].legend(frameon=False,fontsize=8);save(fig,'retained-state-scaling')
    py=[row for fork in (1,2,3) for row in read(root/f'python-{fork}.jsonl')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for game,players in PROFILES:
        rows=[r for r in py if r['operation']=='python_neural_mcts' and (r['game'],r['players'],r['iterations'])==(game,players,32)]
        values=[statistics.median(statistics.median(s['elapsed_ns']/s['calls']/r['parallel']/1e6 for s in r['samples']) for r in rows if r['parallel']==n) for n in [1,4,16]]
        axes[0].plot([1,4,16],values,'o-',label=f'{game} {players}p')
        structures=[r for r in py if r['operation']=='search_structure' and r['neural'] and (r['game'],r['players'],r['parallel'])==(game,players,1)]
        nodes=[statistics.median(r['reports'][0]['nodes'] for r in structures if r['iterations']==n) for n in [8,32,128]]
        axes[1].plot([8,32,128],nodes,'o-')
    axes[0].set_xlabel('Independent searches in one batch');axes[0].set_ylabel('ms / completed neural decision');axes[0].set_xticks([1,4,16]);axes[0].legend(frameon=False,fontsize=8,ncol=2)
    axes[1].set_xlabel('Iteration budget');axes[1].set_ylabel('Nodes in completed search report');axes[1].set_xticks([8,32,128]);save(fig,'neural-batch-and-tree-scaling')
    adapter=[r for fork in (1,2,3) for r in read(root/f'adapter-{fork}.jsonl')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for game,players in PROFILES:
        rows=[r for r in adapter if r['operation']=='session_history_scaling' and (r['game'],r['players'])==(game,players)]
        histories=rows[0]['histories'];xs=[h['cursor'] for h in histories];ys=[h['file_bytes']/1024 for h in histories]
        axes[0].plot(xs,ys,'o-',label=f'{game} {players}p')
        times=[statistics.median(statistics.median(s['elapsed_ns']/s['calls']/1e6 for s in r['histories'][i]['samples']) for r in rows) for i in range(len(histories))]
        axes[1].plot(xs,times,'o-')
    axes[0].set_xlabel('Actual recorded move count');axes[0].set_ylabel('Session export size (KiB)');axes[0].set_yscale('log');axes[0].legend(frameon=False,fontsize=8,ncol=2)
    axes[1].set_xlabel('Actual recorded move count');axes[1].set_ylabel('ms / session import');axes[1].set_yscale('log');save(fig,'history-scaling')
    # Reuse the validated corpus harness's separate process/batch format.
    matched=[]
    for engine in ['rust','java']:
        for fork in [1,2,3]:
            for line in (root/'matched'/f'{engine}-fork-{fork}.tsv').read_text(encoding='utf-8-sig').splitlines():
                if line.startswith('sample\t'):
                    _,_,game,op,batch,calls,ns=line.split('\t');matched.append(dict(engine=engine,game=game,players=2,operation=op,fork=fork,samples=[dict(calls=1,elapsed_ns=float(ns))]))
    clustered=defaultdict(list)
    for r in matched:clustered[r['engine'],r['game'],r['operation'],r['fork']].extend(r['samples'])
    matched=aggregate([dict(engine=e,game=g,players=2,operation=o,fork=f,samples=s) for (e,g,o,f),s in clustered.items()])
    (root/'matched-summary.json').write_text(json.dumps(matched,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    ratios=[]
    for row,game in enumerate(['ttt','c4']):
        ops=['init','actions','copy','transition','replay'];x=np.arange(len(ops))
        for offset,engine in [(-.18,'rust'),(.18,'java')]:
            selected=[next(r for r in matched if (r['engine'],r['game'],r['operation'])==(engine,game,op)) for op in ops]
            values=np.array([r['median_ns']/1000 for r in selected]);low=np.array([r['min_fork_ns']/1000 for r in selected]);high=np.array([r['max_fork_ns']/1000 for r in selected]);axes[row,0].bar(x+offset,values,.36,color=COLORS[engine],label=engine,yerr=[values-low,high-values],capsize=3)
        for op in ops:
            r=next(r for r in matched if (r['engine'],r['game'],r['operation'])==('rust',game,op));j=next(r for r in matched if (r['engine'],r['game'],r['operation'])==('java',game,op));pairs=[b['median_ns']/a['median_ns'] for a,b in zip(r['forks'],j['forks'])]
            ratios.append(dict(game=game,operation=op,java_over_rust=j['median_ns']/r['median_ns'],min_pair_ratio=min(pairs),max_pair_ratio=max(pairs)))
        values=[r['java_over_rust'] for r in ratios if r['game']==game];selected=[r for r in ratios if r['game']==game];lower=[max(0,v-r['min_pair_ratio']) for v,r in zip(values,selected)];upper=[max(0,r['max_pair_ratio']-v) for v,r in zip(values,selected)];axes[row,1].bar(x,values,color='#456e9b',yerr=[lower,upper],capsize=3);axes[row,1].axhline(1,color='#526780',linestyle='--')
        for ax in axes[row]:ax.set_xticks(x,ops);ax.set_yscale('log')
        axes[row,0].set_ylabel(f'{game}: µs / call or replay');axes[row,1].set_ylabel('Java TAG time / Rust time')
    axes[0,0].legend(frameon=False,ncol=2);save(fig,'matched-board-corpus')
    (root/'matched-ratios.json').write_text(json.dumps(ratios,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(len(PROFILES))
    for offset,op,color in [(-.18,'fork','#188477'),(.18,'fork_and_continue','#547fbd')]:
        values=[statistics.median(statistics.median(s['elapsed_ns']/s['calls']/1000 for s in r['samples']) for r in adapter if (r['game'],r['players'],r['operation'])==(g,p,op)) for g,p in PROFILES]
        ax.bar(x+offset,values,.36,color=color,label='Fork + one accepted move' if op=='fork_and_continue' else 'Fork only')
    ax.set_xticks(x,[f'{g}\n{p}p' for g,p in PROFILES]);ax.set_ylabel('Microseconds / prepared-session operation');ax.legend(frameon=False);save(fig,'branch-continuation')
    if (root/'tree-memory.json').exists():
        tree=json.loads((root/'tree-memory.json').read_text());assert len(tree)==96
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for game,players in PROFILES:
            for engine,ax in zip(['rust','java'],axes):
                values=[]
                for budget in [32,128]:
                    rows=[r for r in tree if (r['engine'],r['game'],r['players'],r['iterations'])==(engine,game,players,budget)]
                    if engine=='rust':values.append(statistics.median(max(s['peak_live_bytes']-s['live_before'] for s in r['samples'][:-1]) for r in rows)/1024)
                    else:values.append(statistics.median(r['loaded']['heap_used_bytes']-r['idle']['heap_used_bytes'] for r in rows)/1024)
                ax.plot([32,128],values,'o-',label=f'{game} {players}p')
        for ax in axes:ax.set_xticks([32,128]);ax.set_xlabel('Iteration budget');ax.set_yscale('symlog',linthresh=1)
        axes[0].set_ylabel('Rust: temporary peak requested KiB');axes[1].set_ylabel('Java: retained post-GC heap delta KiB');axes[0].legend(frameon=False,ncol=2,fontsize=8);save(fig,'search-memory-lifetimes')
    if (root/'cli.json').exists():
        cli=json.loads((root/'cli.json').read_text());fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(len(PROFILES))
        for offset,include,color in [(-.18,False,'#188477'),(.18,True,'#547fbd')]:
            values=[statistics.median(r['elapsed_ns']/r['completed_games']/1e6 for r in cli if (r['game'],r['players'],r['operation'],r.get('include_actions'))==(g,p,'match_cli',include)) for g,p in PROFILES]
            ax.bar(x+offset,values,.36,color=color,label='With action arrays' if include else 'Without action arrays')
        ax.set_xticks(x,[f'{g}\n{p}p' for g,p in PROFILES]);ax.set_ylabel('Startup-inclusive ms / game (64-game run)');ax.legend(frameon=False);save(fig,'runner-reporting-cost')
    if (root/'episodes-1.json').exists():
        episodes=[r for fork in [1,2,3] for r in json.loads((root/f'episodes-{fork}.json').read_text())];fig,ax=plt.subplots(figsize=(9,4.5));profiles=[(g,p) for g in ['tictactoe','connect4'] for p in ['teacher','actor_critic']];x=np.arange(len(profiles))
        for offset,count,color in [(-.18,1,'#547fbd'),(.18,16,'#188477')]:
            values=[1e9/statistics.median(statistics.median(s['elapsed_ns']/s['calls']/r['episodes'] for s in r['samples']) for r in episodes if (r['game'],r['phase'],r['episodes'])==(g,p,count)) for g,p in profiles]
            ax.bar(x+offset,values,.36,color=color,label=f'{count} episode group')
        ax.set_xticks(x,[f'{g}\n{p}' for g,p in profiles]);ax.set_ylabel('Completed learner episodes / second');ax.legend(frameon=False);save(fig,'episode-collection')
    print('Wrote summaries and SVG/PDF/PNG plot sets. No figure titles are added.')


if __name__=='__main__':main()
