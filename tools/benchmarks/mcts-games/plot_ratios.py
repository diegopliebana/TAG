"""Plot Java/Rust complete-game time and peak process-memory ratios by player count."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.scratch/mcts-ratio-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BOOTSTRAP_SEED=20261030
BOOTSTRAP_REPLICATES=20000

def bootstrap(values,rng,statistic='mean'):
    """Independent resampling; engine samples are never paired by seed."""
    values=np.asarray(values,dtype=float)
    samples=values[rng.integers(0,len(values),size=(BOOTSTRAP_REPLICATES,len(values)))]
    return np.mean(samples,axis=1) if statistic=='mean' else np.median(samples,axis=1)

def interval(distribution):
    return tuple(float(x) for x in np.quantile(distribution,[.025,.975]))

def plot(root):
    validation=json.loads((root/'validation.json').read_text())
    if not validation['complete']:raise ValueError('A validated complete study is required')
    profiles=json.loads((root/'profile-summary.json').read_text())
    memory=json.loads((root/'process-memory-summary.json').read_text())
    exclusions=json.loads((root/'exclusions.json').read_text())['games'] if (root/'exclusions.json').exists() else []
    excluded={tuple(g[k] for k in ('engine','game','players','shard','id')) for g in exclusions}
    times={};individual_games=[]
    for path in sorted(root.glob('*-shard*.json')):
        if path.name.endswith('.baseline.json'):continue
        raw=json.loads(path.read_text());i=raw['identity'];key=(i['engine'],i['game'],i['players'])
        times.setdefault(key,[]).extend(g['elapsed_ns']/1e9 for g in raw['games']
            if (*key,i['shard'],g['id']) not in excluded)
        individual_games.extend({'engine':key[0],'game':key[1],'players':key[2],'shard':i['shard'],
                                 'id':g['id'],'seed':g['seed'],'seconds':g['elapsed_ns']/1e9}
                                for g in raw['games'] if (*key,i['shard'],g['id']) not in excluded)
    rng=np.random.default_rng(BOOTSTRAP_SEED);distributions={};time_rows=[]
    for row in profiles:
        key=(row['engine'],row['game'],row['players']);values=times[key]
        if len(values)!=row['included_games'] or not np.isclose(np.mean(values),row['mean_game_seconds']):
            raise ValueError('Raw games differ from validated pooled summary')
        dist=bootstrap(values,rng);distributions[(*key,'mean_game_seconds')]=dist
        low,high=interval(dist)
        time_rows.append({'engine':key[0],'game':key[1],'players':key[2],'included_games':len(values),
                          'mean_seconds':row['mean_game_seconds'],'ci95_low_seconds':low,'ci95_high_seconds':high})
        for metric in ('peak_private_bytes','peak_working_set'):
            values=[r[metric] for r in memory if (r['engine'],r['game'],r['players'])==key]
            distributions[(*key,metric)]=bootstrap(values,rng,'median')
    folder=root/'plots/ratios';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=[]
    def value(engine,game,players,metric):
        source=profiles if metric=='mean_game_seconds' else memory
        selected=[r[metric] for r in source if (r['engine'],r['game'],r['players'])==(engine,game,players)]
        return selected[0] if metric=='mean_game_seconds' else statistics.median(selected)
    for game,label in [('cantstop',"Can't Stop"),('loveletter','Love Letter')]:
        for kind,metrics,ylabel in [
            ('speed',[('mean_game_seconds','Mean complete-game time','#188477','o')],
             'Speed ratio (Java time / Rust time)'),
            ('memory',[('peak_private_bytes','Peak private committed','#188477','o'),
                       ('peak_working_set','Peak resident working set','#7c5db3','s')],
             'Memory ratio (Java peak / Rust peak)')]:
            fig,ax=plt.subplots(figsize=(7.2,4.6))
            top=1
            for metric,legend,color,marker in metrics:
                ratios=[];lows=[];highs=[]
                for players in (2,3,4):
                    rust,java=(value(e,game,players,metric) for e in ('rust','java'))
                    ratio=java/rust;ratios.append(ratio)
                    low,high=interval(distributions[('java',game,players,metric)]/distributions[('rust',game,players,metric)])
                    lows.append(low);highs.append(high)
                    rows.append({'game':game,'players':players,'metric':metric,'rust':rust,'java':java,
                                 'java_over_rust':ratio,'ci95_low':low,'ci95_high':high})
                top=max(top,*highs)
                ax.errorbar([2,3,4],ratios,yerr=[np.array(ratios)-lows,np.array(highs)-ratios],
                    label=legend,color=color,marker=marker,linewidth=2.2,markersize=7,capsize=5,elinewidth=1.4)
                for players,ratio,high in zip((2,3,4),ratios,highs):
                    ax.annotate(f'{ratio:.2f}×',(players,high),xytext=(0,9),textcoords='offset points',
                                ha='center',fontsize=10,color=color)
            ax.axhline(1,color='#526071',linewidth=1.3,linestyle=(0,(4,3)))
            ax.text(.98,1,'Equal = 1',transform=ax.get_yaxis_transform(),ha='right',va='bottom',
                    fontsize=9,color='#526071',bbox={'facecolor':'white','edgecolor':'none','pad':2})
            ax.set(xlabel='Number of players',ylabel=ylabel,xlim=(1.8,4.2),ylim=(0,top*1.19),xticks=[2,3,4])
            ax.grid(axis='y',alpha=.18)
            handles,labels=ax.get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),frameon=False,ncol=1,fontsize=10)
            fig.tight_layout(rect=(0,0,1,.88 if kind=='memory' else .93))
            for extension in ('png','svg','pdf'):
                fig.savefig(folder/f'{game}-{kind}.{extension}',dpi=180,bbox_inches='tight')
            plt.close(fig)
    with (folder/'ratios.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (folder/'times.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(time_rows[0]));writer.writeheader();writer.writerows(time_rows)
    with (folder/'individual-game-times.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(individual_games[0]));writer.writeheader();writer.writerows(individual_games)
    variants=[('tictactoe',2),('connect4',2),('cantstop',2),('cantstop',3),('cantstop',4),
              ('loveletter',2),('loveletter',3),('loveletter',4)]
    labels=['Tic-Tac-Toe\n2p','Connect 4\n2p',"Can't Stop\n2p","Can't Stop\n3p","Can't Stop\n4p",
            'Love Letter\n2p','Love Letter\n3p','Love Letter\n4p']
    def variant_guides(ax):
        for index in range(8):
            if index%2==0:ax.axvspan(index-.5,index+.5,color='#eef2f6',alpha=.65,zorder=0)
            ax.axvline(index,color='#9aa7b5',linestyle=(0,(2,4)),linewidth=.8,alpha=.6,zorder=0)
        ax.set_xlim(-.5,7.5)
    for scale in ('linear','log'):
        fig,ax=plt.subplots(figsize=(12,5))
        variant_guides(ax)
        for engine,color,offset in [('rust','#188477',-.14),('java','#7c5db3',.14)]:
            selected=[next(r for r in time_rows if (r['engine'],r['game'],r['players'])==(engine,*v)) for v in variants]
            means=np.array([r['mean_seconds'] for r in selected])
            errors=[means-[r['ci95_low_seconds'] for r in selected],np.array([r['ci95_high_seconds'] for r in selected])-means]
            ax.errorbar(np.arange(8)+offset,means,yerr=errors,fmt='o',markersize=7,capsize=5,
                        color=color,label='Rust' if engine=='rust' else 'Java TAG',elinewidth=1.5)
        ax.set(xticks=np.arange(8),xticklabels=labels,xlabel='Game variant',ylabel='Mean complete-game time (seconds)',yscale=scale)
        if scale=='linear':ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.18,which='major')
        handles,legend=ax.get_legend_handles_labels()
        fig.legend(handles,legend,loc='upper center',frameon=False,ncol=2)
        fig.tight_layout(rect=(0,0,1,.92))
        for extension in ('png','svg','pdf'):
            fig.savefig(folder/f'all-games-time-{scale}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for scale in ('linear','log'):
        fig,ax=plt.subplots(figsize=(14,5.8));variant_guides(ax)
        jitter=np.random.default_rng(BOOTSTRAP_SEED)
        for engine,color,offset in [('rust','#188477',-.2),('java','#7c5db3',.2)]:
            data=[np.asarray(times[(engine,*variant)]) for variant in variants]
            positions=np.arange(8)+offset
            # Estimate density in the displayed coordinate: seconds on the
            # linear plot, log10(seconds) on the logarithmic plot.
            density_data=[np.log10(values) if scale=='log' else values for values in data]
            violins=ax.violinplot(density_data,positions=positions,widths=.35,
                                 showextrema=False,showmeans=False,showmedians=False,points=150)
            for body in violins['bodies']:
                if scale=='log':
                    for path in body.get_paths():path.vertices[:,1]=10**path.vertices[:,1]
                body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.2)
            ax.boxplot(data,positions=positions,widths=.075,patch_artist=True,
                manage_ticks=False,showfliers=False,whis=1.5,
                boxprops={'facecolor':'white','edgecolor':color,'linewidth':1.2},
                medianprops={'color':color,'linewidth':2},whiskerprops={'color':color},
                capprops={'color':color})
            for index,(position,values) in enumerate(zip(positions,data)):
                ax.scatter(position+jitter.uniform(-.13,.13,len(values)),values,s=7,
                           color=color,alpha=.23,linewidths=0,zorder=2,
                           label=('Rust' if engine=='rust' else 'Java TAG') if index==0 else None)
        ax.set(xticks=np.arange(8),xticklabels=labels,xlabel='Game variant',
               ylabel='Complete-game time (seconds)',yscale=scale)
        if scale=='linear':ax.set_ylim(bottom=0)
        else:
            observed=[value for values in times.values() for value in values]
            ax.set_ylim(min(observed)*.8,max(observed)*1.25)
        ax.grid(axis='y',alpha=.18)
        handles,legend=ax.get_legend_handles_labels()
        key=fig.legend(handles,legend,loc='upper center',frameon=False,ncol=2,markerscale=3)
        for handle in key.legend_handles:handle.set_alpha(1)
        fig.tight_layout(rect=(0,0,1,.92))
        for extension in ('png','svg','pdf'):
            fig.savefig(folder/f'all-games-distribution-{scale}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    text='''# Ratios by player count

Every ratio is **Java / Rust**: 1 means equal, and values above 1 mean Rust
uses less time or memory. Speed uses pooled mean complete-game time, including
instrumentation; differences in game length, heuristics and search work remain.
Memory uses the ratio of median sampled process peaks, separately for private
committed memory and resident working set. These include runtime and search
memory, and are not sizes of individual states. Baselines are not subtracted.
Lines connect the measured player counts; they are not fitted predictions.

There are 250 included games per setting except Java Love Letter with three
players, which has 248 after the two approved exclusions. No additional games
were excluded. [Underlying ratios and values](ratios.csv) retain full precision.
See the [study report](../../README.md) for settings and limitations.

Error bars are **95% percentile bootstrap confidence intervals**, using 20,000
resamples and fixed seed 20261030. For time, resample complete included games
independently within each engine/profile and recompute the pooled mean. For
ratios, divide independently resampled Java estimates by Rust estimates; matching
seeds do not imply paired trajectories. These time intervals describe variation
among games conditional on the measured processes; they do not include all
between-process or between-machine timing uncertainty. They are intervals for
the estimated mean, not the range in which 95% of individual games finish.

For memory, resample whole measured processes and recompute each engine's median
peak before taking the ratio. There are only three measured processes per
engine/profile, so these intervals have limited resolution and cannot establish
precise memory uncertainty. They cover variation in sampled peaks, not memory
spikes missed by the 50 ms sampler. No new benchmark measurements were taken.
'''
    for game,label in [('cantstop',"Can't Stop"),('loveletter','Love Letter')]:
        text+=f'\n## {label}\n'
        for kind in ('speed','memory'):
            name=f'{game}-{kind}'
            text+=f'\n![{label}: {kind} ratio by player count]({name}.svg)\n\n[PNG]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg)\n'
    text+='\n## Complete-game time across all variants\n\nPoints show the mean for each engine, with 95% confidence intervals.\nThe second version uses a logarithmic y-axis so the shorter games remain visible.\n[Means, counts and confidence limits](times.csv) give the numerical values.\n'
    for scale in ('linear','log'):
        name=f'all-games-time-{scale}'
        text+=f'\n![Complete-game time by variant, {scale} axis]({name}.svg)\n\n[PNG]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg)\n'
    text+='''
## Distribution of individual game times

Every included game appears as a small dot: 3,998 observations across all
engine/variant combinations. Horizontal jitter separates overlapping dots and
has no numerical meaning. Each tinted violin shows a smoothed distribution;
width indicates relative density within that engine/variant, not sample count.
The small box spans the middle 50% of game times, with a line at the median.
Whiskers extend to the most extreme observations within 1.5 box heights of the
box. All observations beyond the whiskers remain visible as dots.

The logarithmic version makes the shorter games visible. Its violin density is
estimated in log-time; its boxes and whiskers retain their definitions in seconds.
These distributions show variation between individual games; the confidence bars
in the preceding plots describe uncertainty in the estimated means instead.
Vertical guides and alternating bands associate both engines with each variant.
[Individual times with game IDs and seeds](individual-game-times.csv) reproduce
the plotted observations. The same two approved exclusions remain in effect.
'''
    for scale in ('linear','log'):
        name=f'all-games-distribution-{scale}'
        text+=f'\n![Individual game time distributions, {scale} axis]({name}.svg)\n\n[PNG]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg)\n'
    (folder/'README.md').write_text(text,encoding='utf-8')
    report=root/'README.md';text=report.read_text(encoding='utf-8')
    link='\n## Ratios by player count\n\n[Speed and memory plots for Can\'t Stop and Love Letter](plots/ratios/README.md)\nshow Java/Rust ratios against the number of players, with equality at 1.\n'
    if '## Ratios by player count' not in text:text=text.replace('\n## Data and limits',link+'\n## Data and limits')
    report.write_text(text,encoding='utf-8')
    sources=root/'report-source-hashes.json';hashes=json.loads(sources.read_text())
    for source in (Path(__file__).resolve(),Path(__file__).with_name('summarize.py').resolve()):
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    sources.write_text(json.dumps(hashes,indent=2)+'\n')
    hashes={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'checksums.json'}
    (root/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(f'Exported ratio, mean-time and individual-game distribution plots to {folder}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    plot(parser.parse_args().directory)
