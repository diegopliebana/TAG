"""Write current speed tables and captions directly from the accepted summaries."""
import argparse
import json
from pathlib import Path
import statistics
from run import PROFILES

NAMES={"tictactoe":"Tic-Tac-Toe","connect4":"Connect 4","cantstop":"Can't Stop","loveletter":"Love Letter"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    read=lambda name:json.loads((root/name).read_text())
    primary={(r['engine'],r['game'],r['players'],r['operation']):r for r in read('summary.json')}
    games={(r['engine'],r['game'],r['players'],r['operation']):r for r in read('game-summary.json')}
    components={(r['engine'],r['game'],r['players'],r['operation'],r['prefix']):r for r in read('component-summary.json')}
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    def game_rows(profiles,mode='game'):
        for g,n in profiles:
            r,j=(games[e,g,n,mode] for e in ('rust','java'))
            yield [f"{j['ns_per_game']/r['ns_per_game']:.2f}",NAMES[g],n,f"{r['ns_per_game']/1000:.3f}",f"{j['ns_per_game']/1000:.3f}",f"{r['actions_per_game']:.3f}",f"{j['actions_per_game']:.3f}",f"{j['ns_per_action']/r['ns_per_action']:.2f}"]
    text='''# Updated four-game speed and memory measurements

Fresh measurements on 26 October 2026 use the current Rust implementation and
the warmed, unmodified public Java reference. They include the complete benchmark
suite and additional game-length/per-turn/component diagnostics.

The first column of timing comparisons is **Java time / Rust time**: above 1 means
Rust was faster for the measured operation; below 1 means Rust was slower. Rule,
knowledge and search differences remain part of these implementation comparisons.
[Methodology](methodology.md) defines the included work and limits. Tables report
the median of three independent-process medians, each based on five timed batches.

## Complete Random games with observation handling in both engines

Both implementations generate actions from state, construct the acting player's
observation/player copy, call the actual Random controller, then apply the move.
Blocks cover the same 32 seeded controller fixtures within each engine. These
are complete games, including setup; stochastic histories differ between engines.

'''
    text+=table(['Java/Rust game time','Game','Players','Rust µs/game','Java µs/game','Rust actions/game','Java actions/game','Java/Rust time per action'],game_rows(PROFILES))
    text+='''

![Complete-game latency, actual action/turn lengths and time per accepted action.](plots/workflows/random-game-work.svg)

A game taking longer does not by itself imply a slower forward model: it may
execute more actions. The last column normalizes by actual accepted actions.
Logical turns include terminal turns; Can't Stop has several actions per turn.
Full per-turn lengths and timings are in [the normalized CSV](game-summary.csv).

## Love Letter with equal winning-token targets

Defaults use six tokens in Rust and seven in two-player Java. This additional
profile configures Java's public winning-token parameter to six; both sides still
construct observations. Other rule/knowledge/RNG differences are retained.

'''
    text+=table(['Java/Rust game time','Game','Players','Rust µs/game','Java µs/game','Rust actions/game','Java actions/game','Java/Rust time per action'],game_rows([('loveletter',2)],'game_aligned'))
    text+='''

![Two-player Love Letter under default and aligned token thresholds.](plots/workflows/loveletter-token-target.svg)

## Rules and prepared-state APIs

Two-player profiles below use each implementation's prepared opening state.
Stochastic positions are not matched; timings are microseconds per call.
Copy-plus-apply includes an input clone; isolated checked transitions are measured
separately in the component probes.

'''
    rows=[]
    for g in NAMES:
        for op,label in [('init','Initialize'),('actions','Actions'),('copy','Copy'),('observe','Observation/player copy'),('transition','Copy + apply')]:
            r,j=(primary[e,g,2,op]['median_ns'] for e in ('rust','java'))
            rows.append([f'{j/r:.2f}',NAMES[g],label,f'{r/1000:.4f}',f'{j/1000:.4f}'])
    text+=table(['Java/Rust time','Game','Operation','Rust µs','Java µs'],rows)
    text+='''

![Prepared-state APIs for all game/player profiles.](plots/core-operations.svg)

## Agent decisions

These are prepared-root decision costs, not playing-strength comparisons or
complete matches. OSLA and MCTS perform different internal work across the two
implementations. MCTS uses 32 iterations and rollout limit four. First Available
uses Rust's agent and Java's first-list-element baseline; such tiny measurements
can be dominated by harness/dispatch costs. Root preparation and controller setup
are included or excluded as specified in the methodology and workload manifest.

'''
    rows=[]
    for g in NAMES:
        for op,label in [('first','First Available / list baseline'),('random','Random'),('osla','OSLA'),('mcts','MCTS')]:
            r,j=(primary[e,g,2,op]['median_ns'] for e in ('rust','java'))
            rows.append([f'{j/r:.2f}',NAMES[g],label,f'{r/1000:.4f}',f'{j/1000:.4f}'])
    text+=table(['Java/Rust time','Game','Agent/operation','Rust µs','Java µs'],rows)
    text+='''

![Decision costs for all eight game/player profiles.](plots/agent-decisions.svg)

## Matched placement-game replay corpus

This is the strongest equal-work comparison: both engines replay the same 32
validated histories per game and check boards, outcomes and available choices.
Replay values are per complete game; other operations are per call.

'''
    matched={(r['engine'],r['game'],r['operation']):r for r in read('matched-summary.json')}
    rows=[]
    for game,label in [('ttt','Tic-Tac-Toe'),('c4','Connect 4')]:
        for op,name in [('init','Initialize'),('actions','Actions'),('copy','Copy'),('transition','Apply'),('replay','Complete replay')]:
            r,j=(matched[e,game,op]['median_ns'] for e in ('rust','java'))
            rows.append([f'{j/r:.2f}',label,name,f'{r/1000:.4f}',f'{j/1000:.4f}'])
    text+=table(['Java/Rust time','Game','Operation','Rust µs','Java µs'],rows)
    text+='''

![Matched recorded-board corpus and process-pair timing ratios.](plots/matched-board-corpus.svg)

Java full copies include component maps, timers, history bookkeeping and branched
RNGs; Rust uses compact game-specific values. Ratios include those architectural
choices, rather than isolating the programming language.

## Memory, adapters and complete workflows

All earlier memory/adapter categories were recollected. Process committed memory,
resident working set, post-GC Java heap and requested allocation bytes describe
different quantities; initialized runtime/scaffolding must be separated from the
additional retained game states. JVM baseline memory is not per-state resident RAM.

![Initialized and loaded process memory with 2048 retained states.](plots/process-memory-2048-states.svg)

![Allocation volume of full-state copies.](plots/state-copy-allocations.svg)

The [diagnostic details](details.md) include per-turn normalization, direct-choice
controls, state/history copying, aligned token targets, Python/neural batches,
feature/tree memory, histories/branches, learner collection and CLI results.
[All figures](plots/README.md) are reusable SVG/PDF/PNG exports without visible titles.
The [data/reproduction index](data.md), [environment](environment.json),
[source hashes](source-hashes.json) and [class hashes](java-class-hashes.json)
identify the measured working code and reference. No global language multiplier
is inferred from these fixtures.
'''
    allocations={(r['engine'],r['game'],r['players'],r['operation']):r for r in read('allocation-summary.json')}
    text+='\n## Requested allocation volume\n\nThese are bytes requested per full-state copy, measured separately from latency.\nRust values exclude inline state storage; Java values count current-thread allocations.\nThey are not retained or resident bytes per state.\n\n'
    rows=[]
    for g,n in PROFILES:
        r,j=(allocations[e,g,n,'copy']['bytes_per_call'] for e in ('rust','java'))
        rows.append([f'{j/r:.2f}' if r else '—',NAMES[g],n,f'{r:.1f}',f'{j:.1f}'])
    text+=table(['Java/Rust requested bytes','Game','Players','Rust bytes/copy','Java bytes/copy'],rows)+'\n'
    (root/'README.md').write_text(text,encoding='utf-8')
    details='''# Workflow and component diagnostics

See [measurement scope](methodology.md) before comparing these costs. Full-game
elapsed time includes setup and final state disposal/GC; normalization uses actual
measured work, not assumed or warmup-inclusive lengths. Logical turns include
terminal plays, with one bank/bust turn spanning multiple actions in Can't Stop.

## Game lengths and per-turn costs

'''
    rows=[]
    for g,n in PROFILES:
        r,j=(games[e,g,n,'game'] for e in ('rust','java'))
        rounds=[f"{v['rounds_per_game']:.3f}" if g=='loveletter' else '—' for v in (r,j)]
        rows.append([f"{j['ns_per_turn']/r['ns_per_turn']:.2f}",NAMES[g],n,f"{r['turns_per_game']:.3f}",f"{j['turns_per_game']:.3f}",f"{r['ns_per_turn']/1000:.3f}",f"{j['ns_per_turn']/1000:.3f}",*rounds])
    details+=table(['Java/Rust per turn','Game','Players','Rust turns/game','Java turns/game','Rust µs/turn','Java µs/turn','Rust rounds/game','Java rounds/game'],rows)
    details+='''

Only Love Letter has the card-game rounds counted here. Raw board framework
round counters remain available in the data but are not presented as game rounds.

## Observation construction versus direct choice

Both paths use the same within-engine seeded fixture cycle and forced-choice RNG
convention. The diagnostic path omits observation construction and the actual
agent wrapper; it is not the ordinary uniform agent interface. Values are µs/game.

'''
    rows=[]
    for g,n in PROFILES:
        r,j=(games[e,g,n,'game_direct'] for e in ('rust','java'))
        rows.append([f"{j['ns_per_game']/r['ns_per_game']:.2f}",NAMES[g],n,f"{r['ns_per_game']/1000:.3f}",f"{j['ns_per_game']/1000:.3f}",f"{games['rust',g,n,'game']['ns_per_game']/r['ns_per_game']:.2f}",f"{games['java',g,n,'game']['ns_per_game']/j['ns_per_game']:.2f}"])
    details+=table(['Java/Rust direct time','Game','Players','Rust direct µs','Java direct µs','Rust full/direct','Java full/direct'],rows)
    details+='''

![Per-action elapsed time with observations and with direct selection.](plots/workflows/observation-workflows.svg)

Full/direct ratios compare independent process groups collected at different
times. Rust's Can't Stop ratios below one illustrate measurement/code-path
variation; they do not establish that constructing a view has negative cost.
Likewise, the Rust six-token profile is repeated independently from its identical
default configuration. Use process ranges rather than treating small differences
between these runs as causal effects.

## Prepared-position components

Fixtures advance up to 0, 8 or 128 additional actions and stop before terminal
play. Actual lengths are stored per engine/fork in the component JSON/CSV.
Below are the late/capped positions. `validate` re-enumerates legal actions and
checks membership; `apply` includes all checked rule effects, with input copies
prepared outside the timer. These are position-specific probes, not additive
components that sum exactly to mean whole-game latency.

'''
    rows=[]
    for g,n in PROFILES:
        for op in ('copy','observe','validate','apply'):
            r,j=(components[e,g,n,op,128]['median_ns'] for e in ('rust','java'))
            rows.append([f'{j/r:.2f}',NAMES[g],n,op,f'{r/1000:.4f}',f'{j/1000:.4f}'])
    details+=table(['Java/Rust time','Game','Players','Operation','Rust µs','Java µs'],rows)
    details+='''

![Opening and later-position component probes for the four two-player games.](plots/workflows/prepared-components.svg)

At the capped two-player Love Letter position, Rust's checked transition is
slower than Java's. These states and their automatic consequences are not matched;
this result is retained rather than generalizing the whole-game advantage to
every transition. The CSV includes the actual preparation lengths.

Love Letter's history probe copies retained current/previous-round public/private
event buffers in Rust and action/text history lists in Java. Their contents and
deep/shallow-copy semantics differ. Current Rust state has no match-long proof;
round resets can make a later position smaller than an earlier one.

![Love Letter state and history-buffer copy costs for 2-4 players.](plots/workflows/loveletter-history-copy.svg)

Opening `apply_control` measures the clock/loop floor over 1000 prepared blocks.
Its values are retained in component summaries and are never subtracted.
The [component CSV](component-summary.csv) includes all prefixes, save/load costs,
actual prepared lengths and process ranges; [JSON](component-summary.json) retains
per-process fixture metadata. Separate allocation probes use the same modes.

## Python, local models, episodes and memory

CPU fixtures use deterministic untrained networks with one 16-unit hidden layer.
They measure execution cost, not agent strength. Neural MCTS uses full player-value
vectors; scalar discounted OSLA has a separate contract. Independent-tree/model
batches and actual seat/root mixtures remain distinct workload dimensions.

![Independent-tree and prediction batches.](plots/neural-batch-and-tree-scaling.svg)

![Search structure and temporary/retained memory.](plots/search-memory-lifetimes.svg)

![Completed learner episodes per second.](plots/episode-collection.svg)

Learner-step collection here covers the registered Tic-Tac-Toe/Connect 4 workflows;
neural search/model probes cover all eight game/player profiles. Native browser
adapter timings cover host Rust/JSON and local inference, including Q-table,
OSLA/MCTS policies, saved history, branches and continuation. They do not include
Web Worker transport or rendering. Search/memory counters describe algorithmic
work and must not be substituted for physical allocation measurements.

![Process memory versus retained-state count.](plots/retained-state-scaling.svg)

![Complete match/tournament execution and reporting.](plots/runner-reporting-cost.svg)

![Saved-history reconstruction cost.](plots/history-scaling.svg)

![Fork and fork-plus-continuation cost.](plots/branch-continuation.svg)

All raw observations, optional-feature footprints and initialized baselines remain
in the [data index](data.md). Requested bytes, Java heap, committed process memory
and resident memory have the separate meanings defined in the methodology.
'''
    details+='\n## Retained knowledge and persistence costs\n\nTwo-player Rust probes below use opening and later/capped positions.\nAll values are microseconds per call; history copying measures retained knowledge\ncontainers separately, while full copy includes them. These medians overlap in\nincluded work and must not be added together. Save/load is outside the search\ntransition path.\n\n'
    rows=[]
    for op in ('copy','history_copy','observe','validate','apply','save','load'):
        values=[components['rust','loveletter',2,op,p]['median_ns']/1000 for p in (0,8,128)]
        rows.append([op,*[f'{v:.4f}' for v in values]])
    details+=table(['Operation','Opening','Up to 8 actions','Up to 128 actions'],rows)+'\n'
    details+='\nThe complete-snapshot save format preserves current facts and exact RNG continuation.\nIts validation/serialization can cost more than the earlier opening-state proof\nrecipe. It avoids replay proportional to the whole earlier match; it does not\npromise cheaper saves at every position. Retained knowledge still varies by round.\n'
    (root/'details.md').write_text(details,encoding='utf-8')
    originals=['matched-board-corpus','core-operations','agent-decisions','process-memory-2048-states','state-copy-allocations','retained-state-scaling','neural-batch-and-tree-scaling','search-memory-lifetimes','episode-collection','runner-reporting-cost','history-scaling','branch-continuation']
    diagnostics=['random-game-work','observation-workflows','loveletter-token-target','prepared-components','loveletter-history-copy']
    index='# Standalone benchmark figures\n\nExports have labelled axes/legends and no visible titles. See the report captions\nand methodology for included work, repetitions and comparison limits.\n\n'
    index+=table(['Figure','SVG','PDF','PNG'],[[name,*[f'[{ext.upper()}]({name}.{ext})' for ext in ('svg','pdf','png')]] for name in originals]+[[name,*[f'[{ext.upper()}](workflows/{name}.{ext})' for ext in ('svg','pdf','png')]] for name in diagnostics])+'\n'
    (root/'plots/README.md').write_text(index,encoding='utf-8')
    print('Wrote current report tables, diagnostic detail and figure index from accepted summaries.')


if __name__=='__main__':main()
