"""Render saved timing samples as Markdown tables; this module runs no games."""
import statistics

def write_tables(report, destination):
    """Generate the result tables from the saved samples so reruns stay reviewable."""
    mean = statistics.mean
    lines = [
        '# OSLA benchmark results',
        '',
        f"Measured {report['timestamp']}.",
        '',
        f"{report['platform']}; {report['cpu']}; {report['logical_cpus']} logical CPUs.",
        f"Python {report['python']}, NumPy {report['numpy']}, {report['rust']}.",
        '',
        'See [the method and interpretation](README.md) and [raw samples](results.json).',
        '',
        '## Full-game averages',
        '',
        'Batch limit 1,024: all candidates for one move fit in one batch. Times include setup.',
        '',
        '| Board | Rust-only original ms | Rust-only batched ms | Python starts, Rust runs/scores ms | Python loop, Rust scoring ms | Python loop, Python scoring ms | Manual Rust scoring/owned execution |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]

    def name(case):
        parameters = case['parameters']
        return f"Tic-Tac-Toe {parameters['grid_size']}x{parameters['grid_size']}" if case['game'] == 'tictactoe' else f"Connect 4 {parameters['width']}x{parameters['height']}, target {parameters['winning_length']}"
    for case in report['cases']:
        original_seconds = mean(case['native_game_seconds'])
        batched_seconds = mean(case['rust_batch']['1024']['game_seconds'])
        python_seconds = mean(case['python']['1024']['game_seconds'])
        rust_call_seconds = mean(case['python_calls_rust']['1024']['game_seconds'])
        owned_seconds = mean(case['python_starts_rust_runs']['1024']['game_seconds'])
        lines.append(f'| {name(case)} | {original_seconds * 1000:.4f} | {batched_seconds * 1000:.4f} | {owned_seconds * 1000:.3f} | {rust_call_seconds * 1000:.3f} | {python_seconds * 1000:.3f} | {rust_call_seconds / owned_seconds:.1f}x |')
    lines += [
        '',
        '## Isolated heuristic scoring',
        '',
        'Microseconds per candidate. Rust states and candidate arrays are prepared before timing.',
        'The new Rust-owned execution route uses the Rust-only batched scorer; its complete-game cost is measured above.',
        '',
        '| Board | Rust-only original us | Rust-only batched us | Python calls Rust us | Python scores us | Python/Rust scoring-call ratio |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for case in report['cases']:
        original_seconds = mean(case['native_score_seconds'])
        batched_seconds = mean(case['rust_batch_score_seconds'])
        python_seconds = mean(case['python_score_seconds'])
        rust_call_seconds = mean(case['python_calls_rust_score_seconds'])
        lines.append(f'| {name(case)} | {original_seconds * 1000000.0:.3f} | {batched_seconds * 1000000.0:.3f} | {rust_call_seconds * 1000000.0:.2f} | {python_seconds * 1000000.0:.2f} | {python_seconds / rust_call_seconds:.1f}x |')
    lines += [
        '',
        '## Batch limits: complete-game means',
        '',
        '| Board | Route | Limit 1 ms | Limit 3 ms | Limit 1,024 ms |',
        '|---|---|---:|---:|---:|',
    ]
    for case in report['cases']:
        for route, label in [('rust_batch', 'Rust-only batched'), ('python_starts_rust_runs', 'Python starts, Rust runs/scores'), ('python_calls_rust', 'Python loop, Rust scoring'), ('python', 'Python loop, Python scoring')]:
            values = [mean(case[route][str(rows)]['game_seconds']) * 1000 for rows in (1, 3, 1024)]
            lines.append('| ' + name(case) + ' | ' + label + ' | ' + ' | '.join((f'{v:.3f}' for v in values)) + ' |')
    lines += [
        '',
        '## Sample spread',
        '',
        'Each range spans seven sample averages, each covering repeated sets of 16 games. Batch limit 1,024.',
        '',
        '| Board | Route | Median ms | Sample min-max ms |',
        '|---|---|---:|---:|',
    ]
    for case in report['cases']:
        for route, samples in [('Rust-only original', case['native_game_seconds']), ('Rust-only batched', case['rust_batch']['1024']['game_seconds']), ('Python starts, Rust runs/scores',case['python_starts_rust_runs']['1024']['game_seconds']), ('Python loop, Rust scoring', case['python_calls_rust']['1024']['game_seconds']), ('Python loop, Python scoring', case['python']['1024']['game_seconds'])]:
            lines.append(f'| {name(case)} | {route} | {statistics.median(samples) * 1000:.4f} | {min(samples) * 1000:.4f}-{max(samples) * 1000:.4f} |')
    lines += [
        '',
        '## Work per game',
        '',
        '| Board | Mean moves | Mean candidates | Manual Python requests: limits 1 / 3 / 1,024 | Owned advance calls/game |',
        '|---|---:|---:|---|---:|',
    ]
    for case in report['cases']:
        requests = ' / '.join((f"{case['python'][str(rows)]['profile_per_game']['requests']:.2f}" for rows in (1, 3, 1024)))
        lines.append(f"| {name(case)} | {case['mean_moves']:.2f} | {case['mean_candidates']:.2f} | {requests} | 1 |")
    lines += [
        '',
        '## Separate Python call breakdown',
        '',
        'Instrumented pass, mean milliseconds per game, batch limit 1,024. Includes timer overhead.',
        '',
        '| Board | Python calls | Prepare | Score | Finish | Commit | Scoring share of these calls |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for case in report['cases']:
        for route, label in [('python_calls_rust', 'Rust scorer'), ('python', 'Python scorer')]:
            profile = case[route]['1024']['profile_per_game']
            keys = ('prepare', 'evaluate', 'finish', 'commit')
            times = ' | '.join((f'{profile[k] * 1000:.3f}' for k in keys))
            share = profile['evaluate'] / sum((profile[k] for k in keys)) * 100
            lines.append(f'| {name(case)} | {label} | {times} | {share:.1f}% |')
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
