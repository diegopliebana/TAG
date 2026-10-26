# Browser history navigation measurement

## Reading the measurements

A workload is the specific operation being measured, such as copying a state or
replaying a complete game. A harness is the program that repeats and times it.
The median is the middle measured value after sorting; it describes these samples,
not a guaranteed time for a future machine. Cache capacity is how many rebuilt
states are kept to avoid repeated work. A checkpoint interval says how many moves
separate eligible cached positions. Read the units and included/excluded work
before comparing numbers. Raw samples, commands and historical results below are
unchanged. See [the glossary](../../reference/framework/glossary.md) for engine terms.


Recorded 14-10-2026 13:04 (Europe/Paris). The user-run Chrome harness includes
a small navigation measurement; [navigation.json](navigation.json) preserves its
raw output. This is workload-specific evidence, not a comparative benchmark or
performance guarantee.

## Workload and method

Chrome/154.0.8037.99 on the user's Windows environment, using the standard headless
harness. Hardware and the default headless viewport were not recorded.
The release Wasm engine runs in the real module worker.

[review.js](../../../web/tests/review.js) creates a 32x32 Tic-Tac-Toe board with two
human seats, reduced motion and mute enabled. It applies 31 alternating moves:
column floor(index/2), row index%2, with zero-based index 0..30. This exercises
normal board/list rendering before measuring eight jumps to recording positions
0, 10, 20, 31, 10, 0, 31, 20. There is one branch and 31 accepted records.

Each sample starts immediately before clicking the corresponding retained move
button and ends when the displayed cursor matches and navigation unlocks Reset.
It includes worker request/reconstruction/JSON transport, main-thread DOM updates
and the test's polling (up to 5 ms increments). It does not require a subsequent
paint acknowledgement or isolate reconstruction, serialization or rendering costs.
No separately controlled benchmark warmup or hardware/load isolation was performed.

## Observed results

| User run | Samples | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| First full graphical run | 8 | 9.6 ms | 13.6 ms | 17.9 ms |
| Final run, including both-seat/inherited-prefix/rapid-switch cases | 8 | 12.3 ms | 16.4 ms | 32.8 ms |

These two runs use the same navigation workload. They establish that the tested
configured board/list can be browsed within these observed settlement times;
they do not establish behavior at every permitted board/history/branch limit.
The existing bounded reconstruction cache and incremental live move metadata remain
the selected design. Pagination/virtualisation or additional cache changes require
larger measured workloads or specific UX evidence before implementation.

Repeat with `node tools/browser/test-browser.mjs` in normal PowerShell; its review
section prints the raw sample array. [web/README](../../../web/README.md) owns setup,
build and launch commands. These real-browser results were collected by the repository owner; browser launch
restrictions can prevent reproducing them in another environment.
