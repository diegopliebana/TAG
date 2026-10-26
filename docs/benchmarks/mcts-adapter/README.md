# Native browser-adapter MCTS sample

These recorded CPU samples describe default initial placement-game positions.
The [raw report](samples.json) retains model identities, selected actions,
configuration and runtime context. They are not a Java/Rust comparison.

```powershell
cargo run --offline --release -p tag-browser --example mcts_workload -- MODEL...
```

The model determines the game. Each decision uses 256 iterations and zero rollout
length. Twenty preparations per profile are discarded without commitment,
retaining the same seed/position; each model also receives 100 prediction requests.
File loading is outside the timed sections. MODEL paths must name compatible
exported models; the original locally trained artifacts are not bundled here.

| Game / profile | Mean prediction request (ms) | Mean decision preparation (ms) |
|---|---:|---:|
| Tic-Tac-Toe / native UCB1 | 0.0153 | 1.2635 |
| Tic-Tac-Toe / neural PUCT | 0.0143 | 4.8985 |
| Connect 4 / native UCB1 | 0.0173 | 2.3595 |
| Connect 4 / neural PUCT | 0.0163 | 6.1637 |

These are single Windows CPU samples (Intel64 Family 6 Model 154 Stepping 3),
with other validation potentially running. Prediction timing includes adapter
validation and JSON conversion; decision timing includes polling/JSON and local
search/inference. There is no browser worker transport or graphical rendering in
this native-adapter workload. Native-profile rows still measure the same neural
prediction requests separately. Different tree policies/evaluators make this a
cost description, not an equal-algorithm speed comparison or strength claim.
