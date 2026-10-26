# Python adapter

The compiled adapter lets Python configure Rust games, simulations and search.
PyO3/maturin build the extension; NumPy carries independently owned observation
arrays. Optional PyTorch modules evaluate batches and train/export models.

## Build and check

Follow [setup](../../docs/guides/setup.md) for required and optional dependencies.
Run from the repository root in the configured environment:

```powershell
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --offline --manifest-path adapters/python/Cargo.toml
.\.venv\Scripts\python.exe -m unittest discover -s adapters/python/tests -v
```

The adapter has its own workspace and lockfile. Rebuild after changing Rust;
Python source changes are visible through the editable installation. Optional
Gymnasium/PyTorch tests require their corresponding dependencies. Rust adapter
checks are separate from the root workspace:

```powershell
cargo test --offline --manifest-path adapters/python/Cargo.toml
cargo clippy --offline --manifest-path adapters/python/Cargo.toml --all-targets -- -D warnings
```

## Workflows and source ownership

| Need | Main reference | Implementation |
|---|---|---|
| Create, observe, copy/reset and apply actions | [Game API](../../docs/reference/python/game-api.md) | Rust `src/engine.rs`, Python `engine/` |
| Run native/externally evaluated agents | [Rust-owned execution](../../docs/reference/python/rust-owned-execution.md), [external OSLA](../../docs/reference/python/external-osla.md) | Rust `execution.rs`, `evaluations.rs`; Python `search/` |
| Batch independent game evaluations | [Grouped evaluation](../../docs/reference/python/grouped-evaluation.md) | Rust grouping/array modules; Python `search/` |
| Learner-step environments | [Gym](../../docs/reference/python/gym-episodes.md), [bulk episodes](../../docs/reference/python/batched-episodes.md) | Rust episode modules; Python `episodes/` |
| Tabular or teacher/actor-critic training | [Learning](../../docs/reference/learning/README.md) | Python `learning/` |
| Native/neural MCTS and independent-tree batches | [Neural MCTS](../../docs/reference/learning/neural-mcts.md) | Rust `src/mcts.rs` and core search; Python `search/` |
| Complete-game self-play/checkpoints/evaluation | [Self-play](../../docs/reference/learning/mcts-selfplay.md), [evaluation](../../docs/reference/learning/model-evaluation.md) | Python `learning/selfplay/`, Rust native collection/opponent execution |

Paths in the implementation column are relative to this adapter. The
[package map](python/tag_rust/README.md) describes topic imports and compatibility
aliases. [Capabilities](../../docs/reference/framework/overview.md) lists which
games support each workflow. Basic registration does not enable every controller
or learner environment.

The [game-operation guide](../../docs/guides/running/game-operations-from-python.md)
and [MCTS guide](../../docs/guides/agents/rust-mcts-from-python.md) provide runnable
examples. [Build details](../../docs/reference/python/setup.md) describes wheels,
interpreter selection and platform limits. The [OSLA benchmarks](../../docs/benchmarks/osla-python/README.md)
measure complete execution routes without assuming zero-copy transfer.
