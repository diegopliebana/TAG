# Contributing

Start with [setup](docs/guides/setup.md) and the guide for the workflow you want to
change. For extensions, use [new games](docs/guides/games/new-game.md) or
[new agents](docs/guides/agents/new-agent.md). Explain the behavior you intend to
change and provide a small reproducible example for bugs.

Keep rules independent of Python, graphics and filesystem operations. Preserve
legal actions, multiplayer outcomes, explicit randomness and player-visible
information boundaries. Search must not receive actual hidden information.

Run the checks relevant to your change from the repository root:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features --all-targets
cargo test --workspace --all-features --doc
python tools/docs/check_docs.py
```

Python changes also need the compiled adapter and
`python -m unittest discover -s adapters/python/tests -v`. Browser changes need
fresh [Wasm bindings](docs/guides/running/browser.md) and
`node --test web/tests/*.test.js`. State unavailable checks explicitly.

Use issues for reproducible bugs and feature proposals, and pull requests for
reviewable changes with their validation results. Changes accepted in the public
repository are integrated into the canonical source before the next source export;
the next update must preserve accepted contributions. Do not commit generated
builds, local environments, private data or credentials.

Contributions are provided under the [MIT licence](LICENSE). Preserve applicable
upstream notices and identify the source and licence of added assets.
