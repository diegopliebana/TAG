# Build and check the Python connection

For operating instructions, use the [short manual](../../guides/setup.md). This page contains
the detailed reference.

[Learning guide entry point](../learning/overview.md) · [Use the connection](game-api.md)

The Python package contains a compiled extension, a library that Python loads
when you import `tag_rust`. PyO3 connects Python calls to Rust functions. Maturin
builds the extension and places it in the local Python environment. NumPy supplies
the array type used for observations.

The verified first environment is Windows x64 with regular CPython 3.14.8.
Versions are pinned: Python NumPy 2.5.4, maturin 1.15.0, Rust PyO3 0.29.3 and
rust-numpy 0.29.0. The adapter has a separate Cargo lock file. Other Python
versions, free-threaded Python builds and other operating systems require their
own checks. Optional [PyTorch inference](../learning/neural-inference.md) has separate setup
and validation instructions.

## Prepare a local environment

Run from normal PowerShell at the repository root. The virtual environment is a
project-local Python installation with its own packages:

```powershell
# Run from the repository root.
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "numpy==2.5.4" "maturin==1.15.0"
cargo fetch --manifest-path adapters/python/Cargo.toml
```

These installation/fetch commands need package access. Once completed, builds and tests use
the installed packages and cached Rust dependencies offline. If you already have
this environment, reuse it.

## Build the extension

```powershell
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --offline --manifest-path adapters/python/Cargo.toml
```

`PYO3_PYTHON` selects the interpreter used to compile/link the extension.
`VIRTUAL_ENV` tells maturin where to install it. `develop` installs an editable
package: edits to the Python source are available directly, while edits to Rust
require repeating this build. `--release` enables Rust optimizations.

To produce a distributable wheel, use:

```powershell
.\.venv\Scripts\python.exe -m maturin build --release --offline --manifest-path adapters/python/Cargo.toml --out target/wheels
```

A wheel packages the Python modules and compiled extension for the selected
interpreter/platform. Other targets need separately built and tested packages.

The editable extension is installed only in `.venv`. The native engine and browser
workspace keep their existing commands and require no Python setup. The adapter
is built separately with `--manifest-path adapters/python/Cargo.toml`.

## Run games and tests

```powershell
.\.venv\Scripts\python.exe adapters/python/examples/play.py
.\.venv\Scripts\python.exe adapters/python/examples/run_agents.py
.\.venv\Scripts\python.exe adapters/python/examples/run_grouped.py
.\.venv\Scripts\python.exe adapters/python/examples/play_osla_python.py
.\.venv\Scripts\python.exe -m unittest discover -s adapters/python/tests -v
cargo clippy --offline --manifest-path adapters/python/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path adapters/python/Cargo.toml -- --check
```

`play.py` runs three direct-input matches. `run_agents.py` configures agents in
Python and runs the complete example matches inside Rust. Its `--evaluation python`
option demonstrates returning batches for external evaluation. `play_osla_python.py`
shows the lower-level manual batch calls, using Rust scoring by default.
`run_grouped.py` runs two games of each type and combines their external evaluations.
The tests run a native Rust reference program
and compare Python observations with its output at every recorded position.
They also check array types/shapes, independent copies, errors, stale requests,
reset, close, full columns and a 4,096-cell board. They compare candidate chunks,
selected moves and exact RNG progress against native references, then compare
individual/grouped execution, native/Python scoring, retries and cancellation.

The extension imports NumPy when loaded. A missing NumPy installation therefore
produces an import error before game execution. If the extension was built for a
different Python interpreter, rebuild using the environment above. A failed build
should be investigated from its actual error output; compiler installation is a
separate manual setup step if required.

## Why the adapter is separate

The root Cargo workspace contains the engine and browser adapter. The Python
adapter has its own workspace because its builds link against Python and resolve
additional dependencies. This lets the ordinary native and WebAssembly checks
continue independently. The Python package imports the same Rust engine through
a path dependency; game rules stay in their existing modules.

See the [adapter file map](../../../adapters/python/README.md) for implementation details. Dependency behavior is documented by
[PyO3](https://pyo3.rs/v0.29.0/module.html),
[rust-numpy](https://docs.rs/numpy/0.29.0/numpy/array/struct.PyArray.html) and
[maturin](https://www.maturin.rs/project_layout.html).

## Use the Python debugger in RustRover

Enable the IDE's Python support and select the existing interpreter
`.venv\Scripts\python.exe` in its Python interpreter/SDK
settings. Create a Python run/debug configuration for one of the example files,
using that interpreter and the repository root as the working directory. IDE
menu locations depend on the installed version/plugins; a configuration showing
`<No interpreter>` needs a Python SDK configured before it can run.

Put breakpoints in the example or evaluator and start Debug. Python breakpoints
inspect Python control flow and returned NumPy arrays; stepping into a compiled
Rust call requires native/mixed debugging support beyond this verified Python
workflow. Rebuild with maturin after Rust edits before debugging the updated code.

## Run the learner examples

After optional Gymnasium setup in the [episode guide](gym-episodes.md), follow
[tabular learning](../learning/tabular-learning.md) to collect episodes, train a table and
reload its learned values. The command uses the same compiled extension and
existing virtual environment. The [tabular-learning reference](../learning/tabular-learning.md) explains its
versioned table format and evaluation semantics.
