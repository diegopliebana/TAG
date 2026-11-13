# Setup

Run commands from the repository root. The Windows setup script prepares the
workflows you select. Install missing host tools in your own terminal using the
instructions below.

## Choose a setup

```powershell
.\tools\setup.ps1 -Profile native
.\tools\setup.ps1 -Profile browser
.\tools\setup.ps1 -Profile python
.\tools\setup.ps1 -Profile python -Gym
.\tools\setup.ps1 -Profile python -Neural
.\tools\setup.ps1 -Profile python -Gym -Neural
```

Run only the profiles you need. `native` builds and checks a game match. `browser`
installs the matching Rust Wasm target and binding generator, then builds the
browser files; run `.\serve-browser.ps1` afterward to play. `python` creates
`.venv`, installs pinned NumPy and maturin packages, builds the Rust adapter and
checks that it imports. `-Gym` adds Gymnasium for episode workflows; `-Neural`
adds CPU PyTorch for neural training and evaluation. These switches can be
combined. `-Profile all` prepares native, browser and Python; add `-Gym` and
`-Neural` to include those packages. The script reuses `.venv` and matching Rust
tools when rerun after a source change.

To inspect a selected setup without installing or building, add `-Check`, for
example `.\tools\setup.ps1 -Profile python -Gym -Check`. It checks command and
package availability and the Python adapter import; it does not inspect build
artifacts or run a game. A missing item produces an instruction and a nonzero exit
status. The script does not install the Windows C++ tools, Rust, Python or a
browser. Initial project package downloads need internet access.

## Which tools do I need?

These steps target Windows/PowerShell. Install only the host tools for your
chosen workflow. Reopen PowerShell after an installer changes PATH.

| Tool | Needed for | How to install |
|---|---|---|
| Microsoft C++ build tools + Windows SDK | Native Windows MSVC builds | Follow [Microsoft's installer instructions](https://learn.microsoft.com/en-us/windows/dev-environment/rust/setup). Install **Desktop development with C++**, including the MSVC compiler and Windows SDK. Do this before Rust. |
| Rust: rustup, rustc and Cargo | Engine, adapters and Rust tests | Download/run [rustup-init](https://rust-lang.org/tools/install/) for your architecture. Accept the stable MSVC toolchain and default PATH setup. Rustup installs Cargo and the compiler together. |
| Python 3.14 | Adapter/training; `python` on PATH also serves browser files | Install [Python Install Manager](https://www.python.org/downloads/windows/), then run `pymanager install 3.14`. Follow its PATH prompts so `py` and `python` work; [official Windows instructions](https://docs.python.org/3.14/using/windows.html) explain existing-launcher conflicts. Reuse an existing compatible installation. |
| Wasm target + wasm-bindgen-cli | Browser build | The `browser` profile installs these after Rust. For manual setup, run `rustup target add wasm32-unknown-unknown` and `cargo install wasm-bindgen-cli --version 0.2.129 --locked`. The binding generator must match the pinned crate version. |
| Chrome or Edge | Browser play and graphical tests | Keep an existing supported browser. Otherwise download Chrome through [Google's installer instructions](https://support.google.com/chrome/answer/95346), run the installer and follow its prompts. |
| Node.js | JavaScript/Wasm and graphical tests | On [Node's download page](https://nodejs.org/en/download), select version **24**, Windows and your architecture; run the `.msi` installer with PATH integration. The suite was validated with 24.21.0. |
| JDK 21 + Maven | Optional Java/Rust comparisons | Install the **JDK 21** Windows MSI from [Eclipse Temurin](https://adoptium.net/temurin/releases/?version=21), enabling PATH and JAVA_HOME in [Custom Setup](https://adoptium.net/installation/windows). Download/extract a [Maven binary ZIP](https://maven.apache.org/download.cgi) and add its `bin` directory to your user PATH, following [Maven installation](https://maven.apache.org/install.html). |
| Git | Obtaining/updating the repository | Run the [Git for Windows installer](https://git-scm.com/install/windows), enabling command-line PATH access. |
| RustRover or another editor | Optional editing/debugging | Download the Windows installer from [JetBrains](https://www.jetbrains.com/rust/download/), run its wizard, then open the repository root as a project. [Installation details](https://www.jetbrains.com/help/rust/installation-guide.html). RustRover bundles its own Java runtime; the separate JDK above is for benchmarks. |

To edit PATH, search Windows for **Edit environment variables for your account**,
select **Path → Edit → New**, and add the installed tool's `bin` directory. Use
your installation location; `JAVA_HOME` points to the JDK directory, without `bin`.

The Java tools are not needed for ordinary Rust tests or playing games. Their
configuration belongs in the [comparison-tool instructions](../../tools/java-comparison/README.md).
No npm packages, browser-driver installation or CUDA setup is needed for the
current browser tests and CPU training workflows. Cargo manages Rust dependencies
through the repository manifests/lock files; do not install each library manually.

## Understanding the commands

Cargo is Rust's build and dependency tool, comparable to using Maven or Gradle
for a Java project. It reads `Cargo.toml` to find programs, libraries and dependencies.
Run these commands from the repository root unless a guide says otherwise.

For example:

```powershell
cargo run --release --example random -- 7 8
```

| Part | Meaning and when to use it |
|---|---|
| `cargo run` | Build a program if needed, then run it. Cargo reuses an up-to-date build. `cargo build` only builds; `cargo test` builds and runs tests. |
| `--release` | Compile optimized code. Compilation usually takes longer, but simulations and search generally run faster. Use it for experiments and performance measurements. Omit it during development for faster compilation and easier debugging. This selects a build profile, not a publication or deployment action. |
| `--example random` | Select the Rust example program named `random`, defined in this repository. Other guides select programs such as `match_play` or `tournament`. |
| `--` | End Cargo's options. Everything after this separator is passed to the selected program. |
| `7 8` | Arguments for `random`: player 0's agent seed, followed by player 1's agent seed. Their meaning depends on the program; see its argument reference. |

For a first run, `cargo run --example random -- 7 8` is enough. Omitting
`--release` selects the development profile. Ordinary native development outputs
are under `target/debug/`; release outputs are under `target/release/`.

Other build options used in the manuals:

| Option or command | Meaning and when to use it |
|---|---|
| `--features mcts-json,replay` | Enable named optional parts of the Rust package. Here, `mcts-json` enables JSON search configuration and `replay` enables replay support. `inference` enables native portable-model inference. Follow the feature list shown for your workflow. |
| `-p tag-browser` | Select a package in the Cargo workspace, which groups related Rust packages. This selects the browser adapter rather than the default package. |
| `--manifest-path adapters/python/Cargo.toml` | Select a particular package manifest instead of the one Cargo would find from the current directory. The Python adapter has its own manifest. |
| `--target wasm32-unknown-unknown` | Build for WebAssembly in the browser instead of the host operating system. The corresponding Rust target must be installed first. |
| `--offline` | Forbid Cargo from accessing the network. All required dependency versions and registry information must already be cached locally. This is optional; omit it for normal use so Cargo can download missing dependencies. |
| `cargo fetch` | Download the locked dependencies in advance without building the project. Useful before working offline; dependencies for a separate manifest need a separate fetch. |
| `--locked` | Require Cargo to use an existing lockfile without changing dependency resolution; fail if it would need updating. In `cargo install`, this uses the tool package's supplied lockfile. It does not itself disable network access. |

`--offline` controls dependency access; it does not change game rules, seeds or
agent behavior, and is not required to repeat an experiment. A cache populated
for one platform or feature selection may still lack dependencies for another.
For the selected workflow, fetch/build while online before relying on offline use.

Python commands such as `.\.venv\Scripts\python.exe -m pip install ...` use the
interpreter inside the project's virtual environment, keeping its packages separate
from other Python installations. `-m NAME` runs an installed Python module;
`-c "..."` executes the quoted Python code. `maturin develop` builds the Rust
extension and installs it into that environment. Its `--release` and `--offline`
options select an optimized Rust build and prevent Cargo network access, respectively.
`$env:NAME = ...` sets an environment variable in the current PowerShell session;
the adapter build uses these variables to select the Python environment.

In option references, `[value]` means an optional argument; do not type the square
brackets. Names such as `NAME` and `ARGS...` are placeholders to replace, not
literal command arguments. Paths are relative to the repository root; quote a
path containing spaces. Python program options such as `--seed` belong directly
after the script name and need no Cargo separator.

## Rust: matches and tournaments

The `native` profile runs the following build and example. To do it manually:

```powershell
cargo fetch
cargo build --release --features mcts-json,replay,inference
```

Check the installation with a supplied matchup:

```powershell
cargo run --release --features mcts-json,replay --example match_play -- examples/matches/random-vs-osla.json
```

## Browser

The `browser` profile installs the Wasm tools and runs
`tools/browser/build.ps1`. For manual setup, install the Wasm tools:

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
```

Then follow [build and play](running/browser.md). Browser inference needs no Python/PyTorch
process; `serve-browser.ps1` uses Python only to serve the local application files.

## Python: training and Python experiments

The `python` profile performs these steps using the pins in
`adapters/python/pyproject.toml`. For manual setup, the current adapter uses
CPython 3.14. Install the base packages first:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "numpy==2.5.4" "maturin==1.15.0"
cargo fetch --manifest-path adapters/python/Cargo.toml
```

For Gymnasium episodes, install `gymnasium==1.4.0`. For neural workflows,
install the CPU package with
`.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu`.

Build/install the Rust adapter into that environment:

```powershell
$env:VIRTUAL_ENV = (Resolve-Path .venv).Path
$env:PYO3_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
.\.venv\Scripts\python.exe -m maturin develop --release --manifest-path adapters/python/Cargo.toml
.\.venv\Scripts\python.exe -c "import tag_rust; print(tag_rust.registered_games())"
```

Repeat the adapter build after changing Rust. Python source edits are available
through the editable installation. Choose a [training workflow](learning/README.md).

## Optional benchmark plots

The measurement reports include ready-made figures. To regenerate plots locally,
install Matplotlib into the Python environment (the recorded version is 3.11.2):

```powershell
.\.venv\Scripts\python.exe -m pip install "matplotlib==3.11.2"
```

Matplotlib is needed only by plotting tools, not by the engine, browser or training.

## Optional presentation previews

The editable PowerPoint generator uses Python's standard library. Its optional
`--preview` mode uses Pillow and installed Arial/Consolas fonts on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install Pillow
```

Pillow generates layout previews; PowerPoint reads the editable PPTX separately.

## Equivalent POSIX-shell commands

The recorded runtime checks use Windows. On a POSIX host with a working Rust
toolchain/native linker and Python 3.14 already installed, use these shell
equivalents. They describe command syntax; they are not a separate claim of
Linux/macOS runtime validation.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install "numpy==2.5.4" "maturin==1.15.0"
cargo fetch
cargo fetch --manifest-path adapters/python/Cargo.toml
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m maturin develop --release --manifest-path adapters/python/Cargo.toml
.venv/bin/python -c 'import tag_rust; print(tag_rust.registered_games())'
```

For episodes, add `.venv/bin/python -m pip install "gymnasium==1.4.0"`.
For neural workflows, add
`.venv/bin/python -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu`.
The Windows setup script has no POSIX counterpart yet.

Cargo/Node commands elsewhere are the same. Replace the Windows interpreter path
with `.venv/bin/python`; `.exe` binary suffixes and PowerShell launchers are
Windows-specific. Serve browser files with
`.venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory web` instead of
the PowerShell launcher. Install missing host tools using their platform-specific
vendor instructions; do not assume Windows wheels/bindings work on another host.

## Verify the tools

```powershell
rustc --version
cargo --version
py -3.14 --version
python --version
wasm-bindgen --version
node --version
git --version
.\.venv\Scripts\python.exe -c "import numpy; print(numpy.__version__)"
```

Check `gymnasium` and `torch` only if you selected their options. The setup
script's `-Check` mode performs those package checks for the selected profile.

For optional Java comparisons, also check `java -version`, `javac -version` and
`mvn -version`. The graphical test command is `node tools/browser/test-browser.mjs`;
build the Wasm files first using the [browser manual](running/browser.md).

Details: [Python builds/debugging](../reference/python/setup.md),
[browser tools and checks](../../web/README.md).
