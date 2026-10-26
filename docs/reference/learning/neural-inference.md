# Run a small policy/value network on CPU

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

A policy gives a probability for each legal action. A value estimates the future
reward for declared players using the observer's permitted information. The optional `tag_rust.neural` module
computes both together with PyTorch. Rust still supplies observations and executes
actions. [Neural play and collection](neural-play.md) connects policies,
neural-value OSLA and native teacher episodes. [Neural training](neural-training.md)
updates and saves model weights.

## Setup

Reuse the [Python environment](../python/setup.md). From normal PowerShell at the
repository root, install the selected optional dependency manually:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.ones(2).numpy())"
```

The [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)
lists `torch-2.14.0+cpu-cp314-cp314-win_amd64.whl` for regular Windows x64
CPython 3.14. The package's `neural` extra pins the same release; the command
above explicitly selects CPU wheels. Ordinary `import tag_rust` does not import PyTorch. This executor supports CPU
inference; a GPU executor is not implemented.

## Define the network inputs and outputs

```python
import torch
import tag_rust
from tag_rust.learning.network import PolicyValueNetwork, infer
from tag_rust.learning.inputs import ModelContract, pack_observations

torch.manual_seed(42)
with tag_rust.create("tictactoe") as game:
    contract = ModelContract.from_spec(game.spec(), payloads=("cells",))
    model = PolicyValueNetwork(contract)
    observation = game.observe(0, payloads=("cells",))
    batch = pack_observations(contract, [observation])
    result = infer(model, batch)
    action = int(result.choose()[0])
    game.apply(action, position=observation.position)
```

This network has random initial weights; these predictions demonstrate the API.
Replace the game name with `connect4` to build its corresponding network. Changed
board settings need a separately compatible model. Dimensions and permanent
action IDs come from Rust's description. The supported cells format has own
pieces +1, opposing pieces -1 and empty cells 0 in x-major order.

For P players, the public context features contain P observer indicators,
P next-player indicators and one observer-to-act indicator. Each group of P uses
one entry set to 1 to
identify a seat (a **one-hot vector**). When play ends, next-player indicators
are all zero. With two players, this adds five numbers: default Tic-Tac-Toe has
14 inputs and 9 policy outputs; Connect 4 has 47 inputs and 7 policy outputs.
For observer 1 with player 0 next, the context is `[0,1, 1,0, 0]`.
The network shares two 64-unit hidden layers, each with a `tanh` activation,
then predicts raw action scores and a single `tanh` value in [-1,1]. For example,
`hidden_sizes=(32,32)` selects smaller layers. The model input/output specification
(`ModelContract`) records gamma 0.99 per learner step and the observing-player
value convention.

## Read a batch's predictions

`pack_observations` accepts observations from any supported seat and compatible
game instances. It checks identity, named tensor descriptions, values and masks,
then creates independently owned contiguous arrays in their declared types.
`infer` requires the same complete model input/output specification. For B rows
and A permanent actions, it returns:

| Field | Shape | Meaning |
|---|---|---|
| `logits` | [B,A] | Raw action scores, before masking. |
| `values` | [B] | Estimated observer returns. |
| `probabilities` | [B,A] | Legal action probabilities; illegal entries are zero. |
| `policy_rows` | [B] | True where the observer can choose an action. |
| `action_ids` | [B,A] | IDs returned by `choose()`, preserving action meaning. |
| `value_players` | tuple or None | Absolute player order, or None for each row's observer. |

Rows from an opponent turn or a terminal position have zero probabilities.
`choose()` returns -1 for those rows. Otherwise it chooses the first maximum;
`choose(numpy.random.default_rng(seed))` samples from legal probabilities using
the supplied random stream. Only submit actions from applicable rows. Rust
still checks the position token when applying an action.

Terminal predictions are raw network outputs. Use the game's declared terminal
rewards; the current two games use +1 win, 0 draw, -1 loss. Intermediate opponent-turn rows
can be evaluated for their observing-player value. [Neural play](neural-play.md)
connects these rules to candidate requests and episode collection.

## Ownership, devices and checks

Constructing a batch copies the selected observation arrays, context features,
action IDs and masks. Inference validates
buffers again and copies named arrays into local contiguous CPU allocations;
`torch.from_numpy` shares those local allocations. The fully connected model
explicitly converts inputs to float32 and flattens them in the input order declared
by `ModelContract`.
Float64 inputs are narrowed; overflow and integer precision loss are rejected.
Supply an explicitly versioned scaled encoding for large integers.
The result arrays are independent copies and remain usable after another call
or closing the game. No zero-copy or speed claim is made. Keep model parameters
fixed while any related external request is outstanding.

Inference uses `torch.inference_mode()` to avoid recording gradients and temporarily
switches the model to evaluation mode. This disables training-time dropout and
running-statistic updates in custom models. Each module's previous training flag
is restored, including after failure. `forward()` retains ordinary gradient
support for later training.
The executor requires float32 CPU parameters and rejects other devices/types. Dimensions,
finite logits/values, value range and complete batch/model contracts are checked
before returning results. The implemented [training workflow](neural-training.md)
saves model files; [FrozenModel](neural-play.md) identifies the fixed parameters
used by an evaluation request.

Run the focused checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s adapters/python/tests -p 'test_neural*.py' -v
```

The tests use real Rust observations for packing and an independent NumPy formula
for fixed-weight predictions. Inference tests explicitly skip when PyTorch is
missing; those skips do not establish neural runtime correctness.

## Combine inputs for tabletop observations

`ModelContract.from_spec(spec, payloads=("public_tokens", "own_hand", "hand_valid"))`
selects inputs in a recorded order. This illustrative naming could describe a
token vector, a padded table of the observer's cards and a Boolean indicator per
hand slot. Game authors define all meanings and hide unknown cards before encoding.
The model receives the observation's payloads; it has no full-state argument.
Only requested inputs enter the model, so unused payloads never enter inference.
For the real games, this complete example selects two representations:

```python
with tag_rust.create("connect4") as game:
    contract = ModelContract.from_spec(game.spec(), payloads=("board", "cells"))
    batch = pack_observations(contract, [game.observe(0)])
    assert batch.payloads["board"].shape == (1, 3, 6, 7)
    assert batch.inputs.shape == (1, 173)  # 126 board + 42 cells + 5 context
    result = infer(PolicyValueNetwork(contract), batch)
```

Inputs can use float32, float64, int8, int64, uint8 or bool in this Python
interface. Tensor names, shapes, dtypes, axis/component meanings and bounds are
part of the contract. Model compatibility also checks game/configuration,
versions, action mapping, context format and the value specification.
`max_bytes` defaults to 64 MiB and checks packed numeric storage before allocation;
it excludes model weights, temporary copies and total process memory.

## Declare the value outputs

`ValueSpec()` retains the current observer-return defaults: one value per row,
[-1,1], gamma 0.99 per learner step. Other tasks declare their own semantics:

```python
from tag_rust.learning.inputs import ValueSpec
points = ValueSpec(players=(2, 0, 1), bounds=None, units="future-points",
                   gamma=0.95, discount_clock="round")
```

With this illustrative three-player contract, `values` has shape `[B,3]` in the
declared player order. All three predictions use the same row's permitted
observation; predicting another player's reward grants no access to their hidden
hand. Each output is independently learned. A single explicitly named player still
keeps `[B,1]`. `players=None` uses `[B]` for the observer of each row.
Finite bounds select a shifted/scaled tanh output; `bounds=None` selects a linear
output. Neither scores nor rewards are automatically inferred from result labels.

## Score supplied action candidates

`ActionSpec(size=0, mapping="candidates")` describes a variable candidate policy.
Its `ModelContract` additionally needs `candidate_tensor`, a `TensorSpec` for one
action's features. Each row supplies `CandidateActions(features, action_ids)`:
features have shape `[C,*candidate_shape]`, IDs are unique nonnegative int64 values
identifying authoritative actions for that decision. These descriptions must use
only the information the player may know.

Pass one such list per observation to `pack_observations(..., candidates=lists)`.
It pads to the largest C, records a separate legal mask and uses -1 for padded
IDs. The network combines each candidate with the observation's shared features
and returns `[B,C_max]` logits/probabilities. Padding has zero probability. Empty
lists are valid value-only rows. `choose()` returns the supplied ID, preserving
the action mapping even when candidate order changes.

The caller retains the association between each ID and its original Rust action
and position/request token. Use the authoritative API to validate and apply the
choice. General candidate descriptions and Python scoring are implemented here;
the current registered games still use fixed IDs. Adding a candidate-action game
requires its Rust observation and action-feature encoders, request/action mapping and adapter execution
tests. A neural candidate policy is also distinct from OSLA, which scores simulated
successor positions. [Neural play](neural-play.md) connects these operations for the supported episode games.

## Extension boundaries

`infer` can accept another CPU `torch.nn.Module` with a `contract` attribute and
`forward(payloads, context, candidates=None)`. It receives a dictionary of typed
tensors in their original shapes, public float32 context and optional padded
candidate tensors. Return policy logits and values in the contract's declared
shapes. This allows a different architecture without rewriting validation and
selection. Callers must serialise model updates/inference; temporary evaluation
mode does not make concurrent mutation safe.

The supplied packer handles fixed-size features/grids and explicit padding/masks.
It rejects graph/image payloads and variable tensor dimensions. Graph offsets,
image preprocessing and recurrent memory require appropriate adapters and models.
Turn context currently describes one next player; simultaneous actions need an
explicit decision representation before support can be claimed.

Rust `EncodingConfiguration::Named` stores canonical game-defined string settings
without requiring a board. `Placement` retains existing board-game configuration.
This Rust enum stores either variant together with its data, comparable to a Java
sealed interface with record implementations or a C++ `std::variant`. Its named
settings use an ordered map; cloning an identity copies those owned settings.
Python retains width/height/target compatibility fields (zero for general games)
and canonical named settings. `ActionMapping::Indexed` delegates permanent-ID
mapping to the game; `Candidates` declares a variable list. Coordinate helpers
reject these two mappings. Line-heuristic data is optional (`None`).

The existing Rust observation payload container and grouped execution still emit
float32 arrays. Other number types work at the Python model boundary and are
covered by synthetic fixtures; native encoders for additional number types need an adapter
extension. Multiplayer fixture tests establish interface behavior, while the only
implemented playable games remain Tic-Tac-Toe and Connect 4. See
[test_neural_general.py](../../../adapters/python/tests/test_neural_general.py)
for labelled hand/token, player-value, candidate-order and structured-model proofs.

For weight updates, model files and checkpoint continuation, use the implemented
[training workflow](neural-training.md). Implementations are now in topic folders;
the [package map](../../../adapters/python/python/tag_rust/README.md) explains
the paths and retained compatibility imports.
