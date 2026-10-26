# Observation specifications, versions and limits

For operating instructions, use the [short manual](../../guides/running/game-operations-from-python.md). This page contains
the detailed reference.

[Learning guide entry point](overview.md)

A caller needs to know what numbers mean before using them. `LearningSpec`
describes the available inputs for one game configuration. `EncodedObservation`
then contains the values for one player's view of a position. This page explains
how to read the description and how requests are checked.

## LearningSpec describes five things

| Field | What it explains |
|---|---|
| `identity` | Which game, versions and configuration determine the meaning of these inputs. |
| `payloads` | Which named representations can be requested, with their array formats. |
| `metadata` | Formats for the progress counters and result labels returned with every encoding. |
| `actions` | How many action IDs exist and how they map to moves. |
| `heuristic` | Optional cell groups supporting the placement heuristic; None for games without that capability. |

For 3 by 3 Tic-Tac-Toe, `payloads` has two entries: `Features` describing `cells`
and `Grid` describing `board`. `metadata` has two entries describing progress
and results. `actions.size` is 9, and `heuristic` contains eight lines of three
cell indices. Connect 4 has the same named representations and metadata formats,
with dimensions and line groups appropriate to its board.

`LearningSpec` groups several kinds of descriptions. The `payloads` describe
observation arrays intended for learning-model inputs. The `heuristic` field
serves the separate hand-written evaluator proof; it is not part of those model
inputs. See [the heuristic support guide](heuristic-lines.md) for that distinction
and a worked calculation.

## Read an individual array description

Each current observation representation contains a `TensorSpec`, which has these fields:

| Field | Tic-Tac-Toe board example | Meaning |
|---|---|---|
| `name` | `"board"` | The name passed in the request. |
| `dtype` | `DType::Float32` | The type of each stored number, here a 32-bit floating-point value. |
| `shape` | `[3, 3, 3]` | Number of entries along each dimension. Multiplying them gives 27 values. |
| `axes` | `["channel", "y", "x"]` | Meaning of each dimension, in shape order. |
| `components` | `["own", "opponent", "empty"]` | Meaning of the three channel positions. |
| `bounds` | `Some((0,1))` | Inclusive uniform endpoints: every board value lies between 0 and 1. Python exposes the pair as a tuple. |

For Connect 4, the default `board` shape changes to `[3, 6, 7]` and contains
126 values. The axes and channel meanings stay the same. For `cells`, the axis
is `"cell_x_major"`, and components describe the value codes `"own=1"`,
`"opponent=-1"`, `"empty=0"`.

The descriptions refer to one observation. Python batching can group B
Connect 4 boards into a batch with shape `[B, 3, 6, 7]`. That added first dimension
counts observations; the current Rust encoder returns each observation separately.

Bounds are optional integral endpoints, also usable for float32 arrays. Cells
declare (-1,1), board (0,1), progress (0,i64::MAX), and results (0,3). None means
no range has been declared; a Gym space mapper rejects missing/unsupported bounds
instead of guessing. Fractional or per-entry endpoints need a future extension.

## Check the meaning as well as the shape

Rust `EncodingIdentity` records `game`, `rules_version`, `encoding_version`,
`configuration` and `player_count`. `EncodingConfiguration::Named` holds canonical
game-defined string settings, such as a deck/content identity and round limit.
`Placement { width, height, target }` retains the current board settings.
For default Connect 4 the game is
`"connect4"`, dimensions are 7 by 6, target is 4 and player count is 2. Both
current implementations use rules version 1 and encoding version 1.

Rust's width()/height()/target() helpers return placement settings, or zero for a
general configuration. Python retains these fields for existing users and saved
tabular models, and normalizes named configuration entries into an immutable
sorted tuple. Neural models use the complete identity and selected tensor
descriptions; they do not require board geometry.

Two boards with the same dimensions can have different winning targets. Two
encodings of the same board can put channels in a different order. A learned
model must therefore check the agreed game/configuration/version information
alongside array dimensions. An intentional change to array order or meaning
requires an explicit encoding-version decision. The rules version must agree
with the game's persistence implementation.

## Choose what to request

Pass observation-array names to `encode`:

| Request | Contents of `EncodedObservation.tensors` |
|---|---|
| `&["cells"]` | One vector of relative cell values. |
| `&["board"]` | One three-channel board array. |
| `&["cells", "board"]` | Both arrays in that order. |
| `&[]` | An empty vector; observer, progress, results and action information are still returned. |

Each requested array owns its data. A caller can modify it independently of the
game and observation. Requesting only the representation needed by an algorithm
avoids creating the other array.

## Limits and failures

`LearningLimits` defaults to at most 4,096 board cells and 64 MiB of numeric/index
data per operation. The cell limit is separate from the game's board limit: a
board allowed by the rules can exceed the default learning limit.

When building a description, the byte check counts the exported line indices.
When encoding an observation, it counts the requested float arrays, the mask,
offered IDs, four progress counters and two result tags. Game states, strings,
container bookkeeping and later Python objects have additional memory costs.
`max_bytes` specifies the encoder's numeric/index accounting.

`cells` uses `4 * width * height` bytes of numeric values; `board` uses
`12 * width * height`. Requesting both for default Connect 4 therefore produces
`168 + 504 = 672` bytes of float values, plus the always-returned numeric data.

| LearningError | Example and meaning |
|---|---|
| `Invalid(message)` | A repeated payload name, an out-of-range action coordinate, or inconsistent observation data. The message explains the problem. |
| `UnsupportedPayload(name)` | Asking for `"image"` when this game advertises only `"cells"` and `"board"`. |
| `ResourceLimit` | A board or requested output exceeds the configured cell/byte limits. |
| `Overflow` | A size calculation or conversion exceeds the range of its integer type. |

The code uses checked size arithmetic and converts counters to int64 with range
checks. These failures leave the input observation and game unchanged. The
`LearningGame` methods return `Result`, allowing a caller to report the error or
choose a smaller request.

## Image and graph descriptions

`PayloadSpec` also supports descriptions for images and graphs. These provide
standard fields for additional observation encoders. Tic-Tac-Toe and Connect 4
produce `cells` and `board`; Can't Stop and Love Letter use named numeric features
and histories. Image/graph execution needs an explicit compatible producer/model.

`ImageSpec` describes a pixel tensor along with its color space, value range,
renderer identity and rendering settings. For example, a future renderer might
describe a height-by-width RGB image with three color channels. Those settings
allow callers to distinguish inputs drawn by different rendering conventions.
Rendering belongs outside the rules and must respect the observer's information.

`GraphSpec` describes node, edge and whole-graph feature arrays. Nodes can represent
entities, and edges their connections. Its `edge_index` description uses int64
shape `[2, edge_count]`: one row of source-node indices and one of destinations.
It also records whether edges are directed and how multiple graphs are grouped.
Actual graph inputs need their own observation encoder and tests.

All these types are defined in [types.rs](../../../src/learning/types.rs).
For concrete values matching these formats, read [observation arrays](observation-arrays.md).
