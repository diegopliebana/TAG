# Reading this framework from Java or C++

Updated 15-10-2026 15:10 (Europe/Paris). Read this alongside the
[game workflow](implementing-games.md) or [agent workflow](../agents/implementing-agents.md). It explains the Rust notation used here with optional Java/C++ comparisons.
You can use it even if your main language is Python: start with the meaning column
and the example. [The project introduction](../../start-here.md) explains game/AI terms.

## Values, ownership and references

Every value has an owner. Passing an owned value to a function usually transfers
ownership: the caller cannot use it afterward unless the type is `Copy`.
Passing a reference lets the function use the existing value without taking it.
Rust checks that references cannot outlive the data they point to.

| Notation | Meaning here | Java/C++ comparison |
|---|---|---|
| `state: State` | The variable owns a game state. Our game state types implement `Clone` and **do not implement `Copy`**. Passing one by value transfers ownership to the recipient. | Like constructing a C++ value with `State next = std::move(state)`. Rust also prevents the caller from using the moved variable until it is assigned another value. |
| `state: &State` | A shared reference gives read-only access to an existing state. **Shared references implement `Copy`**: assigning or passing this reference copies the reference, while both references point to the same state. | `const State& state` in C++ also provides read-only access to an existing object. Rust checks how long the reference may be used and restricts mutation while it is borrowed. |
| `state: &mut State` | A mutable reference gives exclusive access to an existing state for reading and modification. **Mutable references do not implement `Copy`**. A function call can temporarily borrow through one, then return access to the caller. | `State& state` in C++ allows modification of the referenced object. Rust additionally enforces exclusive access for the duration of the mutable borrow. |
| `let mut state = ...` | Declares a variable whose value may be modified or replaced. Ownership and copying depend on the value's type. | Similar to a local C++ variable declared `State state = ...`, which permits modification and reassignment. Rust requires the explicit `mut` keyword. |
| `Vec<Action>` | A growable collection that owns its element storage and the action values in it. When its owner leaves scope, Rust automatically cleans up its elements and frees its allocated storage. The example below explains this cleanup. | Similar to `std::vector<Action>` in C++, whose destructor cleans up its elements and storage. Java's `ArrayList<Action>` stores references to objects managed by the garbage collector. |
| `&[Action]` | A shared reference to a sequence of actions. It records where the sequence starts and its length, and provides read-only access to the original actions. Rust ensures those actions remain alive and in place for as long as the slice is used. | Similar to `std::span<const Action>` in C++20. |
| `Copy` | A trait allowing assignment and argument passing to copy a value implicitly, so the original remains usable. Integers, player IDs and shared references support it. Every field of a `Copy` struct must itself support `Copy`. | Like assigning a C++ or Java integer: `int b = a;` leaves both variables usable. |
| `Clone` | A trait providing the explicit `.clone()` operation, which creates another value of the same type. Each type defines how to reproduce its fields and whether the resulting value owns separate data or shares some data with the source. | Similar to a C++ copy constructor in that the type determines how its data is copied. Rust calls the operation explicitly with `value.clone()`. |

Rust calls a value's cleanup **dropping** the value. For a vector, this means
cleaning up each element and then freeing the vector's allocated storage. For
example, an element that owns a string releases that string's storage during its
cleanup. Rust automatically drops a local value when its owner leaves its
**scope**, the region of code in which that variable is defined:

```rust
{
    let actions = vec![1, 2, 3];
    println!("{} actions", actions.len());
} // actions leaves scope: Rust cleans up the vector here.
```

This resembles a C++ destructor running when a local `std::vector` leaves scope.
If ownership is transferred to another variable or function, the new owner becomes
responsible for eventual cleanup. Calling `drop(value)` explicitly transfers the
value to a standard-library function that performs its cleanup immediately.

Cloning a `Vec<i32>` creates a new vector with its own element storage.
Changing an integer in the new vector leaves the original vector unchanged.
Cloning an `Rc<T>` (a reference-counted pointer) creates another pointer to the same
allocation and increments its owner count. The two operations both implement
`Clone`, with the storage behavior defined by their respective types.

```rust
let original = vec![1, 2];
let mut duplicate = original.clone();
duplicate[0] = 9;
assert_eq!(original[0], 1);

let original = std::rc::Rc::new(vec![1, 2]);
let another_owner = original.clone();
assert!(std::rc::Rc::ptr_eq(&original, &another_owner));
```

Our game states clone their board storage so simulations can change independently;
snapshot tests check this property. An agent's `Checkpoint` stores its settings,
memory and random-number-generator progress so restoration reproduces future choices.

A **lifetime** is the region of a program during which a reference may be used
because the referenced data remains valid and the borrowing rules permit access.
The compiler checks this when compiling the program. A name such as `'a` lets a
function express how long a returned reference may be used in terms of its inputs:

```rust
fn first<'a>(actions: &'a [Action]) -> Option<&'a Action> {
    actions.first()
}
```

The returned reference points into the supplied action sequence. The shared `'a`
requires that any use of that reference stays within the period for which the
sequence is borrowed. For example, the caller must finish using the returned
reference before dropping or modifying the vector that owns the actions. The
compiler usually infers these annotations; the same function can be written as
`fn first(actions: &[Action]) -> Option<&Action>`.

## Traits, associated types and bounds

We will build a small example in steps. Its `ExampleGame` trait has one operation,
creating an initial board. The framework's `ForwardModel` uses the same Rust
features for a larger set of game operations. The code blocks in this section
belong together.

A `struct` declares stored data. Here, `Board` stores a vector whose entries say
whether each cell is occupied:

```rust
#[derive(Debug, Clone, PartialEq)]
struct Board {
    cells: Vec<bool>,
}
```

`#[derive(Debug, Clone, PartialEq)]` asks Rust to generate three implementations
using the fields of `Board`. `Debug` lets us print a board with `{:?}`. `Clone`
lets us create another board by cloning the `cells` vector into separate storage.
`PartialEq` lets us compare boards with `==`.

An `impl Board` block adds methods to `Board`. This one asks whether all cells
are empty:

```rust
impl Board {
    fn is_empty(&self) -> bool {
        self.cells.iter().all(|occupied| !occupied)
    }
}
```

`&self` borrows the board for reading. `iter()` visits its cell values, and
`all(|occupied| !occupied)` returns true when every cell is false. The expression
`|occupied| !occupied` is a **closure**, a small function written where it is used:
it receives a cell value named `occupied` and returns its Boolean opposite.

A **trait** declares operations that implementing types must provide, much like
a Java interface. It can also declare **associated types**: type names whose
concrete definitions are supplied by each implementation.

```rust
trait ExampleGame {
    type State;
    type Action;

    fn initial_state(&self) -> Self::State;
}
```

`ExampleGame` requires two associated types, `State` and `Action`, and an
`initial_state` method. `Self` means the type implementing the trait, so
`Self::State` means the state type selected by that implementation.

We can now implement the trait for `ThreeCellGame`:

```rust
struct ThreeCellGame;

impl ExampleGame for ThreeCellGame {
    type State = Board;
    type Action = usize; // An action identifies a cell by its index.

    fn initial_state(&self) -> Board {
        Board { cells: vec![false; 3] }
    }
}
```

`struct ThreeCellGame;` declares a type with no stored fields. Its trait
implementation selects `Board` for `State` and `usize` for `Action`; an action
will identify a cell by index. `initial_state` creates a board with three false
entries. Thus `ThreeCellGame` supplies the operations, while `Board` stores the
changing data.

A function can work with several game types by using a **generic type parameter**.
Here, `G` stands for whichever game type the caller selects:

```rust
fn copy_state<G>(state: &G::State) -> G::State
where
    G: ExampleGame,
    G::State: Clone,
{
    state.clone()
}
```

The `where` clause lists **bounds**, requirements checked by the compiler.
`G: ExampleGame` requires `G` to implement `ExampleGame`. `G::State: Clone`
requires that game's state type to implement `Clone`, so the function can call
`state.clone()`. Its argument borrows a state and its return value owns the copy.

For `ThreeCellGame`, `G::State` is `Board`, which supports `Clone` through the
derive above. We can therefore call the function and change the copied board:

```rust
let game = ThreeCellGame;
let original = game.initial_state();
let mut copy = copy_state::<ThreeCellGame>(&original);
copy.cells[0] = true;

assert!(original.is_empty());
assert!(!copy.is_empty());
assert_ne!(original, copy);
```

The `::<ThreeCellGame>` explicitly selects `G`. The compiler needs that choice
here because several games could use `Board` as their state type. The assertions
check that changing `copy` leaves `original` empty and makes the boards different.
`assert_ne!` uses the generated `PartialEq` and `Debug` implementations to compare
them and show their values if the assertion fails.

The framework uses these same ideas in `ForwardModel`: each game selects its
state, action and observation types. Lookahead code adds a `Clone` bound where it
needs to copy states. Requiring that capability on the copying operation keeps
the basic game interface usable by callers that only need to run moves.

## Enums and errors

An **enum** defines a value that holds one of several alternatives, called
**variants**. Each variant can carry its own data, much like the alternatives in
a C++ `std::variant`. Consider this small controller example:

```rust
enum Controller<A> {
    Human,
    Ai(A),
}
```

`Human` carries no extra data. `Ai(A)` carries a value of type `A`, chosen when we
use the enum. We will use `Controller<String>` so an AI carries its name. In the
framework, `Controller<A>` instead stores an agent of type `A`.

A `match` examines the stored variant and runs the corresponding **arm**, the
code after `=>`:

```rust
fn controller_name(controller: &Controller<String>) -> &str {
    match controller {
        Controller::Human => "Human",
        Controller::Ai(name) => name.as_str(),
    }
}
```

In the `Human` arm, the function returns the text `"Human"`. In the `Ai(name)`
arm, `name` refers to the string carried by that variant, and `as_str()` gives
read-only access to its text. The function borrows the controller, so the caller
retains ownership. Adding another variant requires handling it in this match.

Now we can create both kinds of controller and check their displayed names:

```rust
let human: Controller<String> = Controller::Human;
let ai = Controller::Ai(String::from("Random"));
assert_eq!(controller_name(&human), "Human");
assert_eq!(controller_name(&ai), "Random");
```

Two standard enums are useful when choosing actions. `Option<T>` holds either
`Some(value)` or `None`, representing a present value or absence. `Result<T, E>`
holds either `Ok(value)` for success or `Err(error)` for failure. `T` names the
value type and `E` names the error type.

Suppose a caller supplies an optional list index. We want to report two possible
failures: the caller supplied no index, or the supplied index is outside the list.
We can describe those failures with another enum:

```rust
#[derive(Debug, PartialEq)]
enum SelectionError {
    MissingSelection,
    InvalidIndex { index: usize, available: usize },
}
```

`MissingSelection` identifies the first failure. `InvalidIndex` also carries the
attempted index and the number of available entries, so a caller can explain the
problem. Deriving `Debug` and `PartialEq` lets our examples display and compare
these errors.

The selection function returns `Result<usize, SelectionError>`: either the
selected action number or one of the errors above.

```rust
fn select_action(
    actions: &[usize],
    selection: Option<usize>,
) -> Result<usize, SelectionError> {
    let index = selection.ok_or(SelectionError::MissingSelection)?;
    actions.get(index).copied().ok_or(SelectionError::InvalidIndex {
        index,
        available: actions.len(),
    })
}
```

`selection.ok_or(SelectionError::MissingSelection)` converts the optional index
into a result: `Some(index)` becomes `Ok(index)`, and `None` becomes
`Err(SelectionError::MissingSelection)`. The following `?` obtains the index on
success or immediately returns that error from `select_action`.

`actions.get(index)` returns `Some(&action)` when the index exists, or `None`
when it is outside the list. `copied()` copies the `usize` through the reference,
giving us an owned action number. The final `ok_or` converts that option into
either `Ok(action)` or `Err(SelectionError::InvalidIndex { ... })`.

These calls exercise success and both failures:

```rust
let actions = [2, 5];
assert_eq!(select_action(&actions, Some(1)), Ok(5));
assert_eq!(select_action(&actions, None), Err(SelectionError::MissingSelection));
assert_eq!(
    select_action(&actions, Some(4)),
    Err(SelectionError::InvalidIndex { index: 4, available: 2 }),
);
```

Index 1 selects action 5. An absent index produces `MissingSelection`, and index
4 produces `InvalidIndex` because the list has only two entries. The framework
similarly returns typed errors when an agent's selection fails validation; each
game supplies its own action type.

Calling `unwrap()` on an `Option` or `Result` extracts its value when present or
successful, and panics on absence or failure. The fixed-data example below uses
it to stop if an unexpected failure occurs. Code handling user input should
handle errors explicitly, for example with `match`, and report the problem.

## Follow a complete native call

The runnable [Random example](../../../examples/random.rs) contains both games.
Its Tic-Tac-Toe flow is:

```rust
let model = ttt::TicTacToeForwardModel;
let mut state = model
    .init_state(&ttt::TicTacToeParameters::default(), 2)
    .unwrap();
run_game(
    &model,
    &mut state,
    &mut [RandomAgent::new(7), RandomAgent::new(8)],
)
.unwrap();
```

`model` stores the rules; mutable progress belongs in `state`. Initialization
borrows default parameters and returns an owned state. The runner borrows the
model read-only, and the state and agents exclusively while the match runs.
It asks the game for the acting player's observation and actions, asks that
player's agent for an index, validates it and applies the selected move.
Each agent owns its seeded generator. After the runner returns, the caller can
inspect `state`; no ownership transfer or copy occurred at that call.

Run it from the repository root:

```powershell
cargo run --offline --example random
```

Use Session to request one move at a time. RecordedSession adds saved move history
and lets an AI move be calculated, then applied or cancelled. BranchSession adds
alternative continuations without deleting the original moves. Each wrapper stores
the simpler object inside it and delegates rule operations to the game's forward model.
Browser scheduling and file I/O belong to adapters; see the
[browser guide](../browser/integration.md) and [persistence guide](../persistence/recordings.md).

When adding an agent, start with its policy and native example, then integrate
checkpoints, mixed dispatch, files and browser controls using the agent workflow.
When adding a game, follow the game workflow's completion checks before adding
optional platforms. You do not need to understand every generic bound in replay
loading before writing a basic rule transition.
