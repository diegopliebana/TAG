//! Interfaces and helpers shared by games.
//!
//! ForwardModel is the rules interface: create a state, observe it for a player, list
//! choices and apply a move. Its associated types pair each game with its own state,
//! actions, observations, settings and errors. PlayerResult describes win/loss/draw;
//! it is separate from a numeric game score or a learned estimate.

pub mod checkpoint;
pub mod evaluation;
pub mod grid;
pub mod outcomes;
pub mod players;
pub mod progress;
pub mod random;
pub mod simulation;

// Learning notes: Clone permits an explicit .clone(); Copy permits implicit
// duplication and requires Clone. Neither trait is automatically implemented
// merely because fields are small: the derive opts in.
// PartialEq supplies == and !=. Eq requires PartialEq and promises reflexive
// equality (a value equals itself).
// A by-value non-Copy argument moves ownership; Copy arguments are duplicated.
// &T borrows shared access; &mut T borrows exclusive mutable access.
// Returning an owned non-Copy value transfers ownership; returning &T borrows it.

/// A player's seat index. The game validates whether that seat exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// Clone: explicit duplication; Copy: tiny seat value.
// PartialEq: compare seat indices; Eq: equality is reflexive (every seat equals itself).
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(any(feature = "replay", feature = "mcts-json"), serde(transparent))]
pub struct PlayerId(usize);

impl PlayerId {
    /// Construct a seat identifier; game operations validate the seat.
    ///
    /// # Arguments
    /// `index` is a copied zero-based seat number.
    ///
    /// # Returns
    /// A copied `PlayerId`; game operations validate it against their player count.
    pub const fn new(index: usize) -> Self {
        // index: copied usize; returns a new owned PlayerId.
        Self(index) // Stores the copied index.
    }

    /// Return the underlying copied `usize` seat index; the caller may still reuse this `PlayerId`.
    pub const fn index(self) -> usize {
        // self: by value, copied because PlayerId is Copy.
        self.0 // Returns a copied usize; caller's PlayerId remains usable.
    }
}

/// A sequential decision names one acting player or game completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Clone: explicit duplication; Copy: small decision value.
// PartialEq: compare variant and player; Eq: both have reflexive equality.
pub enum Decision {
    /// Player(PlayerId) carries an identifier.
    Player(PlayerId),
    /// Finished identifies game completion.
    Finished,
}

/// Separate from game score and runner failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Clone: explicit duplication; Copy: small result tag.
// PartialEq: compare result variants; Eq: every variant equals itself.
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "snake_case"))]
pub enum PlayerResult {
    Ongoing,
    Win,
    Draw,
    Loss,
}

/// Common rule operations with game-specific data types.
///
/// Mutable progress belongs in State. No cloning, hashing, serialization or
/// threading bounds are imposed. Each game supplies its own implementation of this interface.
pub trait ForwardModel {
    /// These are things the forward model of a game are going to have. Specific
    /// forward model implementations will have to implement a specific type (e.g. TicTacToeState).
    type State;
    type Action;
    type Observation;
    type Parameters;
    type Error;

    /// Create a new game state from configuration.
    ///
    /// # Arguments
    /// `parameters` is borrowed; `player_count` is copied. The model is passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(State)` transfers ownership of the new state to the caller.
    ///
    /// # Errors
    /// Unsupported player counts or invalid/resource-exceeding configuration return the game error.
    fn init_state(
        &self,                         // Shared borrow of model.
        parameters: &Self::Parameters, // Shared borrow; caller retains ownership.
        player_count: usize,           // Copied value.
    ) -> Result<Self::State, Self::Error>; // Returns owned state or owned error.

    /// Read how many seats the game state has.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// A copied `usize` player count.
    fn player_count(
        &self,               // Shared borrow of model.
        state: &Self::State, // Shared borrow of state.
    ) -> usize; // Returns a count by value (usize is Copy).
    /// Identify the next decision maker.
    ///
    /// # Arguments
    /// `state` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// A copied `Decision`: a player who may act, or completion.
    fn next_player(
        &self,               // Shared borrow of model.
        state: &Self::State, // Shared borrow; caller keeps the state.
    ) -> Decision; // Returns a decision by value (Decision is Copy).

    /// Create the information the requested player is allowed to see.
    ///
    /// # Arguments
    /// `self` and complete `state` are read-only references. `observer` identifies the
    /// player receiving the observation; the game validates that the seat exists.
    ///
    /// # Returns
    /// A separate Observation that remains valid if the live game changes. The current
    /// public games copy the board/state; hidden-information games must omit forbidden facts.
    ///
    /// # Errors
    /// An unknown observer returns the game's Error.
    fn observe(
        &self,               // Shared borrow of model.
        state: &Self::State, // Shared borrow; caller retains the state.
        observer: PlayerId,  // Copied seat value.
    ) -> Result<Self::Observation, Self::Error>; // Returns owned observation or owned error.

    /// List available actions using only the player's observation.
    ///
    /// # Arguments
    /// `self` and `observation` are read-only references and remain unchanged.
    ///
    /// # Returns
    /// A new `Vec<Action>` in the game's documented order. Inactive or finished observations
    /// have no choices in the current games. A caller may retain these action values.
    ///
    /// Supports planning and APIs that already hold an observation. Live execution
    /// uses available_actions directly from state and constructs an observation
    /// separately for the agent. Both paths must use the same game action rules and
    /// ordering for the acting player; hidden facts must not influence the list.
    fn offered_actions(
        &self,                           // Shared borrow of model.
        observation: &Self::Observation, // Shared borrow; caller retains ownership.
    ) -> Vec<Self::Action>; // Transfers ownership of list and actions to caller.

    /// List legal choices for a player in a supplied simulation state.
    ///
    /// # Arguments
    /// `self` and `state` are read through references; `actor` is the requested player.
    /// A trusted runner may supply the live state and pass only the resulting actions
    /// to an agent. Action contents and ordering must depend only on information the
    /// actor may know. Planning code uses a state it is allowed to inspect, such as a public-state
    /// copy or a properly sampled hypothesis, not the true hidden opponent data.
    ///
    /// # Returns
    /// A new list of Action values; a valid actor receives an empty list after completion.
    ///
    /// # Errors
    /// Unknown players and a player acting out of turn return the game's Error.
    fn available_actions(
        &self,               // Shared borrow of model.
        state: &Self::State, // Shared borrow of permitted simulation state.
        actor: PlayerId,     // Copied seat value.
    ) -> Result<Vec<Self::Action>, Self::Error>; // Returns owned list/actions or owned error.

    /// Apply one action and its automatic rule consequences to the supplied state.
    ///
    /// # Arguments
    /// `self` is read-only. `state` is accessed through an exclusive mutable reference;
    /// the caller keeps owning it. `actor` identifies the acting player and `action` is
    /// read through a reference, so the caller can retain the same action value.
    ///
    /// # Returns
    /// Unit on success: state now contains the board, progress and result after the move.
    ///
    /// # Errors
    /// Invalid player/action or an already finished game returns the game's Error before
    /// changing state. Rejected input must leave the supplied state unchanged.
    /// An internal failure while executing a legal action must be reported; it
    /// may leave partial changes. Abort that simulation/game and discard its state.
    /// Implementations need not clone state to recover from internal failures.
    fn next(
        &self,                   // Shared borrow of model.
        state: &mut Self::State, // Exclusive mutable borrow; caller still owns state.
        actor: PlayerId,         // Copied value.
        action: &Self::Action,   // Shared action borrow.
    ) -> Result<(), Self::Error>; // Returns `Ok(())` or an owned error.

    /// Query the result for a particular player.
    ///
    /// # Arguments
    /// `state` is passed by a shared (read-only) reference; `player` is a copied seat ID.
    ///
    /// # Returns
    /// `Ok(PlayerResult)` returns a copied outcome category, separate from score and execution errors.
    ///
    /// # Errors
    /// Unknown players return the game error.
    fn player_result(
        &self,               // Shared borrow of model.
        state: &Self::State, // Shared borrow.
        player: PlayerId,    // Copied value.
    ) -> Result<PlayerResult, Self::Error>; // Result tag by value, or owned error.

    /// Query the game-defined score for a particular player.
    ///
    /// # Arguments
    /// `state` is passed by a shared (read-only) reference; `player` is a copied seat ID.
    ///
    /// # Returns
    /// `Ok(f64)` returns that player's game-defined score.
    ///
    /// # Errors
    /// Unknown players return the game error.
    fn player_score(
        &self,               // Shared borrow of model.
        state: &Self::State, // Shared borrow; caller retains state.
        player: PlayerId,    // Copied value.
    ) -> Result<f64, Self::Error>; // Score by value (f64 is Copy), or owned error.
}

pub mod lines;

/// Optional simulation access for games whose observations contain all game facts.
/// Hidden-information search needs a separately designed determinisation or belief
/// sampler to construct possible states consistent with the player's information.
pub trait PublicInformationModel: ForwardModel {
    /// Reconstruct an exact, independently owned simulation state from a public view.
    ///
    /// # Arguments
    /// `observation` and the model are passed by a shared (read-only) reference.
    ///
    /// # Returns
    /// `Ok(State)` owns a snapshot with the view's board, rules and progress.
    /// The returned state owns an independent copy of the observation's snapshot.
    ///
    /// # Errors
    /// Implementations may return their game error if reconstruction is invalid.
    /// Both current games return `Ok` with a cloned view. Allocation follows the
    /// standard allocator's failure policy.
    fn simulation_state(
        &self,                           // Shared rules borrow.
        observation: &Self::Observation, // Shared public-view borrow.
    ) -> Result<Self::State, Self::Error>; // New owned state or owned game error.
}
