//! Describe player observations as numeric arrays, then fill those arrays.
//!
//! LearningGame is an optional interface implemented by the game model. It specifies
//! array names/shapes/meanings and stable action IDs without depending on Python,
//! PyTorch or a renderer. See docs/reference/learning/overview.md for a worked example.
mod placement;
mod types;
use crate::core::ForwardModel;
pub(crate) use placement::{PlacementView, check_placement_size, encode_placement, placement_spec};
pub use types::*;

/// Operations for describing a game's observations as arrays and filling those arrays.
/// Implement this trait on the existing game model. It uses that model's Observation
/// and Action types, so no separate Python game definition is needed.
/// Encode only what the observing player is allowed to know. Full simulation state
/// may contain hidden facts and must not be used as an unchecked substitute.
pub trait LearningGame: ForwardModel {
    /// Describe the array formats and action IDs for a particular game configuration.
    ///
    /// # Arguments
    /// `parameters` is a read-only reference to settings such as board size.
    /// `limits` gives the maximum cell count and numeric-data size this operation may use.
    ///
    /// # Returns
    /// A separate LearningSpec describing available arrays, their shapes and meanings,
    /// action representation, and optional cell groups for the placement heuristic.
    ///
    /// # Errors
    /// Invalid game settings, size calculations that overflow, or data exceeding limits.
    fn learning_spec(
        &self,
        parameters: &Self::Parameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError>;

    /// Describe a configuration with an explicit number of player seats.
    /// Existing fixed-player games retain their original description method.
    fn learning_spec_for_players(
        &self,
        parameters: &Self::Parameters,
        players: usize,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        let spec = self.learning_spec(parameters, limits)?;
        if spec.identity.player_count != players {
            return Err(LearningError::Invalid(
                "encoding player count differs from requested game".into(),
            ));
        }
        Ok(spec)
    }

    /// Turn one player's observation into the requested numeric arrays.
    ///
    /// # Arguments
    /// `observation` is a read-only reference to information that player can see.
    /// `payloads` lists array names, such as ["cells", "board"]. An empty list requests
    /// only progress, results and action information. `limits` restricts output size.
    ///
    /// # Returns
    /// Arrays with their own storage, plus observer, next player, progress and actions.
    /// Changing these returned values cannot change the observation or live game.
    ///
    /// # Errors
    /// An unknown or repeated array name, inconsistent observation data, a number too
    /// large for the output type, or an exceeded size limit.
    fn encode(
        &self,
        observation: &Self::Observation,
        payloads: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError>;

    /// Find the policy-output number assigned to a concrete action.
    ///
    /// # Arguments
    /// `observation` supplies the decision context; `action` is the move to identify.
    /// Both are read through references and are left unchanged.
    ///
    /// # Returns
    /// For a fixed policy, an ID whose meaning stays fixed during the configuration.
    /// For a candidate policy, a decision-local ID retained with that observation's
    /// action mapping. Executors must preserve and validate that decision context.
    ///
    /// # Errors
    /// An action outside the configuration's representable coordinates.
    fn action_id(
        &self,
        observation: &Self::Observation,
        action: &Self::Action,
    ) -> Result<usize, LearningError>;

    /// Turn a policy-output number back into a concrete action.
    ///
    /// # Arguments
    /// `observation` supplies the configuration/decision through a reference; `id` is the number.
    ///
    /// # Returns
    /// A new action value. Decoding does not make it legal: the game still checks whether
    /// the target cell is empty or the target column has room.
    ///
    /// # Errors
    /// An ID outside the set of actions described for this game configuration.
    fn action_from_id(
        &self,
        observation: &Self::Observation,
        id: usize,
    ) -> Result<Self::Action, LearningError>;
}
