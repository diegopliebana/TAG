//! Learning registration delegates shared placement encoding, never full-state reflection.
use super::*;
use crate::core::{ForwardModel, PlayerId};
use crate::learning::*;
/// Describe the game/version and supplied board dimensions/target length.
fn identity(width: usize, height: usize, target: usize) -> EncodingIdentity {
    EncodingIdentity {
        game: "tictactoe",
        rules_version: 1,
        encoding_version: 1,
        configuration: EncodingConfiguration::Placement {
            width,
            height,
            target,
        },
        player_count: 2,
    }
}
/// Read the board dimensions and describe all action IDs, whether currently legal or not.
fn actions(o: &TicTacToeObservation) -> ActionSpec {
    ActionSpec {
        size: o.grid_size() * o.grid_size(),
        mapping: ActionMapping::CellXMajor,
    }
}
impl LearningGame for TicTacToeForwardModel {
    /// Read configuration `p` and return array/action descriptions, using `limits` for maximum sizes.
    /// Invalid settings, size overflow or exceeding the numeric-data limit return an error.
    fn learning_spec(
        &self,
        p: &TicTacToeParameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        let (w, h, k) = (p.grid_size, p.grid_size, p.grid_size);
        let spec =
            crate::learning::placement_spec(identity(w, h, k), ActionMapping::CellXMajor, limits)?;
        self.init_state(p, 2)
            .map_err(|e| LearningError::Invalid(e.to_string()))?;
        Ok(spec)
    }
    /// Read player observation `o`; `payloads` names the arrays requested and `limits` caps their size.
    /// Return separate arrays, progress and actions, or LearningError explaining why encoding failed.
    fn encode(
        &self,
        o: &TicTacToeObservation,
        payloads: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError> {
        let (w, h, k) = (o.grid_size(), o.grid_size(), o.grid_size());
        crate::learning::check_placement_size(&identity(w, h, k), limits)?;
        // The shared encoder borrows permitted cells without copying a temporary board.
        let occupant = |index| match o.board()[index] {
            Cell::Empty => None,
            Cell::X => Some(PlayerId::new(0)),
            Cell::O => Some(PlayerId::new(1)),
        };
        let results = (0..2)
            .map(|seat| {
                o.player_result(PlayerId::new(seat))
                    .map_err(|e| LearningError::Invalid(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offered_ids = self
            .offered_actions(o)
            .iter()
            .map(|a| self.action_id(o, a))
            .collect::<Result<Vec<_>, _>>()?;
        crate::learning::encode_placement(
            crate::learning::PlacementView {
                identity: identity(w, h, k),
                mapping: ActionMapping::CellXMajor,
                observer: o.observer(),
                decision: o.next_player(),
                counters: o.counters(),
                results,
                cell_count: o.board().len(),
                occupant,
                offered_ids,
            },
            payloads,
            limits,
        )
    }
    /// Read dimensions from `o` and convert action `a` to its fixed ID, or report out-of-bounds coordinates.
    /// Decoding an ID does not make the move legal; consult the action mask and game rules.
    fn action_id(&self, o: &TicTacToeObservation, a: &PlaceMark) -> Result<usize, LearningError> {
        let (w, h, k) = (o.grid_size(), o.grid_size(), o.grid_size());
        let (x, y) = (a.x, a.y);
        actions(o).id(&identity(w, h, k), x, y)
    }
    /// Use observation `o` dimensions to turn number `id` into a new action value.
    /// An ID outside the action set is an error; an occupied cell's ID can still be decoded.
    fn action_from_id(
        &self,
        o: &TicTacToeObservation,
        id: usize,
    ) -> Result<PlaceMark, LearningError> {
        let (w, h, k) = (o.grid_size(), o.grid_size(), o.grid_size());
        let (x, y) = actions(o).coordinates(&identity(w, h, k), id)?;
        Ok(PlaceMark { x, y })
    }
}
