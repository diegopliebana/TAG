//! Learning registration delegates shared placement encoding, never full-state reflection.
use super::*;
use crate::core::{ForwardModel, PlayerId};
use crate::learning::*;
/// Describe the game/version and supplied board dimensions/target length.
fn identity(width: usize, height: usize, target: usize) -> EncodingIdentity {
    EncodingIdentity {
        game: "connect4",
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
fn actions(o: &Connect4Observation) -> ActionSpec {
    ActionSpec {
        size: o.width(),
        mapping: ActionMapping::Column,
    }
}
impl LearningGame for Connect4ForwardModel {
    /// Read configuration `p` and return array/action descriptions, using `limits` for maximum sizes.
    /// Invalid settings, size overflow or exceeding the numeric-data limit return an error.
    fn learning_spec(
        &self,
        p: &Connect4Parameters,
        limits: LearningLimits,
    ) -> Result<LearningSpec, LearningError> {
        let (w, h, k) = (p.width, p.height, p.winning_length);
        let spec =
            crate::learning::placement_spec(identity(w, h, k), ActionMapping::Column, limits)?;
        self.init_state(p, 2)
            .map_err(|e| LearningError::Invalid(e.to_string()))?;
        Ok(spec)
    }
    /// Read player observation `o`; `payloads` names the arrays requested and `limits` caps their size.
    /// Return separate arrays, progress and actions, or LearningError explaining why encoding failed.
    fn encode(
        &self,
        o: &Connect4Observation,
        payloads: &[&str],
        limits: LearningLimits,
    ) -> Result<EncodedObservation, LearningError> {
        let (w, h, k) = (o.width(), o.height(), o.winning_length());
        crate::learning::check_placement_size(&identity(w, h, k), limits)?;
        // The shared encoder borrows permitted cells without copying a temporary board.
        let occupant = |index| match o.board()[index] {
            Cell::Empty => None,
            Cell::Piece(p) => Some(p),
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
                mapping: ActionMapping::Column,
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
    fn action_id(&self, o: &Connect4Observation, a: &DropPiece) -> Result<usize, LearningError> {
        let (w, h, k) = (o.width(), o.height(), o.winning_length());
        let (x, y) = (a.column, 0);
        actions(o).id(&identity(w, h, k), x, y)
    }
    /// Use observation `o` dimensions to turn number `id` into a new action value.
    /// An ID outside the action set is an error; an occupied cell's ID can still be decoded.
    fn action_from_id(
        &self,
        o: &Connect4Observation,
        id: usize,
    ) -> Result<DropPiece, LearningError> {
        let (w, h, k) = (o.width(), o.height(), o.winning_length());
        let (x, _) = actions(o).coordinates(&identity(w, h, k), id)?;
        Ok(DropPiece { column: x })
    }
}
