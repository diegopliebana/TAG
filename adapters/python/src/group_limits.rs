//! Preflight grouped output sizes and replies before any game accepts scores.
use crate::engine::{Engine, Error};
use tag_rust::{
    agents::{EvaluatorIdentity, OslaPythonError},
    session::EvaluationKey,
};

impl Engine {
    pub(super) fn pending(&self) -> bool {
        match self {
            Self::TicTacToe(g) => g.session.has_pending(),
            Self::Connect4(g) => g.session.has_pending(),
            Self::CantStop(g) => g.session.has_pending(),
            Self::LoveLetter(g) => g.session.has_pending(),
        }
    }

    pub(super) fn validate_reply(
        &self,
        key: &EvaluationKey,
        evaluator: &EvaluatorIdentity,
        ids: &[usize],
        values: &[f64],
    ) -> Result<(), Error> {
        match self {
            Self::TicTacToe(g) => g.session.validate_response(key, evaluator, ids, values)?,
            Self::Connect4(g) => g.session.validate_response(key, evaluator, ids, values)?,
            Self::CantStop(_) => return Err(crate::engine::unsupported("external evaluation")),
            Self::LoveLetter(_) => {
                return Err(crate::engine::unsupported_game(
                    "Love Letter",
                    "external evaluation",
                ));
            }
        }
        Ok(())
    }

    /// Upper bound for the largest external chunk any configured seat can return.
    /// The byte count covers returned numeric arrays, including row ownership.
    pub(super) fn group_capacity(&self) -> Result<(usize, usize), Error> {
        let spec = match self {
            Self::TicTacToe(game) => &game.description,
            Self::Connect4(game) => &game.description,
            Self::CantStop(game) => &game.description,
            Self::LoveLetter(game) => &game.description,
        };
        let mut rows_max = 0;
        let mut bytes_max = 0;
        for seat in 0..spec.identity.player_count {
            if self.execution_mode(seat)? != "external" {
                continue;
            }
            let Some(checkpoint) = self.checkpoint(seat)? else {
                continue;
            };
            let config = checkpoint.config;
            let rows = config
                .limits
                .max_rows
                .min(config.limits.max_candidates)
                .min(spec.actions.size);
            // IDs, next players, progress, observers, game indices, and results.
            let mut per_row = 64usize
                .checked_add(spec.identity.player_count)
                .ok_or(OslaPythonError::Overflow)?;
            for name in &config.payloads {
                let tensor = spec
                    .payloads
                    .iter()
                    .find(|p| p.name() == name)
                    .and_then(|p| p.tensor())
                    .ok_or_else(|| Error::Invalid(format!("no numeric producer for {name}")))?;
                let elements = tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or(OslaPythonError::Overflow)?;
                per_row = per_row
                    .checked_add(elements.checked_mul(4).ok_or(OslaPythonError::Overflow)?)
                    .ok_or(OslaPythonError::Overflow)?;
            }
            rows_max = rows_max.max(rows);
            bytes_max = bytes_max.max(rows.checked_mul(per_row).ok_or(OslaPythonError::Overflow)?);
        }
        Ok((rows_max, bytes_max))
    }
}
