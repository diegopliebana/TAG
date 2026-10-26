//! Optional native state estimates, distinct from score and terminal outcomes.
use super::{COLUMN_COUNT, CantStopError, CantStopPosition, CantStopState};
use crate::core::{PlayerId, PlayerResult, checkpoint::Checkpoint, evaluation::Evaluator};

/// Java baseline: claimed columns divided by the configured winning target.
#[derive(Clone, Copy, Debug, Default)]
pub struct CantStopBaseline;
impl Checkpoint for CantStopBaseline {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) {}
    fn from_snapshot(_: &()) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}

/// Approved heuristic with separate weights for banked and temporary progress.
/// Nonterminal value = (claimed / target + alpha*B + beta*T) / (1+alpha+beta).
/// B and T average fractional progress over the eleven columns; claimed columns
/// are excluded, and T is zero for a player whose turn is not in progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CantStopProgress {
    alpha: f64,
    beta: f64,
}
impl Default for CantStopProgress {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.25,
        }
    }
}
impl CantStopProgress {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, CantStopError> {
        if !alpha.is_finite() || !beta.is_finite() || alpha < 0.0 || beta < 0.0 {
            return Err(CantStopError::Parameters(
                "progress weights must be finite and nonnegative",
            ));
        }
        Ok(Self { alpha, beta })
    }
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }
}
impl Checkpoint for CantStopProgress {
    type Snapshot = (f64, f64);
    type RestoreError = CantStopError;
    fn snapshot(&self) -> Self::Snapshot {
        (self.alpha, self.beta)
    }
    fn from_snapshot(saved: &Self::Snapshot) -> Result<Self, Self::RestoreError> {
        Self::new(saved.0, saved.1)
    }
}

fn terminal(p: &CantStopPosition, player: PlayerId) -> Result<Option<f64>, CantStopError> {
    Ok(match p.player_result(player)? {
        PlayerResult::Win => Some(1.0),
        PlayerResult::Loss => Some(-1.0),
        PlayerResult::Draw => Some(0.0),
        PlayerResult::Ongoing => None,
    })
}
impl CantStopBaseline {
    pub fn evaluate_position(
        &self,
        p: &CantStopPosition,
        player: PlayerId,
    ) -> Result<f64, CantStopError> {
        if let Some(value) = terminal(p, player)? {
            return Ok(value);
        }
        Ok(p.player_score(player)? / p.rules.columns_to_win as f64)
    }
}
impl CantStopProgress {
    pub fn evaluate_position(
        &self,
        p: &CantStopPosition,
        player: PlayerId,
    ) -> Result<f64, CantStopError> {
        if let Some(value) = terminal(p, player)? {
            return Ok(value);
        }
        let mut banked = 0.0;
        let mut temporary = 0.0;
        for n in 0..COLUMN_COUNT {
            if p.owners[n].is_some() {
                continue;
            }
            let permanent = p.permanent[player.index()][n];
            banked += permanent as f64 / p.rules.column_heights[n] as f64;
            if player == p.current_player {
                temporary += (p.temporary[n].unwrap_or(permanent) - permanent) as f64
                    / p.rules.column_heights[n] as f64;
            }
        }
        // Rescale all three coefficients before summing. This is algebraically
        // the approved formula, but remains finite even for very large weights.
        let scale = self.alpha.max(self.beta).max(1.0);
        let (claim_weight, bank_weight, temp_weight) =
            (1.0 / scale, self.alpha / scale, self.beta / scale);
        Ok(
            (claim_weight * (p.player_score(player)? / p.rules.columns_to_win as f64)
                + bank_weight * banked / COLUMN_COUNT as f64
                + temp_weight * temporary / COLUMN_COUNT as f64)
                / (claim_weight + bank_weight + temp_weight),
        )
    }
}
impl Evaluator<CantStopState> for CantStopBaseline {
    type Error = CantStopError;
    fn evaluate(&mut self, state: &CantStopState, player: PlayerId) -> Result<f64, Self::Error> {
        self.evaluate_position(state.position(), player)
    }
}
impl Evaluator<CantStopState> for CantStopProgress {
    type Error = CantStopError;
    fn evaluate(&mut self, state: &CantStopState, player: PlayerId) -> Result<f64, Self::Error> {
        self.evaluate_position(state.position(), player)
    }
}
