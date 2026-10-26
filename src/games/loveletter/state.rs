//! Exact state owns hidden facts; observations project only permitted information.
use super::*;
use crate::core::{
    Decision, PlayerId, PlayerResult,
    checkpoint::Checkpoint,
    progress::TurnCounters,
    random::{ChanceRng, ChanceRngSnapshot},
};
use std::collections::VecDeque;

/// Owned public facts, shared by every observer at the same decision point.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct LoveLetterPublicState {
    pub rules: LoveLetterRules,
    pub current_player: PlayerId,
    pub last_actor: Option<PlayerId>,
    pub round: usize,
    pub tokens: Vec<u32>,
    pub active: Vec<bool>,
    pub protected: Vec<bool>,
    pub reserve: Vec<Card>,
    pub discards: Vec<Vec<Card>>,
    pub draw_count: usize,
    pub removed_count: usize,
    #[cfg_attr(feature = "replay", serde(with = "super::persistence::counters"))]
    pub counters: TurnCounters,
    pub outcome: LoveLetterOutcome,
    pub events: Vec<PublicEvent>,
    pub last_round: Option<PublicRound>,
}
impl LoveLetterPublicState {
    pub fn player_count(&self) -> usize {
        self.tokens.len()
    }
    pub fn next_player(&self) -> Decision {
        match self.outcome {
            LoveLetterOutcome::Ongoing => Decision::Player(self.current_player),
            _ => Decision::Finished,
        }
    }
    pub fn player_result(&self, p: PlayerId) -> Result<PlayerResult, LoveLetterError> {
        self.validate_player(p)?;
        Ok(match &self.outcome {
            LoveLetterOutcome::Ongoing => PlayerResult::Ongoing,
            LoveLetterOutcome::Won(w) => {
                if w.contains(&p) {
                    PlayerResult::Win
                } else {
                    PlayerResult::Loss
                }
            }
        })
    }
    pub fn player_score(&self, p: PlayerId) -> Result<f64, LoveLetterError> {
        self.validate_player(p)?;
        Ok(f64::from(self.tokens[p.index()]))
    }
    pub(super) fn validate_player(&self, p: PlayerId) -> Result<(), LoveLetterError> {
        if p.index() < self.player_count() {
            Ok(())
        } else {
            Err(LoveLetterError::InvalidPlayer(p))
        }
    }
}
/// Privileged rule state. Clone preserves the complete independent continuation.
/// Research callers may inspect it explicitly; acting agents receive observations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoveLetterState {
    pub(super) public: LoveLetterPublicState,
    pub(super) hands: Vec<Vec<Card>>,
    pub(super) pile: VecDeque<Card>,
    pub(super) removed: Option<Card>,
    pub(super) random: ChanceRng,
    pub(super) private: Vec<Vec<PrivateEvent>>,
    pub(super) completed_private: Vec<Vec<PrivateEvent>>,
}
impl LoveLetterState {
    pub fn public(&self) -> &LoveLetterPublicState {
        &self.public
    }
    /// Full-state audit access. Never pass these values to a real acting agent.
    pub fn privileged_hand(&self, p: PlayerId) -> Result<&[Card], LoveLetterError> {
        self.public.validate_player(p)?;
        Ok(&self.hands[p.index()])
    }
    pub fn privileged_draw_pile(&self) -> &VecDeque<Card> {
        &self.pile
    }
    pub fn privileged_removed_card(&self) -> Option<Card> {
        self.removed
    }
    pub fn chance_snapshot(&self) -> ChanceRngSnapshot {
        self.random.snapshot()
    }
}
impl Checkpoint for LoveLetterState {
    type Snapshot = Self;
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) -> Self {
        self.clone()
    }
    fn from_snapshot(s: &Self) -> Result<Self, Self::RestoreError> {
        Ok(s.clone())
    }
}

/// An observer's independently owned public facts, own cards and private experiences.
/// Unknown cards, real-game seeds and RNG checkpoints are absent from this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoveLetterObservation {
    pub(super) observer: PlayerId,
    pub(super) public: LoveLetterPublicState,
    pub(super) hand: Vec<Card>,
    pub(super) private: Vec<PrivateEvent>,
    pub(super) last_round_private: Vec<PrivateEvent>,
}
impl LoveLetterObservation {
    pub fn observer(&self) -> PlayerId {
        self.observer
    }
    pub fn public(&self) -> &LoveLetterPublicState {
        &self.public
    }
    pub fn hand(&self) -> &[Card] {
        &self.hand
    }
    pub fn private_events(&self) -> &[PrivateEvent] {
        &self.private
    }
    /// Last completed round's own experiences, kept separately from current inference.
    pub fn last_round_private_events(&self) -> &[PrivateEvent] {
        &self.last_round_private
    }
}
