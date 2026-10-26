//! Card ranks, chosen actions and checked setup parameters.
use crate::core::PlayerId;
use std::fmt;

/// Card types present in the rulebook's classic option. Ranks 0 and 6 are absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(rename_all = "snake_case")
)]
pub enum Card {
    Guard = 1,
    Priest = 2,
    Baron = 3,
    Handmaid = 4,
    Prince = 5,
    King = 7,
    Countess = 8,
    Princess = 9,
}
impl Card {
    pub const ALL: [Self; 8] = [
        Self::Guard,
        Self::Priest,
        Self::Baron,
        Self::Handmaid,
        Self::Prince,
        Self::King,
        Self::Countess,
        Self::Princess,
    ];
    pub const DECK: [Self; 16] = [
        Self::Guard,
        Self::Guard,
        Self::Guard,
        Self::Guard,
        Self::Guard,
        Self::Priest,
        Self::Priest,
        Self::Baron,
        Self::Baron,
        Self::Handmaid,
        Self::Handmaid,
        Self::Prince,
        Self::Prince,
        Self::King,
        Self::Countess,
        Self::Princess,
    ];
    pub const fn rank(self) -> u8 {
        self as u8
    }
}
impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Complete player intent. Revealed cards and effect outcomes never enter this value.
/// A play without a target represents self effects or an explicit no-effect play.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum LoveLetterAction {
    Play(Card),
    Target { card: Card, target: PlayerId },
    Guess { target: PlayerId, card: Card },
}
impl LoveLetterAction {
    pub fn played_card(self) -> Card {
        match self {
            Self::Play(card) | Self::Target { card, .. } => card,
            Self::Guess { .. } => Card::Guard,
        }
    }
}

/// One target for each supported player count: entries correspond to 2, 3 and 4.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(default, deny_unknown_fields)
)]
pub struct LoveLetterRules {
    pub winning_tokens: [u32; 3],
}
impl Default for LoveLetterRules {
    fn default() -> Self {
        Self {
            winning_tokens: [6, 5, 4],
        }
    }
}
/// Low-level setup receives a resolved seed. Applications resolve automatic seeds.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(
    any(feature = "replay", feature = "mcts-json"),
    serde(default, deny_unknown_fields)
)]
pub struct LoveLetterParameters {
    pub rules: LoveLetterRules,
    pub seed: u64,
    /// Explicit digital replacement for the rulebook's social first-player choice.
    pub first_player: PlayerId,
}
impl Default for LoveLetterParameters {
    fn default() -> Self {
        Self {
            rules: Default::default(),
            seed: 0,
            first_player: PlayerId::new(0),
        }
    }
}

/// Shared victory identifies every winner. Each receives a Win, rather than a Draw.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "snake_case"))]
pub enum LoveLetterOutcome {
    Ongoing,
    Won(Vec<PlayerId>),
}
impl LoveLetterOutcome {
    pub fn winners(&self) -> &[PlayerId] {
        match self {
            Self::Ongoing => &[],
            Self::Won(players) => players,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoveLetterError {
    UnsupportedPlayerCount(usize),
    InvalidPlayer(PlayerId),
    InvalidParameters(&'static str),
    InvalidDeck,
    SamplingBudgetExceeded,
    WrongPlayer {
        expected: PlayerId,
        actual: PlayerId,
    },
    IllegalAction,
    Finished,
    CounterOverflow,
    InvalidState(&'static str),
}
impl fmt::Display for LoveLetterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlayerCount(n) => {
                write!(f, "Love Letter requires 2–4 players; received {n}")
            }
            Self::InvalidPlayer(p) => write!(f, "invalid player {}", p.index()),
            Self::InvalidParameters(s) => write!(f, "invalid Love Letter parameters: {s}"),
            Self::InvalidDeck => write!(
                f,
                "controlled deck must contain the classic sixteen-card multiset"
            ),
            Self::SamplingBudgetExceeded => write!(
                f,
                "Love Letter simulation sampling exhausted its work budget"
            ),
            Self::WrongPlayer { expected, actual } => write!(
                f,
                "player {} must act, received {}",
                expected.index(),
                actual.index()
            ),
            Self::IllegalAction => write!(f, "action was not offered"),
            Self::Finished => write!(f, "game is finished"),
            Self::CounterOverflow => write!(f, "Love Letter counter overflow"),
            Self::InvalidState(s) => write!(f, "inconsistent Love Letter state: {s}"),
        }
    }
}
impl std::error::Error for LoveLetterError {}
