//! Public events and separately owned, audience-specific private experiences.
use super::{Card, LoveLetterAction};
use crate::core::PlayerId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "replay",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum DrawSource {
    Deal,
    Pile,
    Removed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "replay",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum RoundEnd {
    LastPlayer,
    DeckEmpty,
}

/// All fields in these events are public. The order identifies public chronology;
/// it does not expose the identity of any hidden physical card.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "replay",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum PublicEvent {
    RoundStarted {
        round: usize,
        starter: PlayerId,
    },
    Received {
        player: PlayerId,
        source: DrawSource,
    },
    Played {
        actor: PlayerId,
        action: LoveLetterAction,
    },
    Discarded {
        player: PlayerId,
        card: Card,
    },
    GuardResult {
        actor: PlayerId,
        target: PlayerId,
        guess: Card,
        correct: bool,
    },
    Compared {
        actor: PlayerId,
        target: PlayerId,
        eliminated: Option<PlayerId>,
    },
    Exchanged {
        actor: PlayerId,
        target: PlayerId,
    },
    Protection {
        player: PlayerId,
        enabled: bool,
    },
    Eliminated {
        player: PlayerId,
    },
    RoundEnded {
        winners: Vec<PlayerId>,
        reason: RoundEnd,
        revealed_hands: Vec<(PlayerId, Card)>,
    },
    GameEnded {
        winners: Vec<PlayerId>,
    },
}
/// Payloads supplied only to the player who personally experienced this information.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "replay",
    serde(rename_all = "snake_case", deny_unknown_fields)
)]
pub enum PrivateInformation {
    Received {
        card: Card,
        source: DrawSource,
    },
    Priest {
        target: PlayerId,
        card: Card,
    },
    Comparison {
        opponent: PlayerId,
        own_card: Card,
        other_card: Card,
    },
    Exchange {
        other: PlayerId,
        given: Card,
        received: Card,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct PrivateEvent {
    /// Index into that round's public transcript; private event counts are not public.
    pub event: usize,
    pub information: PrivateInformation,
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct RoundSummary {
    pub round: usize,
    pub reason: RoundEnd,
    pub winners: Vec<PlayerId>,
    pub revealed_hands: Vec<(PlayerId, Card)>,
    pub tokens: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(deny_unknown_fields))]
pub struct PublicRound {
    pub summary: RoundSummary,
    pub events: Vec<PublicEvent>,
}
