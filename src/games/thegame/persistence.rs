//! Privileged state files retain hidden cards and exact chance continuation.
use super::*;
use crate::core::evaluation::Evaluator;
use crate::replay::{PersistableGame, PersistenceError, ReplayGame, RngData};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedState {
    public: PublicState,
    hands: Vec<Vec<u8>>,
    deck: Vec<u8>,
    pending_messages: Vec<Message>,
    chance: RngData,
}

fn invalid(message: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Invalid(message.to_string())
}

impl PersistableGame for TheGameForwardModel {
    type SavedState = SavedState;
    const GAME_ID: &'static str = "thegame";
    const RULES_VERSION: u32 = 1;

    fn storage_units(_: &Parameters) -> Option<usize> {
        Some(CARD_COUNT)
    }
    fn validation_work(_: &Parameters, data: &SavedState) -> Option<usize> {
        data.public
            .placements
            .len()
            .checked_add(data.public.messages.len().checked_mul(5)?)?
            .checked_mul(32)?
            .checked_add(1024)
    }
    fn state_data(&self, state: &State) -> SavedState {
        SavedState {
            public: state.public.clone(),
            hands: state.hands.clone(),
            deck: state.deck.iter().copied().collect(),
            pending_messages: state.pending_messages.clone(),
            chance: RngData::from_snapshot(state.random.snapshot().generator),
        }
    }
    fn restore_state(&self, p: &Parameters, data: &SavedState) -> Result<State, PersistenceError> {
        let n = data.hands.len();
        let public = &data.public;
        if !(1..=5).contains(&n)
            || public.first_player != p.first_player
            || public.communication != p.communication
            || public.first_player.index() >= n
            || public.active_player.index() >= n
            || public.hand_counts.len() != n
            || data.deck.len() > CARD_COUNT
            || data.pending_messages.len() > n
            || public.placements.len() > CARD_COUNT
            || public.messages.len() > CARD_COUNT
            || public.turns_completed > CARD_COUNT
            || public.played_this_turn > starting_hand_size(n)
            || data.hands.iter().any(|h| h.len() > starting_hand_size(n))
        {
            return Err(invalid("invalid The Game state dimensions or rules"));
        }
        let mut inventory = data.deck.clone();
        inventory.extend(data.hands.iter().flatten().copied());
        inventory.extend(public.placements.iter().map(|item| item.card));
        inventory.sort_unstable();
        if inventory != (2..=99).collect::<Vec<_>>()
            || public.hand_counts != data.hands.iter().map(Vec::len).collect::<Vec<_>>()
            || public.draw_count != data.deck.len()
        {
            return Err(invalid("The Game card inventory or counts disagree"));
        }
        let mut tops = [1, 1, 100, 100];
        for placement in &public.placements {
            if placement.player.index() >= n
                || !placement
                    .pile
                    .permits(tops[placement.pile.index()], placement.card)
            {
                return Err(invalid("invalid The Game public placement history"));
            }
            tops[placement.pile.index()] = placement.card;
        }
        if tops != public.pile_tops {
            return Err(invalid(
                "The Game pile tops disagree with placement history",
            ));
        }
        if public
            .messages
            .iter()
            .any(|round| round.active_player.index() >= n || round.messages.len() != n)
            || (n == 1 && !public.messages.is_empty())
            || (public.communication == CommunicationMode::Silent && !public.messages.is_empty())
        {
            return Err(invalid("invalid The Game communication history"));
        }
        let chance = data.chance.snapshot()?;
        let initial = self
            .init_state(p, n)
            .map_err(invalid)?
            .random
            .snapshot()
            .generator;
        if chance != initial {
            return Err(invalid("The Game shuffle stream disagrees with setup seed"));
        }
        let random =
            ChanceRng::from_snapshot(&ChanceRngSnapshot { generator: chance }).map_err(invalid)?;
        let state = State {
            public: public.clone(),
            hands: data.hands.clone(),
            deck: VecDeque::from(data.deck.clone()),
            pending_messages: data.pending_messages.clone(),
            random,
        };
        match public.phase {
            Phase::Communicate { speaker } if public.outcome == TeamOutcome::Ongoing => {
                if n == 1
                    || public.communication != CommunicationMode::LimitedChoices
                    || speaker.index() != data.pending_messages.len()
                    || speaker.index() >= n
                    || public.played_this_turn != 0
                    || !can_finish_minimum(
                        &state.hands[public.active_player.index()],
                        public.pile_tops,
                        minimum(&state),
                    )
                {
                    return Err(invalid("invalid The Game communication phase"));
                }
            }
            Phase::Play if public.outcome == TeamOutcome::Ongoing => {
                if !data.pending_messages.is_empty()
                    || !can_finish_minimum(
                        &state.hands[public.active_player.index()],
                        public.pile_tops,
                        minimum(&state).saturating_sub(public.played_this_turn),
                    )
                {
                    return Err(invalid("invalid The Game card-play phase"));
                }
            }
            Phase::Finished if public.outcome != TeamOutcome::Ongoing => {
                if !data.pending_messages.is_empty()
                    || (public.outcome == TeamOutcome::Won) != (state.cards_remaining() == 0)
                    || (public.outcome == TeamOutcome::Lost
                        && can_finish_minimum(
                            &state.hands[public.active_player.index()],
                            public.pile_tops,
                            minimum(&state).saturating_sub(public.played_this_turn),
                        ))
                {
                    return Err(invalid("invalid The Game terminal result"));
                }
            }
            _ => return Err(invalid("The Game phase and result disagree")),
        }
        Ok(state)
    }
}

#[derive(Default)]
pub struct ReplayEvaluator;
impl Evaluator<State> for ReplayEvaluator {
    type Error = Error;
    fn evaluate(&mut self, state: &State, player: PlayerId) -> Result<f64, Error> {
        TheGameForwardModel.player_score(state, player)
    }
}
impl Checkpoint for ReplayEvaluator {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) {}
    fn from_snapshot(_: &()) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}
impl ReplayGame for TheGameForwardModel {
    type SavedEvaluator = ReplayEvaluator;
    fn verification_work(_: &Parameters, _: bool) -> Option<usize> {
        Some(8192)
    }
}
