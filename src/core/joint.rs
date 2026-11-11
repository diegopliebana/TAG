//! Complete decisions whose actors choose from one unchanged state.
use super::{ForwardModel, PlayerId};
use std::fmt;

/// A seat-ordered choice. The action stays owned by its caller.
pub type Choice<A> = (PlayerId, A);

/// Invalid external input is rejected before the forward model changes state.
#[derive(Debug, PartialEq, Eq)]
pub enum JointError<E> {
    Invalid(String),
    Rule(E),
}

impl<E: fmt::Display> fmt::Display for JointError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid joint decision: {message}"),
            Self::Rule(error) => error.fmt(f),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for JointError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rule(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

/// Validate seat identity, order and all offered actions against the same state.
/// Observation and action enumeration must be read-only rule queries.
pub fn validate<F: ForwardModel>(
    model: &F,
    state: &F::State,
    choices: &[Choice<F::Action>],
) -> Result<(), JointError<F::Error>>
where
    F::Action: PartialEq,
{
    let seats = model.decision_players(state);
    if seats.is_empty() {
        return Err(JointError::Invalid("game has finished".into()));
    }
    if seats.len() != choices.len() {
        return Err(JointError::Invalid("incomplete decision".into()));
    }
    let count = model.player_count(state);
    if seats.iter().any(|seat| seat.index() >= count)
        || seats
            .iter()
            .enumerate()
            .any(|(i, seat)| seats[..i].contains(seat))
    {
        return Err(JointError::Invalid("invalid decision seats".into()));
    }
    for (seat, (submitted, action)) in seats.iter().zip(choices) {
        if seat != submitted {
            return Err(JointError::Invalid(
                "choices must follow decision seat order".into(),
            ));
        }
        let offered = model
            .available_actions(state, *seat)
            .map_err(JointError::Rule)?;
        if offered.is_empty() || !offered.contains(action) {
            return Err(JointError::Invalid(format!(
                "unavailable action for seat {}",
                seat.index()
            )));
        }
    }
    Ok(())
}

/// Apply one complete decision. A rule execution failure aborts the caller's
/// simulation; as with `next`, its state may contain partial internal effects.
pub fn apply<F: ForwardModel>(
    model: &F,
    state: &mut F::State,
    choices: &[Choice<F::Action>],
) -> Result<(), JointError<F::Error>>
where
    F::Action: PartialEq,
{
    validate(model, state, choices)?;
    model.resolve_decision(state, choices)
}

/// Fill the other seats' choices from observations of a simulated state.
/// `pick` selects a list index for each non-focal actor; callers own its RNG
/// and declare the policy. No real pending choice is supplied to this function.
pub fn complete_with<F: ForwardModel>(
    model: &F,
    state: &F::State,
    focal: PlayerId,
    action: &F::Action,
    mut pick: impl FnMut(usize) -> usize,
) -> Result<Vec<Choice<F::Action>>, JointError<F::Error>>
where
    F::Action: Clone + PartialEq,
{
    let seats = model.decision_players(state);
    if !seats.contains(&focal) {
        return Err(JointError::Invalid("focal player is not choosing".into()));
    }
    let mut choices = Vec::with_capacity(seats.len());
    for seat in seats {
        let chosen = if seat == focal {
            action.clone()
        } else {
            let view = model.observe(state, seat).map_err(JointError::Rule)?;
            let actions = model.offered_actions(&view);
            let legal = model
                .available_actions(state, seat)
                .map_err(JointError::Rule)?;
            if actions.is_empty() || actions != legal {
                return Err(JointError::Invalid(format!(
                    "observation and state offer different actions for seat {}",
                    seat.index()
                )));
            }
            actions.get(pick(actions.len())).cloned().ok_or_else(|| {
                JointError::Invalid("simulation policy selected an invalid action".into())
            })?
        };
        choices.push((seat, chosen));
    }
    validate(model, state, &choices)?;
    Ok(choices)
}
