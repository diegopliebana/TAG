//! Seeded controller streams and failures through the shared runner.
use std::error::Error;
use std::fmt;
use tag_rust::agents::{Agent, EmptyActionList, FirstAvailableAgent};
use tag_rust::core::random::AgentRng;
use tag_rust::core::{ForwardModel, PlayerId, PublicInformationModel};
use tag_rust::games::tictactoe::{
    PlaceMark, TicTacToeForwardModel, TicTacToeObservation, TicTacToeParameters,
};
use tag_rust::runner::{RunnerError, run_game};

#[test]
fn seeded_streams_are_reproducible_independent_and_bounded() {
    let mut first = AgentRng::new(42);
    let mut second = AgentRng::new(42);
    let mut other = AgentRng::new(43);
    assert_eq!(first.initial_seed(), 42);
    assert_eq!(other.initial_seed(), 43);
    let mut differs = false;
    for _ in 0..64 {
        let a = first.index(1_000_000).unwrap();
        assert_eq!(a, second.index(1_000_000).unwrap());
        differs |= a != other.index(1_000_000).unwrap();
    }
    assert!(differs);
    let mut independent = AgentRng::new(99);
    for _ in 0..100 {
        first.index(7);
    }
    let mut control = AgentRng::new(99);
    for count in [2, 3, 7, 100, usize::MAX / 2 + 2, usize::MAX] {
        for _ in 0..32 {
            let choice = independent.index(count).unwrap();
            assert!(choice < count);
            assert_eq!(choice, control.index(count).unwrap());
        }
    }
}

#[test]
fn empty_and_single_choice_do_not_advance_rng() {
    let mut rng = AgentRng::new(100);
    let mut control = AgentRng::new(100);
    for _ in 0..10 {
        assert_eq!(rng.index(0), None);
        assert_eq!(rng.index(1), Some(0));
    }
    for _ in 0..32 {
        assert_eq!(rng.index(17), control.index(17));
    }
}

#[test]
fn baseline_owns_its_seed_but_does_not_draw_when_choosing() {
    let model = TicTacToeForwardModel;
    let state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let actor = PlayerId::new(0);
    let view = model.observe(&state, actor).unwrap();
    let actions = model.offered_actions(&view);
    let mut agent = FirstAvailableAgent::new(123);
    assert_eq!(
        <FirstAvailableAgent as Agent<TicTacToeForwardModel>>::rng(&agent).initial_seed(),
        123
    );
    assert_eq!(agent.choose(&model, actor, &view, &actions), Ok(0));
    assert_eq!(
        agent.choose(&model, actor, &view, &[]),
        Err(EmptyActionList)
    );
    let mut control = AgentRng::new(123);
    let rng = <FirstAvailableAgent as Agent<TicTacToeForwardModel>>::rng_mut(&mut agent);
    for _ in 0..32 {
        assert_eq!(rng.index(37), control.index(37));
    }
}

#[derive(Debug, PartialEq, Eq)] // Equality: compare typed failures.
struct DeliberateFailure;
impl fmt::Display for DeliberateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "deliberate controller failure")
    }
}
impl Error for DeliberateFailure {}
struct FailingOnce {
    rng: AgentRng,
    fail: bool,
    calls: usize,
}
impl Agent<TicTacToeForwardModel> for FailingOnce {
    type Error = DeliberateFailure;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,
        model: &TicTacToeForwardModel,
        actor: PlayerId,
        view: &TicTacToeObservation,
        actions: &[PlaceMark],
    ) -> Result<usize, DeliberateFailure> {
        if actions.is_empty() {
            return Err(DeliberateFailure);
        }
        // Rules and the view containing only information the player is allowed to see are accessible.
        assert_eq!(view.observer(), actor);
        let permitted = model.simulation_state(view).unwrap();
        assert_eq!(permitted.current_player(), actor);
        assert_eq!(model.offered_actions(view).len(), actions.len());
        self.calls += 1;
        self.rng.index(17); // Intentional RNG progress before reporting a failure.
        if self.fail {
            self.fail = false;
            Err(DeliberateFailure)
        } else {
            Ok(0)
        }
    }
}

#[test]
fn agent_failure_preserves_prior_moves_and_rng_progress_then_resumes() {
    let model = TicTacToeForwardModel;
    let mut state = model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    let mut agents = [
        FailingOnce {
            rng: AgentRng::new(10),
            fail: false,
            calls: 0,
        },
        FailingOnce {
            rng: AgentRng::new(20),
            fail: true,
            calls: 0,
        },
    ];
    let error = run_game(&model, &mut state, &mut agents).unwrap_err();
    assert_eq!(
        error,
        RunnerError::Agent {
            player: PlayerId::new(1),
            error: DeliberateFailure
        }
    );
    assert_eq!(
        error.source().unwrap().to_string(),
        "deliberate controller failure"
    );
    assert_eq!(state.move_count(), 1);
    assert_eq!(state.current_player(), PlayerId::new(1));
    assert_eq!(agents[1].calls, 1);
    let mut control = AgentRng::new(20);
    control.index(17); // Retain the request consumed by the failed choice.
    assert_eq!(agents[1].rng.index(17), control.index(17));
    run_game(&model, &mut state, &mut agents).unwrap();
    assert_eq!(state.move_count(), 7);
    assert_eq!(agents[1].calls, 4); // Three accepted seat-1 moves plus its earlier failure.
}

#[test]
fn seeded_sequence_matches_the_pinned_generator_reference() {
    use rand_chacha::ChaCha8Rng;
    use rand_chacha::rand_core::{RngCore, SeedableRng};
    let mut rng = AgentRng::new(0x1234_5678);
    let mut reference = ChaCha8Rng::seed_from_u64(0x1234_5678);
    // A power-of-two bound uses the generator's low bits directly.
    for _ in 0..64 {
        assert_eq!(
            rng.index(65536),
            Some((reference.next_u64() & 65535) as usize)
        );
    }
}
