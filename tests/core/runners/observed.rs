//! Accepted-action callbacks must agree with the live game, including on failure.
use tag_rust::agents::Agent;
use tag_rust::core::random::AgentRng;
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::tictactoe::{
    PlaceMark, TicTacToeForwardModel, TicTacToeObservation, TicTacToeParameters,
};

struct FailsOnSecondDecision {
    rng: AgentRng,
    decisions: usize,
}
impl Agent<TicTacToeForwardModel> for FailsOnSecondDecision {
    type Error = std::convert::Infallible;
    fn rng(&self) -> &AgentRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut AgentRng {
        &mut self.rng
    }
    fn choose(
        &mut self,
        _: &TicTacToeForwardModel,
        _: PlayerId,
        _: &TicTacToeObservation,
        _: &[PlaceMark],
    ) -> Result<usize, Self::Error> {
        self.decisions += 1;
        Ok(if self.decisions == 1 { 0 } else { usize::MAX })
    }
}

#[test]
fn observer_receives_only_accepted_moves_and_their_updated_states() {
    let model = TicTacToeForwardModel;
    let parameters = TicTacToeParameters::default();
    let mut state = model.init_state(&parameters, 2).unwrap();
    let mut independently_replayed = state.clone();
    let mut agents = [0, 1].map(|seed| FailsOnSecondDecision {
        rng: AgentRng::new(seed),
        decisions: 0,
    });
    let mut observed = 0;
    let result = tag_rust::runner::run_game_observed(
        &model,
        &mut state,
        &mut agents,
        |rules, updated, actor, action, _| {
            rules
                .next(&mut independently_replayed, actor, &action)
                .unwrap();
            assert_eq!(&independently_replayed, updated);
            observed += 1;
        },
    );
    assert!(matches!(
        result,
        Err(tag_rust::runner::RunnerError::InvalidSelection { .. })
    ));
    assert_eq!(observed, 2);
    assert_eq!(state, independently_replayed);
}
