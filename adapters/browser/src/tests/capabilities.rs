//! A tiny model exercises independent human, state and agent capabilities.
use crate::games::BrowserGame;
use crate::protocol::{BoardView, BridgeError, Command, GameKind, SeatSetup};
use crate::runtime::Runtime;
use tag_rust::agents::{Agent, FirstAvailableAgent};
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::replay::{
    OutcomeData, PersistableGame, PersistenceError, ReplayLimits, export_state, import_state,
};
use tag_rust::session::Controller;

// Clone duplicates stateless rules when creating independent branches.
#[derive(Clone)]
struct TinyGame;

impl ForwardModel for TinyGame {
    type State = bool;
    type Observation = bool;
    type Action = bool;
    type Parameters = ();
    type Error = String;
    fn init_state(&self, _: &(), players: usize) -> Result<bool, String> {
        if players != 2 {
            return Err("expected two seats".into());
        }
        Ok(false)
    }
    fn player_count(&self, _: &bool) -> usize {
        2
    }
    fn next_player(&self, finished: &bool) -> Decision {
        if *finished {
            Decision::Finished
        } else {
            Decision::Player(PlayerId::new(0))
        }
    }
    fn observe(&self, finished: &bool, observer: PlayerId) -> Result<bool, String> {
        if observer.index() >= 2 {
            return Err("unknown observer".into());
        }
        Ok(*finished)
    }
    fn offered_actions(&self, finished: &bool) -> Vec<bool> {
        if *finished { vec![] } else { vec![true] }
    }
    fn available_actions(&self, finished: &bool, actor: PlayerId) -> Result<Vec<bool>, String> {
        if actor.index() >= 2 {
            return Err("unknown actor".into());
        }
        Ok(self.offered_actions(finished))
    }
    fn next(&self, finished: &mut bool, actor: PlayerId, action: &bool) -> Result<(), String> {
        if *finished || actor.index() != 0 || !*action {
            return Err("invalid move".into());
        }
        *finished = true;
        Ok(())
    }
    fn player_result(&self, finished: &bool, player: PlayerId) -> Result<PlayerResult, String> {
        self.observe(finished, player)?;
        Ok(if *finished {
            PlayerResult::Draw
        } else {
            PlayerResult::Ongoing
        })
    }
    fn player_score(&self, finished: &bool, player: PlayerId) -> Result<f64, String> {
        self.observe(finished, player)?;
        Ok(0.0)
    }
}

// Human browser play uses its own controller factory and permitted queries.
impl BrowserGame for TinyGame {
    type Agent = FirstAvailableAgent;
    const KIND: GameKind = GameKind::Tictactoe;
    fn dimensions(_: &()) -> (usize, usize) {
        (1, 1)
    }
    fn board(finished: &bool) -> BoardView {
        BoardView {
            cantstop: None,
            loveletter: None,
            width: 1,
            height: 1,
            cells: vec![finished.then_some(0)],
            current_player: 0,
            counters: [usize::from(*finished), 0, 0, 0],
            outcome: if *finished {
                OutcomeData::Draw.into()
            } else {
                OutcomeData::Ongoing.into()
            },
            winning_cells: vec![],
        }
    }
    fn controller(setup: SeatSetup) -> Result<Controller<Self::Agent>, BridgeError> {
        match setup {
            SeatSetup::Human => Ok(Controller::Human),
            _ => Err(BridgeError::new(
                "invalid_setup",
                "this test supports human input",
            )),
        }
    }
    fn settings(agent: &Self::Agent) -> SeatSetup {
        SeatSetup::FirstAvailable {
            seed: <FirstAvailableAgent as Agent<Self>>::rng(agent)
                .initial_seed()
                .to_string(),
        }
    }
    fn player_values(observation: &bool, player: PlayerId) -> Result<(PlayerResult, f64), String> {
        Ok((
            Self.player_result(observation, player)?,
            Self.player_score(observation, player)?,
        ))
    }
}

#[test]
fn human_runtime_uses_observations_without_public_reconstruction_or_persistence() {
    let mut game = Runtime::create(TinyGame, None, None).unwrap();
    let (view, proposal) = game
        .execute(
            Command::HumanMove {
                observer: None,
                actor: 0,
                action: true.into(),
            },
            Some("0".into()),
        )
        .unwrap();
    assert!(proposal.is_none());
    assert_eq!(view.results, vec![PlayerResult::Draw, PlayerResult::Draw]);
    assert_eq!(view.board.cells, vec![Some(0)]);
}

// State-only saving exercises a separate model with owned stateless rules.
struct StateOnly;
impl ForwardModel for StateOnly {
    type State = bool;
    type Observation = bool;
    type Action = bool;
    type Parameters = ();
    type Error = String;
    fn init_state(&self, p: &(), n: usize) -> Result<bool, String> {
        TinyGame.init_state(p, n)
    }
    fn player_count(&self, s: &bool) -> usize {
        TinyGame.player_count(s)
    }
    fn next_player(&self, s: &bool) -> Decision {
        TinyGame.next_player(s)
    }
    fn observe(&self, s: &bool, p: PlayerId) -> Result<bool, String> {
        TinyGame.observe(s, p)
    }
    fn offered_actions(&self, s: &bool) -> Vec<bool> {
        TinyGame.offered_actions(s)
    }
    fn available_actions(&self, s: &bool, p: PlayerId) -> Result<Vec<bool>, String> {
        TinyGame.available_actions(s, p)
    }
    fn next(&self, s: &mut bool, p: PlayerId, a: &bool) -> Result<(), String> {
        TinyGame.next(s, p, a)
    }
    fn player_result(&self, s: &bool, p: PlayerId) -> Result<PlayerResult, String> {
        TinyGame.player_result(s, p)
    }
    fn player_score(&self, s: &bool, p: PlayerId) -> Result<f64, String> {
        TinyGame.player_score(s, p)
    }
}
impl PersistableGame for StateOnly {
    type SavedState = bool;
    const GAME_ID: &'static str = "test-state";
    const RULES_VERSION: u32 = 1;
    fn storage_units(_: &()) -> Option<usize> {
        Some(1)
    }
    fn validation_work(_: &(), _: &bool) -> Option<usize> {
        Some(1)
    }
    fn state_data(&self, state: &bool) -> bool {
        *state
    }
    fn restore_state(&self, _: &(), data: &bool) -> Result<bool, PersistenceError> {
        Ok(*data)
    }
}

#[test]
fn standalone_saves_support_non_clone_rules_without_evaluators_or_replay_agents() {
    let text = export_state(&StateOnly, &(), &true).unwrap();
    let mut file: serde_json::Value = serde_json::from_str(&text).unwrap();
    file["versions"]["agents"] = 99.into();
    file["versions"]["evaluators"] = 99.into();
    assert_eq!(
        import_state(&file.to_string(), &StateOnly, ReplayLimits::default()).unwrap(),
        ((), true)
    );
    file["versions"]["rules"] = 99.into();
    assert!(import_state(&file.to_string(), &StateOnly, ReplayLimits::default()).is_err());
}
