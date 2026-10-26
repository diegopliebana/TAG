use super::{Controller, Session, SessionError};
use crate::agents::FirstAvailableAgent;
use crate::core::PlayerId;
use crate::games::tictactoe::{PlaceMark, TicTacToeForwardModel, TicTacToeParameters};

#[test]
fn exhausted_position_version_rejects_before_applying_a_move() {
    let mut session: Session<_, FirstAvailableAgent> = Session::create(
        TicTacToeForwardModel,
        &TicTacToeParameters::default(),
        2,
        vec![Controller::Human, Controller::Human],
    )
    .unwrap();
    session.position_version = u64::MAX;
    let original = session.state().clone();
    assert!(matches!(
        session.submit_human(u64::MAX, PlayerId::new(0), PlaceMark { x: 0, y: 0 }),
        Err(SessionError::PositionVersionExhausted)
    ));
    assert_eq!(session.state(), &original);
    assert_eq!(session.position_version(), u64::MAX);
}
