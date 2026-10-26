use super::{BranchOrigin, Command, Runtime};
use serde_json::json;
use tag_rust::games::tictactoe::TicTacToeForwardModel;

#[test]
fn browser_branch_depth_and_count_limits_reject_before_mutation() {
    let mut game = Runtime::create(TicTacToeForwardModel, None, None).unwrap();
    for _ in 1..256 {
        let version = game.view(None).unwrap().position_version;
        game.execute(Command::Reset, Some(version)).unwrap();
    }
    let before = game.view(None).unwrap();
    assert!(matches!(
        before.branches.last().unwrap().origin,
        BranchOrigin::Reset
    ));
    let error = game
        .execute(Command::Reset, Some(before.position_version.clone()))
        .unwrap_err();
    assert_eq!(error.code, "resource_limit");
    let after = game.view(None).unwrap();
    assert_eq!(after.position_version, before.position_version);
    assert_eq!(after.branches.len(), 256);
    let mut nested = Runtime::create(TicTacToeForwardModel, None, None).unwrap();
    for _ in 1..64 {
        let view = nested.view(None).unwrap();
        nested
            .execute(
                Command::Fork {
                    branch: view.branch,
                    cursor: 0,
                    replacements: vec![None, None],
                },
                Some(view.position_version),
            )
            .unwrap();
    }
    let before = nested.view(None).unwrap();
    let error = nested
        .execute(
            Command::Fork {
                branch: before.branch,
                cursor: 0,
                replacements: vec![None, None],
            },
            Some(before.position_version.clone()),
        )
        .unwrap_err();
    assert_eq!(error.code, "resource_limit");
    assert_eq!(
        nested.view(None).unwrap().position_version,
        before.position_version
    );
    assert_eq!(nested.game.branch_count(), 64);
}
#[test]
fn stored_move_limit_and_oversized_osla_replacement_are_atomic() {
    let mut game = Runtime::create(TicTacToeForwardModel, None, None).unwrap();
    game.stored_moves = 100_000;
    let error = game
        .execute(
            Command::HumanMove {
                observer: None,
                actor: 0,
                action: json!({"x":0,"y":0}),
            },
            Some("0".into()),
        )
        .unwrap_err();
    assert_eq!(error.code, "resource_limit");
    assert_eq!(game.view(None).unwrap().cursor, 0);
    let mut large = Runtime::create(
        TicTacToeForwardModel,
        Some(json!({"grid_size":17,"limits":{"max_cells":4096}})),
        None,
    )
    .unwrap();
    let error = large
        .execute(
            Command::Fork {
                branch: 0,
                cursor: 0,
                replacements: vec![Some(super::SeatSetup::Osla { seed: "7".into() }), None],
            },
            Some("0".into()),
        )
        .unwrap_err();
    assert_eq!(error.code, "resource_limit");
    assert_eq!(large.game.branch_count(), 1);
}
