//! Compare Rust replays with saved Java results.
use std::collections::HashSet;
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::tictactoe::{
    Cell, PlaceMark, TicTacToeForwardModel, TicTacToeParameters, TicTacToeState,
};

fn result_name(result: PlayerResult) -> &'static str {
    // Copied Rust result in; returns a shared static Java enum name for comparison.
    match result {
        PlayerResult::Ongoing => "GAME_ONGOING",
        PlayerResult::Win => "WIN_GAME",
        PlayerResult::Draw => "DRAW_GAME",
        PlayerResult::Loss => "LOSE_GAME",
    }
}

fn signature(model: &TicTacToeForwardModel, state: &TicTacToeState) -> String {
    // Shared rules/state borrows; returns an owned comparison string.
    let mut board = String::new();
    for cell in state.board() {
        board.push(match cell {
            Cell::Empty => '.',
            Cell::X => 'x',
            Cell::O => 'o',
        });
    }
    let next = match model.next_player(state) {
        Decision::Player(player) => player.index() as i32,
        Decision::Finished => -1,
    };
    let status = match state.status() {
        SingleWinnerOutcome::Ongoing => "GAME_ONGOING",
        SingleWinnerOutcome::Won(_) => "GAME_END",
        SingleWinnerOutcome::Draw => "DRAW_GAME",
    };
    let mut actions = Vec::new();
    for action in model
        .available_actions(state, state.current_player())
        .unwrap()
    {
        actions.push((action.x * 3 + action.y).to_string());
    }
    format!(
        "{board}|{}|{next}|{}|{}|{}|{}|{status}|{}|{}|{:.1}|{:.1}|{}",
        state.current_player().index(),
        state.move_count(),
        state.completed_turn_count(),
        state.round_count(),
        state.first_player().index(),
        result_name(model.player_result(state, PlayerId::new(0)).unwrap()),
        result_name(model.player_result(state, PlayerId::new(1)).unwrap()),
        model.player_score(state, PlayerId::new(0)).unwrap(),
        model.player_score(state, PlayerId::new(1)).unwrap(),
        actions.join(",")
    )
}

fn compare_fields(
    actual: &str,   // Shared actual signature borrow.
    expected: &str, // Shared recorded signature borrow.
    sequence: &str, // Shared recorded move sequence for diagnostics.
    context: &str,  // Shared label identifies full state or player copy.
) {
    const NAMES: [&str; 13] = [
        "board",
        "historical current player",
        "next player (-1 means finished)",
        "move count",
        "completed-turn count",
        "round count",
        "first player",
        "game status",
        "player 0 result",
        "player 1 result",
        "player 0 score",
        "player 1 score",
        "ordered available actions",
    ];
    let actual_fields: Vec<_> = actual.split('|').collect();
    let expected_fields: Vec<_> = expected.split('|').collect();
    assert_eq!(
        actual_fields.len(),
        NAMES.len(),
        "malformed {context} signature"
    );
    assert_eq!(
        expected_fields.len(),
        NAMES.len(),
        "malformed recorded signature"
    );
    let mut moves = Vec::new();
    for coordinate in sequence.bytes() {
        let index = usize::from(coordinate - b'0');
        moves.push(format!("({}, {})", index / 3, index % 3));
    }
    let history = if moves.is_empty() {
        "initial position".to_owned()
    } else {
        moves.join(", ")
    };
    for (index, name) in NAMES.iter().enumerate() {
        assert_eq!(
            actual_fields[index], expected_fields[index],
            "{context}: {name} differs after {history}; left = actual, right = recorded Java"
        );
    }
}

#[test]
fn java_positions_match_rust_replays() {
    // Default fixture is compiled into the test.
    // An optional owned override allows checking regenerated output before replacing the fixture.
    let override_fixture = std::env::var("JAVA_TAG_FIXTURE").ok().map(|path| {
        // Read the newly exported Java results from this supplied path.
        std::fs::read_to_string(path).expect("Read Java fixture override")
    });
    let fixture = override_fixture
        .as_deref()
        .unwrap_or(include_str!("../../fixtures/tictactoe-java.tsv"));
    let mut lines = fixture.lines();
    assert_eq!(
        lines.next(),
        Some(
            "# tag-java-tictactoe-v1 d65ad79f04992e238663c37c812dc2c5f6ce7ff2 clean-user-reported"
        )
    );
    let model = TicTacToeForwardModel;
    let mut boards = HashSet::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 4, "malformed Java fixture row");
        let sequence = fields[0];
        let mut state = model
            .init_state(&TicTacToeParameters::default(), 2)
            .unwrap();
        for coordinate in sequence.bytes() {
            assert!((b'0'..=b'8').contains(&coordinate));
            let index = usize::from(coordinate - b'0');
            let Decision::Player(actor) = model.next_player(&state) else {
                panic!("move after completion: {sequence}");
            };
            model
                .next(
                    &mut state,
                    actor,
                    &PlaceMark {
                        x: index / 3,
                        y: index % 3,
                    },
                )
                .unwrap();
        }
        let actual = signature(&model, &state);
        compare_fields(&actual, fields[1], sequence, "Rust full state");
        assert!(boards.insert(fields[1].split('|').next().unwrap()));
        for observer in [PlayerId::new(0), PlayerId::new(1)] {
            // Java player copies still enumerate current-player moves even for a nonacting observer.
            // Compare visible facts separately for inactive observers.
            let view = model.observe(&state, observer).unwrap();
            compare_fields(
                fields[observer.index() + 2],
                fields[1],
                sequence,
                &format!("Java player {} copy", observer.index()),
            );
            assert_eq!(view.board(), state.board());
            assert_eq!(view.next_player(), model.next_player(&state));
            assert_eq!(view.move_count(), state.move_count());
            for seat in [PlayerId::new(0), PlayerId::new(1)] {
                assert_eq!(view.player_result(seat), model.player_result(&state, seat));
                assert_eq!(view.player_score(seat), model.player_score(&state, seat));
            }
        }
    }
    assert_eq!(boards.len(), 5478);
    println!(
        "Compared all {} Java positions with Rust replays",
        boards.len()
    );
}
