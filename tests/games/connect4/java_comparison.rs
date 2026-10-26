//! Replay recorded comparison results from the saved fixture.
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::connect4::{
    Cell, Connect4ForwardModel, Connect4Parameters, Connect4State, DropPiece,
};

fn result_name(result: PlayerResult) -> &'static str {
    match result {
        PlayerResult::Ongoing => "GAME_ONGOING",
        PlayerResult::Win => "WIN_GAME",
        PlayerResult::Draw => "DRAW_GAME",
        PlayerResult::Loss => "LOSE_GAME",
    }
}

fn signature(model: &Connect4ForwardModel, state: &Connect4State) -> String {
    // Borrow model/state; return owned canonical text for named-field diagnostics.
    let board: String = state
        .board()
        .iter()
        .map(|cell| {
            // Encode each occupied seat using the recorded board symbols.
            match cell {
                Cell::Empty => '.',
                Cell::Piece(p) if p.index() == 0 => 'x',
                Cell::Piece(_) => 'o',
            }
        })
        .collect();
    let next = match state.next_player() {
        Decision::Player(p) => p.index() as i32,
        Decision::Finished => -1,
    };
    let status = match state.status() {
        SingleWinnerOutcome::Ongoing => "GAME_ONGOING",
        SingleWinnerOutcome::Won(_) => "GAME_END",
        SingleWinnerOutcome::Draw => "DRAW_GAME",
    };
    let actions = model
        .available_actions(state, state.current_player())
        .unwrap()
        .iter()
        .map(|action| {
            // Canonical actions include the selected column and derived landing row.
            let y = (0..state.height())
                .rev()
                .find(|&y| {
                    // The first empty cell from the bottom is the landing cell.
                    state.cell(action.column, y).unwrap() == Cell::Empty
                })
                .unwrap();
            format!("{}:{y}", action.column)
        })
        .collect::<Vec<_>>()
        .join(",");
    let winning = state
        .winning_cells()
        .iter()
        .map(|&(x, y)| {
            // Preserve the ordered winning coordinates in the recorded representation.
            format!("{x}:{y}")
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{board}|{}|{next}|{}|{}|{}|{}|{status}|{}|{}|{:.1}|{:.1}|{actions}|{winning}",
        state.current_player().index(),
        state.move_count(),
        state.turn_count_in_round(),
        state.round_count(),
        state.first_player().index(),
        result_name(state.player_result(PlayerId::new(0)).unwrap()),
        result_name(state.player_result(PlayerId::new(1)).unwrap()),
        state.player_score(PlayerId::new(0)).unwrap(),
        state.player_score(PlayerId::new(1)).unwrap()
    )
}

#[test]
fn recorded_positions_copies_actions_and_winning_lines_match_replays() {
    let override_fixture = std::env::var("JAVA_CONNECT4_FIXTURE").ok().map(|path| {
        // An explicit override checks newly generated records before replacing accepted results.
        std::fs::read_to_string(path).unwrap()
    });
    let fixture = override_fixture
        .as_deref()
        .unwrap_or(include_str!("../../fixtures/connect4-java.tsv"));
    let mut lines = fixture.lines();
    assert_eq!(
        lines.next(),
        Some("# tag-java-connect4-v1 d65ad79f04992e238663c37c812dc2c5f6ce7ff2 clean-user-reported")
    );
    let model = Connect4ForwardModel;
    let names = [
        "board",
        "historical actor",
        "next player",
        "moves",
        "turns within round",
        "rounds",
        "first player",
        "status",
        "seat 0 result",
        "seat 1 result",
        "seat 0 score",
        "seat 1 score",
        "ordered actions and landing rows",
        "ordered winning cells",
    ];
    let mut records = 0;
    let mut finished = 0;
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        assert_eq!(fields.len(), 5);
        let config: Vec<usize> = fields[0]
            .split(',')
            .map(|n| {
                // Read each recorded dimension and target as an integer.
                n.parse().unwrap()
            })
            .collect();
        let mut state = model
            .init_state(
                &Connect4Parameters {
                    width: config[0],
                    height: config[1],
                    winning_length: config[2],
                    ..Default::default()
                },
                2,
            )
            .unwrap();
        if !fields[1].is_empty() {
            for column in fields[1].split(',') {
                let actor = state.current_player();
                model
                    .next(
                        &mut state,
                        actor,
                        &DropPiece {
                            column: column.parse().unwrap(),
                        },
                    )
                    .unwrap();
            }
        }
        let actual = signature(&model, &state);
        let actual_fields: Vec<_> = actual.split('|').collect();
        for (copy, expected) in fields[2..].iter().enumerate() {
            let expected_fields: Vec<_> = expected.split('|').collect();
            assert_eq!(expected_fields.len(), names.len());
            for (i, name) in names.iter().enumerate() {
                assert_eq!(
                    actual_fields[i], expected_fields[i],
                    "config {} history [{}], copy {copy}, {name}",
                    fields[0], fields[1]
                );
            }
        }
        for seat in 0..2 {
            let p = PlayerId::new(seat);
            let view = model.observe(&state, p).unwrap();
            assert_eq!(view.board(), state.board());
            assert_eq!(
                (view.width(), view.height(), view.winning_length()),
                (config[0], config[1], config[2])
            );
            assert_eq!(view.winning_cells(), state.winning_cells());
            assert_eq!(view.current_player(), state.current_player());
            assert_eq!(view.first_player(), state.first_player());
            assert_eq!(view.move_count(), state.move_count());
            assert_eq!(view.completed_turn_count(), state.completed_turn_count());
            assert_eq!(view.turn_count_in_round(), state.turn_count_in_round());
            assert_eq!(view.round_count(), state.round_count());
            assert_eq!(view.status(), state.status());
            assert_eq!(view.next_player(), state.next_player());
            for query in 0..2 {
                assert_eq!(
                    view.player_result(PlayerId::new(query)),
                    state.player_result(PlayerId::new(query))
                );
                assert_eq!(
                    view.player_score(PlayerId::new(query)),
                    state.player_score(PlayerId::new(query))
                );
            }
            if state.next_player() == Decision::Player(p) {
                assert_eq!(
                    model.offered_actions(&view),
                    model.available_actions(&state, p).unwrap()
                );
            } else {
                assert!(model.offered_actions(&view).is_empty());
            }
        }
        records += 1;
        if state.next_player() == Decision::Finished {
            finished += 1;
        }
    }
    assert_eq!(records, 3268);
    assert_eq!(finished, 128);
}
