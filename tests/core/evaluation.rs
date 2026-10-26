//! Shared window geometry, contribution arithmetic and cross-game terminal policy.
use crate::support::evaluation::close;
use tag_rust::core::evaluation::{EVALUATION_TIE_TOLERANCE, Evaluator, open_line_score};
use tag_rust::core::lines::windows;
use tag_rust::core::{ForwardModel, PlayerId};
use tag_rust::games::connect4::{
    Connect4Evaluator, Connect4ForwardModel, Connect4Parameters, DropPiece,
};
use tag_rust::games::tictactoe::{
    PlaceMark, TicTacToeEvaluator, TicTacToeForwardModel, TicTacToeParameters,
};

#[test]
fn windows_cover_four_directions_without_duplicate_single_cells() {
    let lines = windows(3, 3, 3)
        .map(|line| {
            // Collect this window's cells to check its exact geometry.
            line.collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 8);
    assert!(lines.contains(&vec![(0, 0), (1, 1), (2, 2)]));
    assert!(lines.contains(&vec![(0, 2), (1, 1), (2, 0)]));
    assert_eq!(windows(7, 6, 4).count(), 69);
    assert_eq!(windows(3, 2, 1).count(), 6);
    assert_eq!(windows(0, 2, 1).count(), 0);
    assert_eq!(windows(2, 2, 0).count(), 0);
    assert_eq!(windows(2, 2, usize::MAX).count(), 0);
    assert_eq!(windows(1, 5, 4).count(), 2);
    assert_eq!(windows(5, 1, 4).count(), 2);
}

#[test]
fn open_line_formula_counts_mixed_and_empty_windows_in_average() {
    let cells = [
        Some(PlayerId::new(0)),
        Some(PlayerId::new(0)),
        None,
        None,
        Some(PlayerId::new(1)),
        None,
        None,
        None,
        Some(PlayerId::new(0)),
        Some(PlayerId::new(1)),
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let lines = (0..4).map(|row| {
        // Each row in this test represents one four-cell window.
        (0..4).map(move |column| (column, row))
    });
    close(
        open_line_score(lines, PlayerId::new(0), |(x, y)| {
            // Read the occupant of this test window's cell.
            cells[y * 4 + x]
        }),
        0.9 * (0.25 - 0.0625) / 4.0,
    );
    close(
        open_line_score(
            std::iter::empty::<Vec<(usize, usize)>>(),
            PlayerId::new(0),
            |_| {
                // Empty traversal returns zero.
                None
            },
        ),
        0.0,
    );
    for (occupant, expected) in [(PlayerId::new(0), 0.9), (PlayerId::new(1), -0.9)] {
        close(
            open_line_score(windows(2, 2, 2), PlayerId::new(0), |_| {
                // Every cell belongs to the same selected seat in this synthetic board.
                Some(occupant)
            }),
            expected,
        );
    }
    assert_eq!(EVALUATION_TIE_TOLERANCE, 0.001);
}

#[test]
fn terminal_wins_for_the_second_seat_override_position_estimates() {
    let ttt_model = TicTacToeForwardModel;
    let mut ttt = ttt_model
        .init_state(&TicTacToeParameters::default(), 2)
        .unwrap();
    for (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 1)] {
        let actor = ttt.current_player();
        ttt_model
            .next(&mut ttt, actor, &PlaceMark { x, y })
            .unwrap();
    }
    let mut ttt_evaluator = TicTacToeEvaluator;
    assert_eq!(ttt_evaluator.evaluate(&ttt, PlayerId::new(1)).unwrap(), 1.0);
    assert_eq!(
        ttt_evaluator.evaluate(&ttt, PlayerId::new(0)).unwrap(),
        -1.0
    );

    let c4_model = Connect4ForwardModel;
    let mut c4 = c4_model
        .init_state(&Connect4Parameters::default(), 2)
        .unwrap();
    for column in [0, 1, 0, 1, 2, 1, 2, 1] {
        let actor = c4.current_player();
        c4_model
            .next(&mut c4, actor, &DropPiece { column })
            .unwrap();
    }
    let mut c4_evaluator = Connect4Evaluator;
    assert_eq!(c4_evaluator.evaluate(&c4, PlayerId::new(1)).unwrap(), 1.0);
    assert_eq!(c4_evaluator.evaluate(&c4, PlayerId::new(0)).unwrap(), -1.0);
}

#[test]
fn window_counts_match_independent_rectangle_formula() {
    for width in 1usize..=10 {
        for height in 1usize..=8 {
            for target in 1usize..=12 {
                let expected = if target == 1 {
                    width * height
                } else {
                    let horizontal_starts = (width + 1).saturating_sub(target);
                    let vertical_starts = (height + 1).saturating_sub(target);
                    horizontal_starts * height
                        + vertical_starts * width
                        + 2 * horizontal_starts * vertical_starts
                };
                assert_eq!(windows(width, height, target).count(), expected);
            }
        }
    }
}
