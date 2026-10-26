//! Reproducible legal play checked against independent row-major rules.
use std::collections::HashSet;
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::connect4::{Cell, Connect4ForwardModel, Connect4Parameters, DropPiece};

// This oracle counts directly from each cell.
fn outcome(
    board: &[u8],
    width: usize,
    height: usize,
    target: usize,
) -> (SingleWinnerOutcome, Option<usize>) {
    for y in 0..height {
        for x in 0..width {
            let mark = board[y * width + x];
            if mark == 0 {
                continue;
            }
            for (direction, (dx, dy)) in [(1, 0), (0, 1), (1, 1), (1, -1)].into_iter().enumerate() {
                let mut matched = true;
                for distance in 1..target {
                    let xx = x as isize + dx * distance as isize;
                    let yy = y as isize + dy * distance as isize;
                    if xx < 0
                        || yy < 0
                        || xx >= width as isize
                        || yy >= height as isize
                        || board[yy as usize * width + xx as usize] != mark
                    {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    return (
                        SingleWinnerOutcome::Won(PlayerId::new((mark - 1) as usize)),
                        Some(direction),
                    );
                }
            }
        }
    }
    if board.contains(&0) {
        (SingleWinnerOutcome::Ongoing, None)
    } else {
        (SingleWinnerOutcome::Draw, None)
    }
}

#[test]
fn reproducible_replays_match_independent_rules_across_experimental_shapes() {
    let model = Connect4ForwardModel;
    let mut prefixes = 0;
    let mut wins = [0; 2];
    let mut draws = 0;
    let mut directions = HashSet::new();
    let configurations = [
        (7, 6, 4),
        (5, 4, 3),
        (10, 8, 6),
        (3, 2, 3),
        (1, 9, 5),
        (9, 1, 5),
        (13, 2, 7),
        (2, 13, 7),
        (1, 1, 1),
        (11, 9, 8),
        (4, 4, 4),
    ];
    for (config, (width, height, target)) in configurations.into_iter().enumerate() {
        for trial in 0..32 {
            let mut state = model
                .init_state(
                    &Connect4Parameters {
                        width,
                        height,
                        winning_length: target,
                        ..Default::default()
                    },
                    2,
                )
                .unwrap();
            let mut board = vec![0_u8; width * height];
            let mut random = 0x9e3779b9_u32 ^ (config as u32 * 97 + trial + 1);
            let mut moves = 0;
            loop {
                prefixes += 1;
                let (expected, direction) = outcome(&board, width, height, target);
                assert_eq!(
                    state.status(),
                    expected,
                    "config {width}x{height}/{target}, trial {trial}, move {moves}"
                );
                for y in 0..height {
                    for x in 0..width {
                        let expected = match board[y * width + x] {
                            0 => Cell::Empty,
                            mark => Cell::Piece(PlayerId::new((mark - 1) as usize)),
                        };
                        assert_eq!(state.cell(x, y), Ok(expected));
                    }
                }
                assert_eq!(state.move_count(), moves);
                let terminal = expected != SingleWinnerOutcome::Ongoing;
                let turns = moves - usize::from(terminal);
                assert_eq!(state.completed_turn_count(), turns);
                assert_eq!(state.turn_count_in_round(), turns % 2);
                assert_eq!(state.round_count(), turns / 2);
                assert_eq!(
                    state.current_player().index(),
                    if terminal { (moves - 1) % 2 } else { moves % 2 }
                );
                for seat in 0..2 {
                    let result = match expected {
                        SingleWinnerOutcome::Ongoing => PlayerResult::Ongoing,
                        SingleWinnerOutcome::Draw => PlayerResult::Draw,
                        SingleWinnerOutcome::Won(p) if p.index() == seat => PlayerResult::Win,
                        SingleWinnerOutcome::Won(_) => PlayerResult::Loss,
                    };
                    assert_eq!(model.player_result(&state, PlayerId::new(seat)), Ok(result));
                    let score = match result {
                        PlayerResult::Win => 1.0,
                        PlayerResult::Loss => -1.0,
                        _ => 0.0,
                    };
                    assert_eq!(model.player_score(&state, PlayerId::new(seat)), Ok(score));
                }
                if terminal {
                    assert_eq!(model.next_player(&state), Decision::Finished);
                    match expected {
                        SingleWinnerOutcome::Won(p) => {
                            wins[p.index()] += 1;
                            if target > 1 {
                                directions.insert(direction.unwrap());
                            }
                            let cells = state.winning_cells();
                            assert_eq!(cells.len(), target);
                            for &(x, y) in cells {
                                assert_eq!(board[y * width + x], (p.index() + 1) as u8);
                            }
                            if target > 1 {
                                let delta = (
                                    cells[1].0 as isize - cells[0].0 as isize,
                                    cells[1].1 as isize - cells[0].1 as isize,
                                );
                                assert!(matches!(delta, (0, -1) | (1, 0) | (1, -1) | (-1, -1)));
                                for pair in cells.windows(2) {
                                    assert_eq!(
                                        (
                                            pair[1].0 as isize - pair[0].0 as isize,
                                            pair[1].1 as isize - pair[0].1 as isize
                                        ),
                                        delta
                                    );
                                }
                            }
                        }
                        SingleWinnerOutcome::Draw => {
                            draws += 1;
                            assert!(state.winning_cells().is_empty());
                        }
                        _ => unreachable!(),
                    }
                    break;
                }
                // Oracle legal choices use row-major top cells, independent of engine enumeration.
                let mut legal = Vec::new();
                for (column, &cell) in board.iter().take(width).enumerate() {
                    if cell == 0 {
                        legal.push(column);
                    }
                }
                let actor = PlayerId::new(moves % 2);
                let offered = model.available_actions(&state, actor).unwrap();
                assert_eq!(
                    offered,
                    legal
                        .iter()
                        .map(|&column| {
                            // Each independently available column corresponds to one drop action.
                            DropPiece { column }
                        })
                        .collect::<Vec<_>>()
                );
                // Test-only xorshift selection generates reproducible replay moves.
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                let column = legal[random as usize % legal.len()];
                let y = (0..height)
                    .rev()
                    .find(|&y| {
                        // Find the landing cell in the independent row-major board.
                        board[y * width + column] == 0
                    })
                    .unwrap();
                let frozen = model.observe(&state, actor).unwrap();
                let frozen_board = frozen.board().to_vec();
                board[y * width + column] = (actor.index() + 1) as u8;
                model
                    .next(&mut state, actor, &DropPiece { column })
                    .unwrap();
                assert_eq!(frozen.board(), frozen_board);
                assert_eq!(frozen.move_count(), moves);
                moves += 1;
            }
        }
    }
    assert!(wins[0] > 0 && wins[1] > 0 && draws > 0);
    assert_eq!(directions.len(), 4);
    eprintln!("352 games, {prefixes} prefixes, wins {wins:?}, draws {draws}, all four directions");
}
