//! Player views, choices and ownership through the shared rule interface.
use tag_rust::core::outcomes::SingleWinnerOutcome;
use tag_rust::core::{Decision, ForwardModel, PlayerId, PlayerResult};
use tag_rust::games::connect4::{
    Cell, Connect4Error, Connect4ForwardModel, Connect4Parameters, DropPiece,
};

#[test]
fn both_observers_see_the_same_facts_but_only_the_actor_has_choices() {
    let model = Connect4ForwardModel;
    let mut state = model
        .init_state(
            &Connect4Parameters {
                width: 5,
                height: 2,
                winning_length: 4,
                ..Default::default()
            },
            2,
        )
        .unwrap();
    // Two alternating pieces fill column 0 without winning.
    model
        .next(&mut state, PlayerId::new(0), &DropPiece { column: 0 })
        .unwrap();
    model
        .next(&mut state, PlayerId::new(1), &DropPiece { column: 0 })
        .unwrap();
    let before = state.clone();
    for seat in 0..2 {
        let player = PlayerId::new(seat);
        let view = model.observe(&state, player).unwrap();
        assert_eq!(view.observer(), player);
        assert_eq!(view.board(), state.board());
        assert_eq!(
            (view.width(), view.height(), view.winning_length()),
            (5, 2, 4)
        );
        assert_eq!(view.first_player(), state.first_player());
        assert_eq!(view.current_player(), state.current_player());
        assert_eq!(view.next_player(), Decision::Player(PlayerId::new(0)));
        assert_eq!(view.move_count(), 2);
        assert_eq!(view.completed_turn_count(), 2);
        assert_eq!(view.turn_count_in_round(), 0);
        assert_eq!(view.round_count(), 1);
        assert_eq!(view.status(), SingleWinnerOutcome::Ongoing);
        assert!(view.winning_cells().is_empty());
        assert_eq!(view.cell(0, 1), Ok(Cell::Piece(PlayerId::new(0))));
        for query in 0..2 {
            assert_eq!(
                view.player_result(PlayerId::new(query)),
                Ok(PlayerResult::Ongoing)
            );
            assert_eq!(view.player_score(PlayerId::new(query)), Ok(0.0));
        }
        let choices = model.offered_actions(&view);
        if seat == 0 {
            assert_eq!(choices, model.available_actions(&state, player).unwrap());
        } else {
            assert!(choices.is_empty());
        }
    }
    assert_eq!(state, before);
}

#[test]
fn observations_and_their_clones_remain_independent_of_later_moves() {
    let model = Connect4ForwardModel;
    let mut state = model.init_state(&Connect4Parameters::default(), 2).unwrap();
    let initial = model.observe(&state, PlayerId::new(0)).unwrap();
    for column in [0, 1, 0, 1, 0, 1, 0] {
        let actor = state.current_player();
        model
            .next(&mut state, actor, &DropPiece { column })
            .unwrap();
    }
    assert_eq!(initial.cell(0, 5), Ok(Cell::Empty));
    assert_eq!(initial.move_count(), 0);
    assert!(initial.winning_cells().is_empty());
    for seat in 0..2 {
        let view = model.observe(&state, PlayerId::new(seat)).unwrap();
        let copy = view.clone(); // Explicit independent board and winning-coordinate snapshot.
        assert_eq!(view, copy);
        assert_eq!(view.status(), SingleWinnerOutcome::Won(PlayerId::new(0)));
        assert_eq!(view.winning_cells(), &[(0, 5), (0, 4), (0, 3), (0, 2)]);
        assert_eq!(view.next_player(), Decision::Finished);
        assert!(model.offered_actions(&view).is_empty());
        assert_eq!(view.player_result(PlayerId::new(0)), Ok(PlayerResult::Win));
        assert_eq!(view.player_result(PlayerId::new(1)), Ok(PlayerResult::Loss));
        assert_eq!(view.player_score(PlayerId::new(0)), Ok(1.0));
        assert_eq!(view.player_score(PlayerId::new(1)), Ok(-1.0));
    }
}

#[test]
fn draw_views_and_invalid_queries_preserve_the_state() {
    let model = Connect4ForwardModel;
    let mut state = model
        .init_state(
            &Connect4Parameters {
                width: 3,
                height: 2,
                winning_length: 3,
                ..Default::default()
            },
            2,
        )
        .unwrap();
    for column in [0, 1, 2, 0, 1, 2] {
        let actor = state.current_player();
        model
            .next(&mut state, actor, &DropPiece { column })
            .unwrap();
    }
    assert_eq!(state.status(), SingleWinnerOutcome::Draw);
    let before = state.clone();
    for seat in 0..2 {
        let view = model.observe(&state, PlayerId::new(seat)).unwrap();
        assert_eq!(view.next_player(), Decision::Finished);
        assert!(view.winning_cells().is_empty());
        assert!(model.offered_actions(&view).is_empty());
        assert_eq!(
            view.player_result(PlayerId::new(seat)),
            Ok(PlayerResult::Draw)
        );
        assert_eq!(view.player_score(PlayerId::new(seat)), Ok(0.0));
        assert!(view.cell(usize::MAX, 0).is_err());
        let invalid = Connect4Error::InvalidPlayer {
            player: PlayerId::new(2),
        };
        assert_eq!(view.player_result(PlayerId::new(2)), Err(invalid.clone()));
        assert_eq!(view.player_score(PlayerId::new(2)), Err(invalid));
    }
    assert_eq!(
        model.observe(&state, PlayerId::new(2)),
        Err(Connect4Error::InvalidPlayer {
            player: PlayerId::new(2)
        })
    );
    assert_eq!(state, before);
}
