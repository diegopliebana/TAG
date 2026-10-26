use tag_rust::core::checkpoint::Checkpoint;
use tag_rust::core::random::{AgentRng, RngSnapshotError};

#[test]
fn generator_checkpoints_restore_exact_future_draws_across_buffer_boundaries() {
    for seed in [0, 1, 100, u64::MAX] {
        for draws in [0, 1, 7, 8, 15, 16, 31, 32, 33, 127] {
            let mut source = AgentRng::new(seed);
            for _ in 0..draws {
                source.index(17);
            }
            let saved = source.snapshot();
            assert_eq!(saved.word_position, draws * 2);
            let mut restored = AgentRng::from_snapshot(&saved).unwrap();
            for count in [0, 1, 2, 7, 51, usize::MAX, 9, 100] {
                assert_eq!(source.index(count), restored.index(count));
                assert_eq!(source.snapshot(), restored.snapshot());
            }
        }
    }
}

#[test]
fn checkpoints_preserve_stream_offsets_and_rejection_sampling_progress() {
    let mut snapshot = AgentRng::new(42).snapshot();
    snapshot.stream = 987;
    snapshot.word_position = (1u128 << 65) + 63; // Exercise a large, odd buffered word offset.
    let mut source = AgentRng::from_snapshot(&snapshot).unwrap();
    assert_eq!(source.snapshot(), snapshot);
    let mut found_rejection = false;
    let count = (usize::MAX / 2) + 2;
    for _ in 0..100 {
        let before = source.snapshot().word_position;
        source.index(count);
        found_rejection |= source.snapshot().word_position - before > 2;
        let mut restored = AgentRng::from_snapshot(&source.snapshot()).unwrap();
        assert_eq!(source.index(count), restored.index(count));
        assert_eq!(source.snapshot(), restored.snapshot());
    }
    // Rejection probability is substantial on a 64-bit host; small bounds on wasm32 remain valid.
    if usize::BITS == 64 {
        assert!(found_rejection);
    }
}

#[test]
fn incompatible_rng_snapshots_are_rejected_instead_of_silently_wrapping() {
    let mut snapshot = AgentRng::new(17).snapshot();
    snapshot.version = 99;
    assert!(matches!(
        AgentRng::from_snapshot(&snapshot),
        Err(RngSnapshotError::UnsupportedVersion(99))
    ));
    snapshot.version = 1;
    snapshot.word_position = 1u128 << 68;
    assert!(matches!(
        AgentRng::from_snapshot(&snapshot),
        Err(RngSnapshotError::InvalidWordPosition(_))
    ));
}
