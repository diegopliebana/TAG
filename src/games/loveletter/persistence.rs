//! Privileged snapshots and compatibility import for older reproduction recipes.
mod snapshot;
use super::*;
use crate::{
    core::{ForwardModel, PlayerId, checkpoint::Checkpoint, evaluation::Evaluator},
    replay::{PersistableGame, PersistenceError, ReplayGame},
};
use serde::{Deserialize, Serialize};
pub use snapshot::SavedSnapshot as LoveLetterSnapshot;
pub(super) use snapshot::counters;

/// Complete saved facts live outside simulation state. New exports use a bounded
/// snapshot; older match-long recipes remain readable for compatibility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum LoveLetterSavedState {
    Snapshot {
        snapshot: Box<LoveLetterSnapshot>,
    },
    LegacyProof {
        players: usize,
        initial_deck: Option<[Card; 16]>,
        actions: Vec<(PlayerId, LoveLetterAction)>,
    },
}
fn invalid(e: impl std::fmt::Display) -> PersistenceError {
    PersistenceError::Invalid(e.to_string())
}
impl PersistableGame for LoveLetterForwardModel {
    type SavedState = LoveLetterSavedState;
    const GAME_ID: &'static str = "loveletter";
    const RULES_VERSION: u32 = 1;
    fn storage_units(_: &LoveLetterParameters) -> Option<usize> {
        Some(16)
    }
    fn validation_work(_: &LoveLetterParameters, data: &LoveLetterSavedState) -> Option<usize> {
        match data {
            LoveLetterSavedState::Snapshot { snapshot } => snapshot.validation_work(),
            LoveLetterSavedState::LegacyProof { actions, .. } => {
                actions.len().checked_mul(256)?.checked_add(256)
            }
        }
    }
    fn state_data(&self, s: &LoveLetterState) -> LoveLetterSavedState {
        LoveLetterSavedState::Snapshot {
            snapshot: Box::new(snapshot::SavedSnapshot::from_state(s)),
        }
    }
    fn restore_state(
        &self,
        p: &LoveLetterParameters,
        d: &LoveLetterSavedState,
    ) -> Result<LoveLetterState, PersistenceError> {
        let (players, initial_deck, actions) = match d {
            LoveLetterSavedState::Snapshot { snapshot } => return snapshot.restore(self, p),
            LoveLetterSavedState::LegacyProof {
                players,
                initial_deck,
                actions,
            } => (players, initial_deck, actions),
        };
        if actions.len() > 100_000 {
            return Err(invalid("Love Letter proof exceeds 100000 actions"));
        }
        let mut s = match initial_deck {
            Some(deck) => self.init_with_deck(p, *players, deck),
            None => self.init_state(p, *players),
        }
        .map_err(invalid)?;
        for (actor, action) in actions {
            self.next(&mut s, *actor, action).map_err(invalid)?;
        }
        Ok(s)
    }
}
/// Compatibility evaluator required by the shared replay trait. Browser policies
/// persist their actual heuristic and RNG state in their independent codec.
#[derive(Default)]
pub struct ReplayEvaluator;
impl Evaluator<LoveLetterState> for ReplayEvaluator {
    type Error = LoveLetterError;
    fn evaluate(&mut self, s: &LoveLetterState, p: PlayerId) -> Result<f64, Self::Error> {
        LoveLetterHeuristic.evaluate(s, p)
    }
}
impl Checkpoint for ReplayEvaluator {
    type Snapshot = ();
    type RestoreError = std::convert::Infallible;
    fn snapshot(&self) {}
    fn from_snapshot(_: &()) -> Result<Self, Self::RestoreError> {
        Ok(Self)
    }
}
impl ReplayGame for LoveLetterForwardModel {
    type SavedEvaluator = ReplayEvaluator;
    // A privileged root snapshot preserves chance; ordinary move events contain
    // only publicly played intents, never hidden draws.
    const HAS_CHANCE: bool = true;
    fn verification_work(_: &LoveLetterParameters, osla: bool) -> Option<usize> {
        Some(if osla { 100_000 } else { 256 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::Decision,
        replay::{ReplayLimits, export_state, import_state},
    };
    #[test]
    fn controlled_and_shuffled_states_preserve_private_memory_and_exact_future_chance() {
        let model = LoveLetterForwardModel;
        for n in 2..=4 {
            for controlled in [false, true] {
                let p = LoveLetterParameters {
                    seed: 42,
                    rules: LoveLetterRules {
                        winning_tokens: [2; 3],
                    },
                    ..Default::default()
                };
                let mut state = if controlled {
                    model.init_with_deck(&p, n, &Card::DECK)
                } else {
                    model.init_state(&p, n)
                }
                .unwrap();
                for _ in 0..50 {
                    let text = export_state(&model, &p, &state).unwrap();
                    let (_, mut restored) =
                        import_state(&text, &model, ReplayLimits::default()).unwrap();
                    assert_eq!(state, restored);
                    let Decision::Player(actor) = model.next_player(&state) else {
                        break;
                    };
                    let action = model.available_actions(&state, actor).unwrap()[0];
                    model.next(&mut state, actor, &action).unwrap();
                    model.next(&mut restored, actor, &action).unwrap();
                    assert_eq!(state, restored);
                }
            }
        }
    }
    #[test]
    fn malformed_and_excessive_proofs_fail_before_becoming_states() {
        let model = LoveLetterForwardModel;
        let p = LoveLetterParameters::default();
        let state = model.init_state(&p, 3).unwrap();
        let d = LoveLetterSavedState::LegacyProof {
            players: 3,
            initial_deck: None,
            actions: vec![(PlayerId::new(99), LoveLetterAction::Play(Card::Guard))],
        };
        assert!(model.restore_state(&p, &d).is_err());
        let d = LoveLetterSavedState::LegacyProof {
            players: 3,
            initial_deck: None,
            actions: vec![(PlayerId::new(0), LoveLetterAction::Play(Card::Guard)); 10_001],
        };
        let mut text: serde_json::Value =
            serde_json::from_str(&export_state(&model, &p, &state).unwrap()).unwrap();
        text["state"] = serde_json::to_value(d).unwrap();
        assert!(
            import_state(
                &text.to_string(),
                &model,
                ReplayLimits {
                    max_verification_work: 1_000_000,
                    ..ReplayLimits::default()
                }
            )
            .unwrap_err()
            .to_string()
            .contains("work")
        );
        let wrong = LoveLetterParameters { seed: 1, ..p };
        assert!(export_state(&model, &wrong, &state).is_err());
    }
    #[test]
    fn legacy_round_recipes_import_but_new_exports_contain_only_complete_snapshots() {
        use crate::core::random::AgentRng;
        let model = LoveLetterForwardModel;
        for (n, initial_deck) in
            (2..=4).flat_map(|n| [None, Some(Card::DECK)].map(|deck| (n, deck)))
        {
            let p = LoveLetterParameters {
                seed: 91,
                rules: LoveLetterRules {
                    winning_tokens: [1000; 3],
                },
                ..Default::default()
            };
            let mut live = match initial_deck {
                Some(deck) => model.init_with_deck(&p, n, &deck).unwrap(),
                None => model.init_state(&p, n).unwrap(),
            };
            let mut rng = AgentRng::new(77);
            let mut actions = Vec::new();
            for step in 0..300 {
                let Decision::Player(actor) = model.next_player(&live) else {
                    panic!("high-target fixture ended");
                };
                let legal = model.available_actions(&live, actor).unwrap();
                let action = legal[rng.index(legal.len()).unwrap()];
                model.next(&mut live, actor, &action).unwrap();
                actions.push((actor, action));
                if step % 37 == 0 {
                    let legacy = LoveLetterSavedState::LegacyProof {
                        players: n,
                        initial_deck,
                        actions: actions.clone(),
                    };
                    assert_eq!(model.restore_state(&p, &legacy).unwrap(), live);
                    let text = export_state(&model, &p, &live).unwrap();
                    let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
                    assert!(json["state"].get("snapshot").is_some());
                    assert!(json["state"].get("actions").is_none());
                    assert!(json["state"]["snapshot"].get("provenance").is_none());
                    assert!(text.len() < 100_000); // Retained rounds, not the whole match.
                    let (_, copy) = import_state(&text, &model, ReplayLimits::default()).unwrap();
                    assert_eq!(copy, live);
                    json["state"] = serde_json::to_value(&legacy).unwrap();
                    let (_, imported_old_file) =
                        import_state(&json.to_string(), &model, ReplayLimits::default()).unwrap();
                    assert_eq!(imported_old_file, live);
                }
            }
            assert!(live.public.round > 10);
        }
    }

    #[test]
    fn snapshots_reject_forged_cards_memory_counters_rng_and_extra_fields() {
        let model = LoveLetterForwardModel;
        let p = LoveLetterParameters {
            seed: 42,
            ..Default::default()
        };
        let mut live = model.init_state(&p, 3).unwrap();
        while live.public.round < 3 {
            let actor = live.public.current_player;
            let legal = model.available_actions(&live, actor).unwrap();
            model.next(&mut live, actor, &legal[0]).unwrap();
        }
        let text = export_state(&model, &p, &live).unwrap();
        let original: serde_json::Value = serde_json::from_str(&text).unwrap();
        for pointer in [
            "/state/snapshot/hands/0/0",
            "/state/snapshot/public/reserve",
            "/state/snapshot/chance/initial_seed",
            "/state/snapshot/completed_private/0",
            "/state/snapshot/private/0",
            "/state/snapshot/public/counters",
            "/state/snapshot/public/last_round/summary/tokens",
        ] {
            let mut value = original.clone();
            let slot = value.pointer_mut(pointer).expect(pointer);
            *slot = match pointer {
                "/state/snapshot/hands/0/0" => serde_json::json!("princess"),
                "/state/snapshot/public/reserve" => serde_json::json!(["guard"]),
                "/state/snapshot/chance/initial_seed" => serde_json::json!("999"),
                "/state/snapshot/public/counters" => serde_json::json!([0, 0, 0, 0]),
                "/state/snapshot/public/last_round/summary/tokens" => {
                    serde_json::json!([999, 0, 0])
                }
                _ => serde_json::json!([]),
            };
            // Replacing a Guard with Princess must change the multiset in this fixture.
            assert_ne!(value, original);
            assert!(
                import_state(&value.to_string(), &model, ReplayLimits::default()).is_err(),
                "{pointer}"
            );
        }
        let mut value = original;
        value["state"]["snapshot"]["unexpected_secret"] = serde_json::json!(1);
        assert!(import_state(&value.to_string(), &model, ReplayLimits::default()).is_err());
    }
}
