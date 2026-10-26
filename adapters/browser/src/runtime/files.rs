//! Use portable file validation without reading/writing a filesystem in the engine.
//!
//! The browser supplies file text and downloads exported text. Imports construct a
//! separate session and replace live play only after successful validation.
use super::*;
use crate::games::BrowserReplay;
use crate::learned::Registry;
use tag_rust::replay::{
    ReplayLimits, export_branch_with, export_replay_with, export_state, import_replay_with,
    import_state,
};

fn file_error(error: impl std::fmt::Display) -> BridgeError {
    BridgeError::new("invalid_file", error.to_string())
}

impl<F> Runtime<F>
where
    F: BrowserReplay,
    F::State: PartialEq,
    F::Parameters: Clone + serde::Serialize,
{
    /// Inspect raw model predictions for a live permitted observer view.
    /// Borrowed identity selects immutable weights; observer selects public encoding.
    /// Returns independent numeric JSON; mismatches leave game and RNG unchanged.
    pub(crate) fn predict_model(&self, id: &str, observer: usize) -> Result<Value, BridgeError> {
        let loaded = self
            .models
            .get(id)
            .ok_or_else(|| file_error("unknown model"))?;
        let session = self.game.active().session();
        session
            .model()
            .prediction(session.state(), &self.parameters, loaded, observer)
    }
    /// Import validated replay/state into an independent runtime.
    ///
    /// # Arguments
    /// `text` is borrowed bounded JSON; `model` is consumed; `seats` consumes optional
    /// fresh state-file controllers. Replay controllers always come from the file.
    ///
    /// # Returns
    /// Owned candidate runtime, saved delay and whether state controller setup is needed.
    ///
    /// # Errors
    /// Invalid/unsupported files, exceeded limits or incompatible controller setup.
    pub(crate) fn import_with_models(
        text: &str,
        model: F,
        seats: Option<Vec<SeatSetup>>,
        models: &Registry,
    ) -> Result<(Self, u32, bool), BridgeError> {
        let data: Value = serde_json::from_str(text).map_err(file_error)?;
        let parameters: F::Parameters =
            serde_json::from_value(data["parameters"].clone()).map_err(file_error)?;
        let (width, height) = F::dimensions(&parameters);
        if width > 128 || height > 128 || width.checked_mul(height).is_none_or(|n| n > 4096) {
            return Err(BridgeError::new(
                "resource_limit",
                "browser board limit is 128 per dimension and 4096 cells",
            ));
        }
        let limits = ReplayLimits {
            max_depth: 63,
            ..ReplayLimits::default()
        };
        let (game, delay, needs_controllers) = match data["kind"].as_str() {
            Some("tag-replay") => {
                if seats.is_some() {
                    return Err(file_error("replay controllers must come from the file"));
                }
                // Check every recorded policy, including replacement boundaries, before verification.
                fn has_osla(value: &Value) -> bool {
                    match value {
                        Value::Object(fields) => {
                            fields.get("kind").and_then(Value::as_str) == Some("osla")
                                || fields.values().any(has_osla)
                        }
                        Value::Array(items) => items.iter().any(has_osla),
                        _ => false,
                    }
                }
                if width * height > 256 && has_osla(&data["branches"]) {
                    return Err(BridgeError::new(
                        "resource_limit",
                        "browser OSLA limit is 256 cells",
                    ));
                }
                model.validate_models(&parameters, models, text)?;
                let loaded = import_replay_with(text, model, limits, &F::codec(models))
                    .map_err(file_error)?;
                if loaded.delay_ms > 5000 {
                    return Err(file_error(
                        "browser move delay must be between 0 and 5000 ms",
                    ));
                }
                (loaded.session, (loaded.delay_ms, loaded.cursor), false)
            }
            Some("tag-state") => {
                let (parameters, state) = import_state(text, &model, limits).map_err(file_error)?;
                let needs = seats.is_none();
                let seats =
                    seats.unwrap_or_else(|| vec![SeatSetup::Human; model.player_count(&state)]);
                F::validate_seats(&parameters, &seats, models)?;
                if width * height > 256 && seats.iter().any(|s| matches!(s, SeatSetup::Osla { .. }))
                {
                    return Err(BridgeError::new(
                        "resource_limit",
                        "browser OSLA limit is 256 cells",
                    ));
                }
                let controllers = seats
                    .into_iter()
                    .map(|setup| F::controller_with_models(&model, &parameters, setup, models))
                    .collect::<Result<Vec<_>, _>>()?;
                let session = Session::from_state(model, state, controllers).map_err(file_error)?;
                let _ = parameters;
                (BranchSession::new(session), (500, 0), needs)
            }
            _ => return Err(file_error("expected tag-replay or tag-state document")),
        };
        let stored_moves = (0..game.branch_count())
            .map(|id| {
                game.moves(BranchId(id))
                    .map(|m| m.len())
                    .map_err(file_error)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        let origins = (0..game.branch_count())
            .map(|id| {
                if id == 0 {
                    BranchOrigin::Original
                } else {
                    BranchOrigin::Fork
                }
            })
            .collect();
        let branch = game.active_branch();
        let tip = game.info(branch).map_err(file_error)?.move_count;
        let display = (delay.1 < tip).then_some((branch, delay.1));
        Ok((
            Self {
                calculation: None,
                view_observer: None,
                public_only: false,
                reveal_hidden_information: false,
                parameters,
                models: models.clone(),
                game,
                pending: None,
                next_proposal: 0,
                display,
                stored_moves,
                origins,
            },
            delay.0,
            needs_controllers,
        ))
    }

    /// Export an independent owned file from the selected displayed position.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference only for historical caching; `scope` selects
    /// session/branch/state and copied `cursor`/`delay_ms` select display settings.
    ///
    /// # Returns
    /// Owned portable JSON, without changing live progress or pending controller RNG.
    ///
    /// # Errors
    /// Invalid cursor/delay, parameter disagreement or serialization/size failure.
    pub(crate) fn export(
        &mut self,
        scope: SaveScope,
        cursor: usize,
        delay_ms: u32,
    ) -> Result<String, BridgeError> {
        if delay_ms > 5000 {
            return Err(file_error("move delay exceeds 5000 ms"));
        }
        let text = match scope {
            SaveScope::Session => export_replay_with(
                &self.game,
                &self.parameters,
                cursor,
                delay_ms,
                &F::codec(&self.models),
            ),
            SaveScope::Branch => export_branch_with(
                &self.game,
                &self.parameters,
                cursor,
                delay_ms,
                &F::codec(&self.models),
            ),
            SaveScope::State => {
                let state = self
                    .game
                    .position_at(self.game.active_branch(), cursor)
                    .map_err(file_error)?;
                export_state(
                    self.game.active().session().model(),
                    &self.parameters,
                    &state,
                )
            }
        }
        .map_err(file_error)?;
        if text.len() > 10 * 1024 * 1024 {
            return Err(BridgeError::new(
                "resource_limit",
                "export exceeds 10 MiB file limit",
            ));
        }
        let references = crate::learned::references(&text)?;
        if !matches!(scope, SaveScope::State) && !references.is_empty() {
            let mut used = Registry::new();
            for (id, model) in &self.models {
                if references.contains(id) {
                    used.insert(id.clone(), model.clone());
                }
            }
            let text=serde_json::to_string(&serde_json::json!({"kind":"tag-learned-browser","version":1,"models":crate::learned::documents(&used),"replay":serde_json::from_str::<Value>(&text).map_err(file_error)?})).map_err(file_error)?;
            if text.len() > 10 * 1024 * 1024 {
                return Err(file_error("model bundle exceeds 10 MiB"));
            }
            return Ok(text);
        }
        Ok(text)
    }
    /// Serialize the runtime's original parameters for file setup/display.
    ///
    /// # Arguments
    /// `self` is passed by a shared (read-only) reference; the runtime retains parameter ownership.
    ///
    /// # Returns
    /// Owned JSON configuration, including defaults expanded at creation.
    ///
    /// # Errors
    /// Parameter serialization failure.
    pub(crate) fn file_parameters(&self) -> Result<Value, BridgeError> {
        F::parameters_view(&self.parameters)
    }
    /// Construct a loaded display view with its full selected move list.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference only for historical reconstruction caching.
    ///
    /// # Returns
    /// Owned view containing only information the player is allowed to see at the saved cursor or live tip.
    ///
    /// # Errors
    /// Invalid saved position or observation/record reconstruction failure.
    pub(crate) fn file_display(&mut self) -> Result<PublicView, BridgeError> {
        if let Some((branch, cursor)) = self.display {
            self.history_view(branch, cursor, None)
        } else {
            let mut view = self.view(None)?;
            view.moves = Some(self.move_list(self.game.active_branch(), view.tip)?);
            Ok(view)
        }
    }
}
