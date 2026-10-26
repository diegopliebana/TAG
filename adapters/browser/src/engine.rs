//! Receive browser requests and route them to the selected game.
//!
//! Replies carry request IDs so JavaScript knows which operation completed. A game
//! generation distinguishes a newly created/replaced session from the previous one.
use crate::protocol::*;
use crate::runtime::{Runtime, integer};
use tag_rust::games::{
    cantstop::CantStopForwardModel, connect4::Connect4ForwardModel,
    loveletter::LoveLetterForwardModel, tictactoe::TicTacToeForwardModel,
};

pub(crate) enum GameSession {
    Tictactoe(Runtime<TicTacToeForwardModel>),
    Connect4(Runtime<Connect4ForwardModel>),
    Cantstop(Box<Runtime<CantStopForwardModel>>),
    Loveletter(Box<Runtime<LoveLetterForwardModel>>),
}
impl GameSession {
    fn import(
        text: &str,
        controllers: Option<Vec<SeatSetup>>,
        models: &crate::learned::Registry,
    ) -> Result<(Self, u32, bool), BridgeError> {
        let header: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| BridgeError::new("invalid_file", e.to_string()))?;
        match header["versions"]["game"].as_str() {
            Some("loveletter") => {
                Runtime::import_with_models(text, LoveLetterForwardModel, controllers, models)
                    .map(|(r, d, n)| (Self::Loveletter(Box::new(r)), d, n))
            }
            Some("cantstop") => {
                Runtime::import_with_models(text, CantStopForwardModel, controllers, models)
                    .map(|(r, d, n)| (Self::Cantstop(Box::new(r)), d, n))
            }
            Some("tictactoe") => {
                Runtime::import_with_models(text, TicTacToeForwardModel, controllers, models)
                    .map(|(r, d, n)| (Self::Tictactoe(r), d, n))
            }
            Some("connect4") => {
                Runtime::import_with_models(text, Connect4ForwardModel, controllers, models)
                    .map(|(r, d, n)| (Self::Connect4(r), d, n))
            }
            _ => Err(BridgeError::new(
                "invalid_file",
                "unsupported or missing game identity",
            )),
        }
    }
    fn export(
        &mut self,
        scope: SaveScope,
        cursor: usize,
        delay: u32,
    ) -> Result<String, BridgeError> {
        match self {
            Self::Tictactoe(r) => r.export(scope, cursor, delay),
            Self::Connect4(r) => r.export(scope, cursor, delay),
            Self::Cantstop(r) => r.export(scope, cursor, delay),
            Self::Loveletter(r) => r.export(scope, cursor, delay),
        }
    }
    fn file_data(&mut self) -> Result<(PublicView, PublicView, serde_json::Value), BridgeError> {
        match self {
            Self::Tictactoe(r) => Ok((r.file_display()?, r.view(None)?, r.file_parameters()?)),
            Self::Connect4(r) => Ok((r.file_display()?, r.view(None)?, r.file_parameters()?)),
            Self::Cantstop(r) => Ok((r.file_display()?, r.view(None)?, r.file_parameters()?)),
            Self::Loveletter(r) => Ok((r.file_display()?, r.view(None)?, r.file_parameters()?)),
        }
    }
    fn create(
        game: GameKind,
        parameters: Option<serde_json::Value>,
        controllers: Option<Vec<SeatSetup>>,
        models: &crate::learned::Registry,
    ) -> Result<Self, BridgeError> {
        match game {
            GameKind::Loveletter => {
                Runtime::create_with_models(LoveLetterForwardModel, parameters, controllers, models)
                    .map(|r| Self::Loveletter(Box::new(r)))
            }
            GameKind::Cantstop => {
                Runtime::create_with_models(CantStopForwardModel, parameters, controllers, models)
                    .map(|r| Self::Cantstop(Box::new(r)))
            }
            GameKind::Tictactoe => {
                Runtime::create_with_models(TicTacToeForwardModel, parameters, controllers, models)
                    .map(Self::Tictactoe)
            }
            GameKind::Connect4 => {
                Runtime::create_with_models(Connect4ForwardModel, parameters, controllers, models)
                    .map(Self::Connect4)
            }
        }
    }
    fn view(&self) -> Result<PublicView, BridgeError> {
        match self {
            Self::Tictactoe(g) => g.view(None),
            Self::Connect4(g) => g.view(None),
            Self::Cantstop(g) => g.view(None),
            Self::Loveletter(g) => g.view(None),
        }
    }
    fn execute(
        &mut self,
        command: Command,
        version: Option<String>,
    ) -> Result<(PublicView, Option<ProposalView>), BridgeError> {
        match self {
            Self::Tictactoe(g) => g.execute(command, version),
            Self::Connect4(g) => g.execute(command, version),
            Self::Cantstop(g) => g.execute(command, version),
            Self::Loveletter(g) => g.execute(command, version),
        }
    }
}
/// One browser-owned engine lifecycle. JavaScript exchanges owned JSON messages.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub struct BrowserEngine {
    models: crate::learned::Registry,
    generation: u64,
    session: Option<GameSession>,
}
impl Default for BrowserEngine {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
impl BrowserEngine {
    /// Construct an owned empty browser lifecycle at generation zero.
    /// There are no inputs; a Create command selects game/parameters/controllers.
    #[cfg_attr(
        target_arch = "wasm32",
        wasm_bindgen::prelude::wasm_bindgen(constructor)
    )]
    pub fn new() -> Self {
        Self {
            models: Default::default(),
            generation: 0,
            session: None,
        }
    }
    /// Process one command through the portable Rust engine.
    ///
    /// # Arguments
    /// Exclusive self access serializes mutations. `text` is passed by a shared (read-only) reference request
    /// JSON with request_id, generation, optional position_version and command.
    ///
    /// # Returns
    /// Owned JSON reply echoing request identity, with view containing only information the player is allowed to see/proposal or typed error.
    ///
    /// # Errors
    /// Errors are represented in the reply: malformed/oversized requests, stale
    /// lifecycle/position, invalid setup/action/proposal or engine failure.
    pub fn handle(&mut self, text: &str) -> String {
        let request = if text.len() > 32768 {
            Err("browser command exceeds 32 KiB".to_string())
        } else {
            serde_json::from_str::<serde_json::Value>(text).map_err(|e| e.to_string())
        };
        let mut reply = Reply {
            request_id: String::new(),
            request_generation: String::new(),
            generation: self.generation.to_string(),
            view: None,
            proposal: None,
            error: None,
        };
        match request {
            Err(e) => reply.error = Some(BridgeError::new("invalid_request", e)),
            Ok(value) => {
                // Retain request identity even when its command or fields fail decoding.
                reply.request_id = value
                    .get("request_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .into();
                reply.request_generation = value
                    .get("generation")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .into();
                let result = serde_json::from_value::<Request>(value)
                    .map_err(|e| BridgeError::new("invalid_request", e.to_string()))
                    .and_then(|r| self.execute(r));
                reply.generation = self.generation.to_string();
                match result {
                    Ok((view, proposal)) => {
                        reply.view = Some(view);
                        reply.proposal = proposal;
                    }
                    Err(e) => reply.error = Some(e),
                }
            }
        }
        serde_json::to_string(&reply).expect("browser reply fields are JSON-serializable")
    }
    /// Process a bounded file transfer independently of ordinary command limits.
    ///
    /// # Arguments
    /// `self` is accessed through an exclusive mutable reference. `envelope` is borrowed small file-operation
    /// JSON; `text` is borrowed file contents before decoding content (empty for exports).
    ///
    /// # Returns
    /// Owned reply carrying the ID of the request it answers with exported text or validated display/live views.
    ///
    /// # Errors
    /// Reply contains malformed/oversized, stale lifecycle or validation errors.
    /// Failed imports never replace the current session.
    pub fn handle_file(&mut self, envelope: &str, text: &str) -> String {
        let mut reply = FileReply {
            models: None,
            model_sources: None,
            request_id: String::new(),
            request_generation: String::new(),
            generation: self.generation.to_string(),
            view: None,
            live_view: None,
            parameters: None,
            delay_ms: None,
            needs_controllers: false,
            text: None,
            error: None,
        };
        let result = (|| {
            if envelope.len() > 32768 {
                return Err(BridgeError::new(
                    "resource_limit",
                    "file envelope exceeds 32 KiB",
                ));
            }
            let value: serde_json::Value = serde_json::from_str(envelope)
                .map_err(|e| BridgeError::new("invalid_request", e.to_string()))?;
            reply.request_id = value["request_id"].as_str().unwrap_or_default().into();
            reply.request_generation = value["generation"].as_str().unwrap_or_default().into();
            let request: FileRequest = serde_json::from_value(value)
                .map_err(|e| BridgeError::new("invalid_request", e.to_string()))?;
            integer(&request.request_id)?;
            if integer(&request.generation)? != self.generation {
                return Err(BridgeError::new(
                    "stale_generation",
                    "file request belongs to a replaced game",
                ));
            }
            if text.len() > 10 * 1024 * 1024 {
                return Err(BridgeError::new("resource_limit", "file exceeds 10 MiB"));
            }
            match request.command {
                FileCommand::LoadModel => {
                    let id = crate::learned::insert(&mut self.models, text)
                        .map_err(|e| BridgeError::new("invalid_model", e.to_string()))?;
                    reply.text = Some(id);
                    if let Some(session) = &mut self.session {
                        session.update_models(&self.models);
                    }
                    reply.models = Some(crate::learned::summaries(&self.models));
                }
                FileCommand::Models => {
                    reply.models = Some(crate::learned::summaries(&self.models));
                }
                FileCommand::PredictModel { model_id, observer } => {
                    let session = self
                        .session
                        .as_ref()
                        .ok_or_else(|| BridgeError::new("no_game", "create a game first"))?;
                    reply.text = Some(
                        match session {
                            GameSession::Tictactoe(r) => r.predict_model(&model_id, observer)?,
                            GameSession::Connect4(r) => r.predict_model(&model_id, observer)?,
                            GameSession::Cantstop(r) => r.predict_model(&model_id, observer)?,
                            GameSession::Loveletter(r) => r.predict_model(&model_id, observer)?,
                        }
                        .to_string(),
                    );
                }
                FileCommand::Export {
                    scope,
                    cursor,
                    delay_ms,
                } => {
                    reply.text = Some(
                        self.session
                            .as_mut()
                            .ok_or_else(|| BridgeError::new("no_game", "create a game first"))?
                            .export(scope, cursor, delay_ms)?,
                    );
                }
                FileCommand::Import { controllers } => {
                    let generation = self.generation.checked_add(1).ok_or_else(|| {
                        BridgeError::new("resource_limit", "session generation exhausted")
                    })?;
                    let document: serde_json::Value = serde_json::from_str(text)
                        .map_err(|e| BridgeError::new("invalid_file", e.to_string()))?;
                    let mut models = if document["kind"] == "tag-learned-browser" {
                        crate::learned::Registry::new()
                    } else {
                        self.models.clone()
                    };
                    let replay = if document["kind"] == "tag-learned-browser" {
                        if document["version"] != 1 {
                            return Err(BridgeError::new(
                                "invalid_file",
                                "unsupported learned bundle version",
                            ));
                        }
                        let entries = document["models"]
                            .as_array()
                            .filter(|a| a.len() <= 16)
                            .ok_or_else(|| {
                                BridgeError::new("invalid_file", "invalid model bundle")
                            })?;
                        for entry in entries {
                            crate::learned::insert(&mut models, &entry.to_string())
                                .map_err(|e| BridgeError::new("invalid_file", e.to_string()))?;
                        }
                        document["replay"].to_string()
                    } else {
                        text.into()
                    };
                    let (mut next, delay, needs) =
                        GameSession::import(&replay, controllers, &models)?;
                    let (display, live, parameters) = next.file_data()?;
                    reply.view = Some(display);
                    reply.live_view = Some(live);
                    reply.parameters = Some(parameters);
                    reply.delay_ms = Some(delay);
                    reply.needs_controllers = needs;
                    self.session = Some(next);
                    self.models = models;
                    reply.models = Some(crate::learned::summaries(&self.models));
                    if document["kind"] == "tag-learned-browser" {
                        reply.model_sources = Some(
                            self.models
                                .values()
                                .map(|model| model.as_document().to_string())
                                .collect(),
                        );
                    }
                    self.generation = generation;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            reply.error = Some(error);
        }
        reply.generation = self.generation.to_string();
        serde_json::to_string(&reply).expect("file reply is JSON-serializable")
    }
}
impl BrowserEngine {
    fn execute(&mut self, r: Request) -> Result<(PublicView, Option<ProposalView>), BridgeError> {
        integer(&r.request_id)?;
        let generation = integer(&r.generation)?;
        if generation != self.generation {
            return Err(BridgeError::new(
                "stale_generation",
                "request belongs to a replaced game",
            ));
        }
        match r.command {
            Command::Create {
                game,
                parameters,
                controllers,
            } => {
                let next_generation = self.generation.checked_add(1).ok_or_else(|| {
                    BridgeError::new("resource_limit", "session generation exhausted")
                })?;
                let next = GameSession::create(game, parameters, controllers, &self.models)?;
                let view = next.view()?;
                self.session = Some(next);
                self.generation = next_generation;
                Ok((view, None))
            }
            command => self
                .session
                .as_mut()
                .ok_or_else(|| BridgeError::new("no_game", "create a game first"))?
                .execute(command, r.position_version),
        }
    }
}

impl GameSession {
    fn update_models(&mut self, models: &crate::learned::Registry) {
        match self {
            Self::Tictactoe(r) => r.models = models.clone(),
            Self::Connect4(r) => r.models = models.clone(),
            Self::Cantstop(r) => r.models = models.clone(),
            Self::Loveletter(r) => r.models = models.clone(),
        }
    }
}
