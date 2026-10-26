//! Optional portable model readers and CPU reference inference, independent of rules.
//! Models own immutable weights. Callers supply permitted encoded observations.
mod hash;
mod mcts;
mod network;
pub use mcts::{
    LocalMctsModel, LocalMctsSearch, ModelBindings, ValuePerspective, build_local_mcts,
};
#[cfg(test)]
mod tests;
use crate::learning::{EncodedObservation, EncodingIdentity, LearningSpec};
pub use network::Network;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Maximum JSON file size; parsed object overhead is additional memory.
pub const MAX_MODEL_BYTES: usize = 10 * 1024 * 1024;
/// Checked parsing/inference failure, leaving caller-owned model/game data unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelError(pub String);
impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for ModelError {}
pub(super) fn invalid(message: &str) -> ModelError {
    ModelError(message.into())
}
pub(super) fn json_bytes(value: &Value) -> Vec<u8> {
    // Match Python's sorted, compact ensure_ascii JSON used by portable model identities.
    let text = serde_json::to_string(value).expect("finite JSON data");
    let mut ascii = String::new();
    for character in text.chars() {
        if (character as u32) < 127 {
            ascii.push(character);
        } else {
            for unit in character.encode_utf16(&mut [0; 2]).iter() {
                use std::fmt::Write;
                write!(ascii, "\\u{unit:04x}").expect("writing a string cannot fail");
            }
        }
    }
    ascii.into_bytes()
}
pub(super) fn identity(i: &EncodingIdentity, include_defaults: bool) -> Value {
    let mut result = json!({"game":i.game,"rules_version":i.rules_version,"encoding_version":i.encoding_version,
                         "width":i.width(),"height":i.height(),"target":i.target()});
    if include_defaults {
        result["player_count"] = json!(i.player_count);
        result["configuration"] = match &i.configuration {
            crate::learning::EncodingConfiguration::Placement { .. } => json!([]),
            crate::learning::EncodingConfiguration::Named(settings) => {
                json!(settings.iter().collect::<Vec<_>>())
            }
        };
    }
    result
}

/// A validated Q-table or portable reference network and its original portable document.
#[derive(Clone, Debug)]
pub struct LearnedModel {
    pub(crate) document: Value,
    id: String,
    network: Option<Network>,
    rows: BTreeMap<Vec<i8>, Vec<f64>>,
}
impl PartialEq for LearnedModel {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for LearnedModel {}
impl LearnedModel {
    /// Parse bounded borrowed JSON into a new immutable model; reject invalid formats/weights.
    ///
    /// # Arguments
    /// `text` borrows evaluation JSON; no file access or model code execution occurs.
    ///
    /// # Returns
    /// Owned immutable numeric model and its portable identity/document.
    ///
    /// # Errors
    /// Unsupported formats, invalid metadata/weights/digest or exceeded model limits.
    pub fn parse(text: &str) -> Result<Self, ModelError> {
        if text.len() > MAX_MODEL_BYTES {
            return Err(invalid("model exceeds 10 MiB file limit"));
        }
        let document: Value = serde_json::from_str(text).map_err(|e| ModelError(e.to_string()))?;
        if document["format"] == "tag-neural-model" {
            let network = Network::parse(&document)?;
            return Ok(Self {
                id: document["model_identity"]
                    .as_str()
                    .ok_or_else(|| invalid("missing digest"))?
                    .into(),
                document,
                network: Some(network),
                rows: BTreeMap::new(),
            });
        }
        if document["format"] != "tag-rust-tictactoe-q"
            || !matches!(document["version"].as_u64(), Some(1 | 2))
        {
            return Err(invalid(
                "expected a Q-table or portable neural evaluation JSON",
            ));
        }
        let task = &document["task"];
        if task["game"] != "tictactoe"
            || task["parameters"] != json!({})
            || task["opponent_settings"] != json!({})
            || !(task["learner"].is_null() || matches!(task["learner"].as_u64(), Some(0 | 1)))
            || (document["version"] == 1 && task["learner"].is_null())
        {
            return Err(invalid(
                "Q-table requires its original default Tic-Tac-Toe task",
            ));
        }
        for (field, lower) in [("alpha", false), ("gamma", true)] {
            let n = document[field]
                .as_f64()
                .ok_or_else(|| invalid("missing learning rate/discount"))?;
            if !(0.0..=1.0).contains(&n) || (!lower && n == 0.) {
                return Err(invalid("invalid learning rate/discount"));
            }
        }
        let records = document["rows"]
            .as_array()
            .ok_or_else(|| invalid("missing table rows"))?;
        if records.len() > 19683 {
            return Err(invalid("Q-table exceeds possible nine-cell keys"));
        }
        let mut rows = BTreeMap::new();
        for row in records {
            let cells = row["cells"]
                .as_array()
                .ok_or_else(|| invalid("invalid table key"))?;
            if cells.len() != 9 {
                return Err(invalid("Q-table key requires nine cells"));
            }
            let key = cells
                .iter()
                .map(|v| {
                    v.as_i64()
                        .filter(|n| (-1..=1).contains(n))
                        .map(|n| n as i8)
                        .ok_or_else(|| invalid("invalid cell code"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let values = row["values"]
                .as_array()
                .ok_or_else(|| invalid("invalid Q values"))?;
            if values.len() != 9 {
                return Err(invalid("Q-table requires nine values"));
            }
            let values = values
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|n| n.is_finite())
                        .ok_or_else(|| invalid("nonfinite Q value"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if rows.insert(key, values).is_some() {
                return Err(invalid("duplicate table key"));
            }
        }
        let id = hash::sha256(&json_bytes(&document));
        Ok(Self {
            document,
            id,
            network: None,
            rows,
        })
    }
    /// Borrow the immutable content identity; loading another model cannot change it.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Return whether this model supplies neural successor values.
    pub fn is_neural(&self) -> bool {
        self.network.is_some()
    }
    /// Return an owned copy of the portable document for a saved model bundle.
    pub fn document(&self) -> Value {
        self.document.clone()
    }
    /// Borrow the portable document without copying its weights or metadata.
    /// Use `document` when a caller needs an independently mutable copy.
    pub fn as_document(&self) -> &Value {
        &self.document
    }
    /// Borrow the network implementation, if this is a neural artifact.
    pub fn network(&self) -> Option<&Network> {
        self.network.as_ref()
    }
    /// Check a borrowed Rust game description before binding this model to a controller.
    ///
    /// # Arguments
    /// `self` borrows immutable weights; `spec` borrows the actual game's input/action meanings.
    ///
    /// # Returns
    /// Unit when the encoding/configuration and selected model contract match.
    ///
    /// # Errors
    /// Different game, rules, configuration, action mapping or tensor descriptions.
    pub fn validate_spec(&self, spec: &LearningSpec) -> Result<(), ModelError> {
        if let Some(network) = &self.network {
            return network.validate_spec(spec);
        }
        if spec.identity.game != "tictactoe"
            || spec.identity.width() != 3
            || spec.identity.height() != 3
            || spec.identity.target() != 3
            || !self.table_identity_matches(&spec.identity)
        {
            return Err(invalid("Q-table encoding differs from game"));
        }
        Ok(())
    }
    fn table_identity_matches(&self, source: &EncodingIdentity) -> bool {
        let mut saved = self.document["identity"].clone();
        if let Some(fields) = saved.as_object_mut() {
            fields.entry("player_count").or_insert(json!(2));
            fields.entry("configuration").or_insert(json!([]));
        }
        saved == identity(source, true)
    }
    /// Borrow a permitted current decision and return its stored action values; absent rows are zero.
    ///
    /// # Arguments
    /// `self` borrows a Q-table; `observation` borrows the learner-relative cells.
    ///
    /// # Returns
    /// An independent nine-entry float64 vector; callers restrict selection to legal actions.
    ///
    /// # Errors
    /// A neural model, mismatched encoding, missing cells or invalid cell values.
    pub fn q_values(&self, observation: &EncodedObservation) -> Result<Vec<f64>, ModelError> {
        if self.is_neural() || !self.table_identity_matches(&observation.identity) {
            return Err(invalid("Q-table observation mismatch"));
        }
        let cells = observation
            .tensors
            .iter()
            .find(|t| t.name == "cells")
            .ok_or_else(|| invalid("Q-table requires cells"))?;
        if cells.values.len() != 9 || cells.values.iter().any(|v| ![-1., 0., 1.].contains(v)) {
            return Err(invalid("invalid Q-table cells"));
        }
        let key: Vec<_> = cells.values.iter().map(|v| *v as i8).collect();
        Ok(self.rows.get(&key).cloned().unwrap_or_else(|| vec![0.; 9]))
    }
}
