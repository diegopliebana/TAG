//! Bounded float32 fully connected reference model, with the Python format-2 context.
use super::{ModelError, hash, identity, invalid, json_bytes};
use crate::learning::{ActionMapping, DType, EncodedObservation, LearningSpec};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct Layer {
    rows: usize,
    columns: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}
impl Layer {
    fn run(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .chunks_exact(self.columns)
            .zip(&self.bias)
            .map(|(w, b)| {
                // Separate multiplication/addition preserves the declared CPU reference order.
                let mut value = 0f32;
                for (w, x) in w.iter().zip(input) {
                    value += w * x;
                }
                value + b
            })
            .collect()
    }
}

/// Immutable reference network; inputs follow the declared named tensors, then public context.
#[derive(Clone, Debug)]
pub struct Network {
    contract: Value,
    names: Vec<String>,
    input_size: usize,
    hidden: Vec<Layer>,
    policy: Layer,
    value: Layer,
    parameters: usize,
    value_players: Option<Vec<usize>>,
}
fn positive(v: &Value) -> Result<usize, ModelError> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("expected positive model dimension"))
}
fn frame(bytes: &mut Vec<u8>, data: &[u8]) {
    bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
    bytes.extend_from_slice(data);
}
impl Network {
    pub(super) fn parse(doc: &Value) -> Result<Self, ModelError> {
        if doc["version"] != 1 || doc["implementation"] != "tag_rust.PolicyValueNetwork/2" {
            return Err(invalid(
                "unsupported reference model version/implementation",
            ));
        }
        let contract = doc["contract"].clone();
        if contract.as_object().is_none_or(|fields| fields.len() != 6)
            || contract["feature_version"] != 2
            || !contract["candidate_tensor"].is_null()
        {
            return Err(invalid(
                "reference browser requires format 2, fixed policy and observer win/draw/loss values",
            ));
        }
        let players = positive(&contract["identity"]["player_count"])?;
        if players > 256 {
            return Err(invalid("player context exceeds 256 seats"));
        }
        let value = &contract["value"];
        let value_players = if value["players"].is_null() {
            None
        } else {
            let order = value["players"]
                .as_array()
                .ok_or_else(|| invalid("invalid player value order"))?
                .iter()
                .map(|p| {
                    p.as_u64()
                        .and_then(|p| usize::try_from(p).ok())
                        .ok_or_else(|| invalid("invalid value player ID"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut sorted = order.clone();
            sorted.sort_unstable();
            if sorted != (0..players).collect::<Vec<_>>() {
                return Err(invalid(
                    "vector output must identify every player exactly once",
                ));
            }
            Some(order)
        };
        let legacy = value_players.is_none()
            && value["gamma"] == 0.99
            && value["discount_clock"] == "learner-step";
        let selfplay = value["gamma"] == 1.0 && value["discount_clock"] == "environment-action";
        if value.as_object().is_none_or(|v| v.len() != 5)
            || value["bounds"] != json!([-1.0, 1.0])
            || value["units"] != "win=1,draw=0,loss=-1"
            || !(legacy || selfplay)
        {
            return Err(invalid(
                "unsupported value contract: expected discounted observer values or gamma=1 self-play outcomes",
            ));
        }
        let tensors = contract["tensors"]
            .as_array()
            .ok_or_else(|| invalid("missing named input descriptions"))?;
        if tensors.is_empty() || tensors.len() > 16 {
            return Err(invalid("select 1-16 named inputs"));
        }
        let mut input_size = 2 * players + 1;
        let mut names = Vec::new();
        for tensor in tensors {
            let name = tensor["name"]
                .as_str()
                .ok_or_else(|| invalid("missing input name"))?;
            if name.is_empty() || names.iter().any(|n| n == name) || tensor["dtype"] != "<f4" {
                return Err(invalid("expected unique float32 named inputs"));
            }
            let shape = tensor["shape"]
                .as_array()
                .ok_or_else(|| invalid("missing input shape"))?;
            if shape.is_empty() || shape.len() > 8 {
                return Err(invalid("input rank must be 1-8"));
            }
            let size = shape.iter().try_fold(1usize, |n, v| {
                n.checked_mul(positive(v)?)
                    .ok_or_else(|| invalid("input shape overflow"))
            })?;
            input_size = input_size
                .checked_add(size)
                .ok_or_else(|| invalid("input size overflow"))?;
            // Love Letter includes bounded current/previous event transcripts.
            // Keep a finite input cap alongside file and parameter-count limits.
            if input_size > 16_384 {
                return Err(invalid("reference input exceeds 16384 features"));
            }
            names.push(name.into());
        }
        let widths = doc["hidden_sizes"]
            .as_array()
            .ok_or_else(|| invalid("missing hidden sizes"))?;
        if widths.is_empty() || widths.len() > 8 {
            return Err(invalid("select 1-8 hidden layers"));
        }
        let widths = widths.iter().map(positive).collect::<Result<Vec<_>, _>>()?;
        if widths.iter().any(|n| *n > 512) {
            return Err(invalid("hidden layers exceed 512 units"));
        }
        let actions = positive(&contract["actions"]["size"])?;
        if actions > 4096
            || !matches!(
                contract["actions"]["mapping"].as_str(),
                Some("cell_x_major" | "column" | "indexed")
            )
        {
            return Err(invalid("unsupported fixed policy size/mapping"));
        }
        let weights = doc["weights"]
            .as_object()
            .ok_or_else(|| invalid("missing weights"))?;
        let mut specifications = BTreeMap::new();
        let mut previous = input_size;
        for (index, width) in widths.iter().enumerate() {
            specifications.insert(format!("features.{}", index * 2), (*width, previous));
            previous = *width;
        }
        specifications.insert("policy".into(), (actions, previous));
        specifications.insert(
            "value".into(),
            (value_players.as_ref().map_or(1, Vec::len), previous),
        );
        let parameter_count = specifications.values().try_fold(0usize, |n, (r, c)| {
            n.checked_add(r * (c + 1))
                .ok_or_else(|| invalid("parameter count overflow"))
        })?;
        if parameter_count > 1_000_000 || weights.len() != specifications.len() * 2 {
            return Err(invalid(
                "unexpected weights or more than one million parameters",
            ));
        }
        let mut digest_data = Vec::new();
        let description = json!({"version":1,"implementation":"tag_rust.PolicyValueNetwork/2","settings":{"hidden_sizes":widths},"contract":contract});
        frame(&mut digest_data, &json_bytes(&description));
        let mut decoded = BTreeMap::new();
        for (name, weight) in weights {
            let (prefix, suffix) = name
                .rsplit_once('.')
                .ok_or_else(|| invalid("invalid weight name"))?;
            let (rows, columns) = specifications
                .get(prefix)
                .ok_or_else(|| invalid("unknown weight name"))?;
            let shape = match suffix {
                "weight" => vec![*rows, *columns],
                "bias" => vec![*rows],
                _ => return Err(invalid("unknown parameter kind")),
            };
            if weight["dtype"] != "float32" || weight["shape"] != json!(shape) {
                return Err(invalid("weight type/shape mismatch"));
            }
            let values = weight["values"]
                .as_array()
                .ok_or_else(|| invalid("missing flat weights"))?;
            if values.len() != shape.iter().product::<usize>() {
                return Err(invalid("weight count mismatch"));
            }
            let values = values
                .iter()
                .map(|v| {
                    let n = v
                        .as_f64()
                        .ok_or_else(|| invalid("invalid numeric weight"))?;
                    let f = n as f32;
                    if !f.is_finite() || f as f64 != n {
                        return Err(invalid("weights must be exact finite float32 values"));
                    }
                    Ok(f)
                })
                .collect::<Result<Vec<_>, _>>()?;
            frame(
                &mut digest_data,
                &json_bytes(&json!({"name":name,"dtype":"torch.float32","shape":shape})),
            );
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            frame(&mut digest_data, &bytes);
            decoded.insert(name.clone(), values);
        }
        if doc["model_identity"].as_str() != Some(&hash::sha256(&digest_data)) {
            return Err(invalid("model digest mismatch"));
        }
        let mut layer = |name: &str| -> Result<Layer, ModelError> {
            let (rows, columns) = specifications[name];
            Ok(Layer {
                rows,
                columns,
                weights: decoded
                    .remove(&format!("{name}.weight"))
                    .ok_or_else(|| invalid("missing weight"))?,
                bias: decoded
                    .remove(&format!("{name}.bias"))
                    .ok_or_else(|| invalid("missing bias"))?,
            })
        };
        let hidden = (0..widths.len())
            .map(|i| layer(&format!("features.{}", i * 2)))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            contract,
            names,
            input_size,
            hidden,
            policy: layer("policy")?,
            value: layer("value")?,
            parameters: parameter_count,
            value_players,
        })
    }
    /// Check the complete borrowed Rust description against the model contract.
    ///
    /// # Arguments
    /// `self` borrows immutable model metadata; `spec` borrows the game's description.
    ///
    /// # Returns
    /// Unit after matching configuration, action mapping and each selected tensor.
    ///
    /// # Errors
    /// Unsupported action representation or different encoding/tensor meanings.
    pub fn validate_spec(&self, spec: &LearningSpec) -> Result<(), ModelError> {
        let mapping = match spec.actions.mapping {
            ActionMapping::CellXMajor => "cell_x_major",
            ActionMapping::Column => "column",
            ActionMapping::Indexed => "indexed",
            _ => return Err(invalid("candidate policies need another backend")),
        };
        if self.contract["identity"] != identity(&spec.identity, true)
            || self.contract["actions"] != json!({"size":spec.actions.size,"mapping":mapping})
        {
            return Err(invalid("model game/configuration/action identity mismatch"));
        }
        for tensor in self.contract["tensors"].as_array().unwrap() {
            let t = spec
                .payloads
                .iter()
                .find(|p| Some(p.name()) == tensor["name"].as_str())
                .and_then(|p| p.tensor())
                .ok_or_else(|| invalid("game does not supply the requested input"))?;
            if t.dtype != DType::Float32
                || *tensor
                    != json!({"name":t.name,"dtype":"<f4","shape":t.shape,"axes":t.axes,"components":t.components,"bounds":t.bounds})
            {
                return Err(invalid(
                    "model tensor meanings differ from game description",
                ));
            }
        }
        Ok(())
    }
    /// Return the permanent policy-output count.
    pub fn action_count(&self) -> usize {
        self.policy.rows
    }
    /// Return the selected observation-array names in the model's input order.
    pub fn payloads(&self) -> Vec<&str> {
        self.names.iter().map(String::as_str).collect()
    }
    /// Return the checked parameter count for replay inference-work limits.
    pub fn parameter_count(&self) -> usize {
        self.parameters
    }
    /// Construct the network input vector from one player-visible observation.
    ///
    /// # Arguments
    /// `self` borrows the model contract; `observation` borrows independently encoded player data.
    ///
    /// # Returns
    /// A separate flat vector containing the selected observation arrays in their
    /// declared order, followed by the public player-context features.
    ///
    /// # Errors
    /// Identity/shape/bounds mismatch, missing payload, nonfinite input or invalid player context.
    pub fn inputs(&self, observation: &EncodedObservation) -> Result<Vec<f32>, ModelError> {
        if self.contract["identity"] != identity(&observation.identity, true) {
            return Err(invalid("observation identity mismatch"));
        }
        let mut input = Vec::with_capacity(self.input_size);
        for tensor in self.contract["tensors"].as_array().unwrap() {
            let t = observation
                .tensors
                .iter()
                .find(|t| Some(t.name) == tensor["name"].as_str())
                .ok_or_else(|| invalid("missing input payload"))?;
            if json!(t.shape) != tensor["shape"]
                || t.values.len() != t.shape.iter().product::<usize>()
                || t.values.iter().any(|n| !n.is_finite())
            {
                return Err(invalid("invalid input shape/values"));
            }
            if let Some(bounds) = tensor["bounds"].as_array() {
                let lower = bounds
                    .first()
                    .and_then(Value::as_f64)
                    .ok_or_else(|| invalid("invalid input bounds"))?;
                let upper = bounds
                    .get(1)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| invalid("invalid input bounds"))?;
                if t.values
                    .iter()
                    .any(|n| (*n as f64) < lower || (*n as f64) > upper)
                {
                    return Err(invalid("input values exceed declared bounds"));
                }
            }
            input.extend_from_slice(&t.values);
        }
        let players = observation.identity.player_count;
        if observation.observer.index() >= players
            || observation
                .next_player
                .is_some_and(|p| p.index() >= players)
        {
            return Err(invalid("invalid observer/next-player context"));
        }
        input.extend((0..players).map(|p| f32::from(p == observation.observer.index())));
        input.extend((0..players).map(|p| {
            f32::from(
                observation
                    .next_player
                    .is_some_and(|next| next.index() == p),
            )
        }));
        input.push(f32::from(
            observation.next_player == Some(observation.observer),
        ));
        Ok(input)
    }
    /// Evaluate one borrowed encoded row; return independent unmasked logits and observer value.
    ///
    /// # Arguments
    /// `self` borrows immutable weights; `observation` borrows this player's encoded data.
    ///
    /// # Returns
    /// Raw policy logits in a separate vector and an estimated observer value in [-1,1].
    /// Terminal reward overrides belong to the agent evaluator, which bypasses this prediction.
    ///
    /// # Errors
    /// Invalid encoded inputs or nonfinite numerical output.
    pub fn predict(&self, observation: &EncodedObservation) -> Result<(Vec<f32>, f32), ModelError> {
        if self.value_players.is_some() {
            return Err(invalid(
                "observer prediction requires a scalar model; use predict_values for a player vector",
            ));
        }
        let (logits, values) = self.predict_values(observation)?;
        Ok((logits, values[0]))
    }
    /// Predict raw policy logits and every declared value output. Vector entries
    /// follow `value_players()`; scalar output estimates this observation's player.
    pub fn predict_values(
        &self,
        observation: &EncodedObservation,
    ) -> Result<(Vec<f32>, Vec<f32>), ModelError> {
        let mut features = self.inputs(observation)?;
        for layer in &self.hidden {
            features = layer.run(&features).into_iter().map(f32::tanh).collect();
        }
        let logits = self.policy.run(&features);
        let raw = self.value.run(&features);
        // Preserve the Python reference's bounded-output arithmetic.
        let values: Vec<_> = raw
            .into_iter()
            .map(|v| -1. + 2. * (v.tanh() + 1.) / 2.)
            .collect();
        if values.iter().any(|v| !v.is_finite()) || logits.iter().any(|n| !n.is_finite()) {
            return Err(invalid("nonfinite reference prediction"));
        }
        debug_assert_eq!(logits.len(), self.policy.rows);
        Ok((logits, values))
    }
    /// Absolute player IDs for vector outputs, or None for an observer scalar.
    pub fn value_players(&self) -> Option<&[usize]> {
        self.value_players.as_deref()
    }
    /// Check the existing Neural OSLA consumer's discounted scalar convention.
    pub fn validate_osla(&self) -> Result<(), ModelError> {
        if self.value_players.is_some()
            || self.contract["value"]["discount_clock"] != "learner-step"
        {
            return Err(invalid(
                "Neural OSLA requires a discounted observer-value model",
            ));
        }
        Ok(())
    }
}
