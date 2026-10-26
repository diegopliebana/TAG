//! Resolve browser setup seeds once, independently of any selected game.
use crate::{protocol::BridgeError, runtime::integer};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

fn invalid(message: impl std::fmt::Display) -> BridgeError {
    BridgeError::new("invalid_setup", message.to_string())
}

fn automatic_seed() -> Result<u64, BridgeError> {
    #[cfg(target_arch = "wasm32")]
    {
        #[wasm_bindgen::prelude::wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(catch, js_namespace=crypto, js_name=getRandomValues)]
            fn fill(bytes: &mut [u8]) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
        }
        let mut bytes = [0; 8];
        fill(&mut bytes).map_err(|_| invalid("Web Crypto seed generation failed"))?;
        Ok(u64::from_le_bytes(bytes))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        tag_rust::core::random::automatic_seed().map_err(invalid)
    }
}

/// Decode a stochastic game's parameters after replacing its optional decimal
/// seed with one resolved u64. Game-specific serde validation remains unchanged.
pub(crate) fn parameters<T: DeserializeOwned>(value: Option<Value>) -> Result<T, BridgeError> {
    let mut value = value.unwrap_or_else(|| json!({}));
    if !value.is_object() {
        return Err(invalid("game parameters must be an object"));
    }
    let seed = match value.get("seed") {
        None | Some(Value::Null) => automatic_seed()?,
        Some(Value::String(seed)) if seed == "auto" => automatic_seed()?,
        Some(Value::String(seed)) => integer(seed)?,
        _ => {
            return Err(invalid(
                "game seed requires a decimal string or null for automatic seeding",
            ));
        }
    };
    value["seed"] = json!(seed);
    serde_json::from_value(value).map_err(invalid)
}
