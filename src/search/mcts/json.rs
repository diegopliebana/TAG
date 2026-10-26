//! JSON text conversion. File reading remains with the application or adapter.
use super::{BudgetOverrides, Error, Override, Result, Settings};
use serde::Deserialize;

impl Settings {
    /// Read a versioned configuration from borrowed JSON text and validate all settings.
    pub fn from_json(text: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| Error::Invalid(e.to_string()))?;
        if value.get("schema_version").is_none() {
            return Err(Error::Invalid("schema_version is required".into()));
        }
        let settings: Self =
            serde_json::from_str(text).map_err(|e| Error::Invalid(e.to_string()))?;
        // Serde accepts null for optional fields. Only the explicitly disableable
        // settings and seed allow it; mode-specific options use omission instead.
        reject_other_nulls(&value, "")?;
        settings.validate()?;
        Ok(settings)
    }
    /// Serialize effective settings; constructed agents have a resolved integer seed.
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        let mut value = serde_json::to_value(self).map_err(|e| Error::Invalid(e.to_string()))?;
        remove_mode_nulls(&mut value, "");
        serde_json::to_string_pretty(&value).map_err(|e| Error::Invalid(e.to_string()))
    }
}
fn nullable(path: &str) -> bool {
    matches!(
        path,
        "seed"
            | "tree_depth"
            | "rollout_length"
            | "budgets.time_ms"
            | "budgets.iterations"
            | "budgets.forward_model_calls"
    ) || path.contains("parameters.")
}
fn reject_other_nulls(value: &serde_json::Value, prefix: &str) -> Result<()> {
    if let serde_json::Value::Object(object) = value {
        for (key, value) in object {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            if value.is_null() && !nullable(&path) {
                return Err(Error::Invalid(format!("null is not allowed for {path}")));
            }
            reject_other_nulls(value, &path)?;
        }
    }
    Ok(())
}
fn remove_mode_nulls(value: &mut serde_json::Value, prefix: &str) {
    if let serde_json::Value::Object(object) = value {
        object.retain(|key, value| {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            if value.is_null() && !nullable(&path) {
                return false;
            }
            remove_mode_nulls(value, &path);
            true
        });
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Override<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Ok(match Option::<T>::deserialize(deserializer)? {
            Some(value) => Self::Replace(value),
            None => Self::Disable,
        })
    }
}
impl BudgetOverrides {
    /// Read an override while preserving the difference between omission and null.
    pub fn from_json(text: &str) -> Result<Self> {
        #[derive(Default, Deserialize)]
        #[serde(default, deny_unknown_fields)]
        struct Fields {
            time_ms: Override<f64>,
            iterations: Override<u64>,
            forward_model_calls: Override<u64>,
        }
        let fields: Fields =
            serde_json::from_str(text).map_err(|e| Error::Invalid(e.to_string()))?;
        Ok(Self {
            time_ms: fields.time_ms,
            iterations: fields.iterations,
            forward_model_calls: fields.forward_model_calls,
        })
    }
}
