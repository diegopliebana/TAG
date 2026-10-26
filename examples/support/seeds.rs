//! Resolve optional example seeds at the application boundary.
pub fn resolve_seed(value: Option<&str>) -> Result<u64, Box<dyn std::error::Error>> {
    match value {
        None | Some("auto" | "null") => Ok(tag_rust::core::random::automatic_seed()?),
        Some(value) => Ok(value.parse()?),
    }
}
