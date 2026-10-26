//! Inspect a portable model without Python or a browser.
use tag_rust::inference::LearnedModel;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("provide an evaluation JSON path")?;
    let model = LearnedModel::parse(&std::fs::read_to_string(path)?)?;
    println!(
        "{} {}",
        if model.is_neural() {
            "neural"
        } else {
            "q_learning"
        },
        model.id()
    );
    Ok(())
}
