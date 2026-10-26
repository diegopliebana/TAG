//! Convert already-owned Rust output into Python descriptions and NumPy arrays.
use numpy::{
    PyArray, PyArray1,
    ndarray::{ArrayD, IxDyn},
};
use pyo3::{
    exceptions::{PyKeyError, PyOverflowError, PyRuntimeError},
    prelude::*,
    types::PyDict,
};
use tag_rust::learning::*;

/// Read one required field from a Python dictionary, reporting its missing name.
pub(crate) fn field<'py>(dict: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(name)?
        .ok_or_else(|| PyKeyError::new_err(name.to_owned()))
}

fn dtype(value: DType) -> &'static str {
    match value {
        DType::Float32 => "float32",
        DType::Float64 => "float64",
        DType::Int8 => "int8",
        DType::Int64 => "int64",
        DType::Uint8 => "uint8",
        DType::Bool => "bool",
    }
}

pub(crate) fn identity<'py>(
    py: Python<'py>,
    value: EncodingIdentity,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("game", value.game)?;
    out.set_item("rules_version", value.rules_version)?;
    out.set_item("encoding_version", value.encoding_version)?;
    out.set_item("width", value.width())?;
    out.set_item("height", value.height())?;
    out.set_item("target", value.target())?;
    out.set_item("player_count", value.player_count)?;
    if let EncodingConfiguration::Named(settings) = value.configuration {
        out.set_item("configuration", settings)?;
    }
    Ok(out)
}

fn tensor_spec<'py>(py: Python<'py>, value: TensorSpec) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("name", value.name)?;
    out.set_item("dtype", dtype(value.dtype))?;
    out.set_item("shape", value.shape)?;
    out.set_item("axes", value.axes)?;
    out.set_item("components", value.components)?;
    out.set_item("bounds", value.bounds)?;
    Ok(out)
}

/// Convert a spec returned by Rust; heuristic groups remain separately named evaluator support.
pub fn spec<'py>(py: Python<'py>, value: LearningSpec) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("identity", identity(py, value.identity)?)?;
    let mut payloads = Vec::new();
    for payload in value.payloads {
        let entry = PyDict::new(py);
        match payload {
            PayloadSpec::Features(t) => {
                entry.set_item("kind", "features")?;
                entry.set_item("tensor", tensor_spec(py, t)?)?;
            }
            PayloadSpec::Grid(t) => {
                entry.set_item("kind", "grid")?;
                entry.set_item("tensor", tensor_spec(py, t)?)?;
            }
            PayloadSpec::Image(image) => {
                entry.set_item("kind", "image")?;
                entry.set_item("tensor", tensor_spec(py, image.pixels)?)?;
                entry.set_item("colour_space", image.colour_space)?;
                entry.set_item("value_range", image.value_range)?;
                entry.set_item("renderer_identity", image.renderer_identity)?;
                entry.set_item("render_settings_identity", image.render_settings_identity)?;
            }
            PayloadSpec::Graph(graph) => {
                entry.set_item("kind", "graph")?;
                entry.set_item("name", graph.name)?;
                entry.set_item("node_features", tensor_spec(py, graph.node_features)?)?;
                entry.set_item("edge_features", tensor_spec(py, graph.edge_features)?)?;
                entry.set_item("global_features", tensor_spec(py, graph.global_features)?)?;
                entry.set_item("edge_index", tensor_spec(py, graph.edge_index)?)?;
                entry.set_item("directed", graph.directed)?;
                entry.set_item("batching", graph.batching)?;
            }
        }
        payloads.push(entry.unbind());
    }
    out.set_item("payloads", payloads)?;
    let metadata = value
        .metadata
        .into_iter()
        .map(|t| tensor_spec(py, t).map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("metadata", metadata)?;
    let actions = PyDict::new(py);
    actions.set_item("size", value.actions.size)?;
    actions.set_item(
        "mapping",
        match value.actions.mapping {
            ActionMapping::CellXMajor => "cell_x_major",
            ActionMapping::Column => "column",
            ActionMapping::Indexed => "indexed",
            ActionMapping::Candidates => "candidates",
        },
    )?;
    out.set_item("actions", actions)?;
    if let Some(lines) = value.heuristic {
        let heuristic = PyDict::new(py);
        heuristic.set_item("version", lines.version)?;
        heuristic.set_item("window_length", lines.window_length)?;
        heuristic.set_item("indices", PyArray1::from_vec(py, lines.indices))?;
        out.set_item("heuristic", heuristic)?;
    } else {
        out.set_item("heuristic", py.None())?;
    }
    Ok(out)
}

/// Copy player-visible metadata from borrowed `value` into Python-owned objects.
///
/// # Arguments
/// `py` permits Python allocation. `value` borrows an encoded player observation.
///
/// # Returns
/// A dictionary owning identity, player IDs, progress and result arrays.
///
/// # Errors
/// Python allocation or dictionary construction can fail.
pub(crate) fn observation_metadata<'py>(
    py: Python<'py>,
    value: &EncodedObservation,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("identity", identity(py, value.identity.clone())?)?;
    out.set_item("observer", value.observer.index())?;
    out.set_item("next_player", value.next_player.map(|p| p.index()))?;
    out.set_item("progress", PyArray1::from_vec(py, value.progress.to_vec()))?;
    out.set_item(
        "results",
        PyArray1::from_vec(py, value.results.iter().map(|v| *v as i8).collect()),
    )?;
    Ok(out)
}

/// Transfer encoded allocations to Python-owned objects, independently of the live state.
pub fn observation<'py>(
    py: Python<'py>,
    value: EncodedObservation,
) -> PyResult<Bound<'py, PyDict>> {
    let out = observation_metadata(py, &value)?;
    let payloads = PyDict::new(py);
    for tensor in value.tensors {
        let data = ArrayD::from_shape_vec(IxDyn(&tensor.shape), tensor.values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        payloads.set_item(tensor.name, PyArray::from_owned_array(py, data))?;
    }
    out.set_item("payloads", payloads)?;
    out.set_item("action_mask", PyArray1::from_vec(py, value.action_mask))?;
    let ids = value
        .offered_action_ids
        .into_iter()
        .map(|id| {
            i64::try_from(id).map_err(|_| PyOverflowError::new_err("action ID exceeds int64"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("offered_action_ids", PyArray1::from_vec(py, ids))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_description_crosses_python_without_board_or_heuristic() {
        Python::initialize();
        Python::attach(|py| {
            let settings = std::collections::BTreeMap::from([("deck".into(), "example-v1".into())]);
            let description = LearningSpec {
                identity: EncodingIdentity {
                    game: "synthetic-card-fixture",
                    rules_version: 2,
                    encoding_version: 3,
                    configuration: EncodingConfiguration::Named(settings.clone()),
                    player_count: 4,
                },
                payloads: vec![PayloadSpec::Features(TensorSpec {
                    name: "own_hand",
                    dtype: DType::Float32,
                    shape: vec![8, 3],
                    axes: vec!["slot", "feature"],
                    components: vec![],
                    bounds: None,
                })],
                metadata: vec![],
                actions: ActionSpec {
                    size: 0,
                    mapping: ActionMapping::Candidates,
                },
                heuristic: None,
            };
            let converted = spec(py, description).unwrap();
            assert!(field(&converted, "heuristic").unwrap().is_none());
            let identity = field(&converted, "identity")
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            assert_eq!(
                field(&identity, "player_count")
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                4
            );
            assert_eq!(
                field(&identity, "configuration")
                    .unwrap()
                    .extract::<std::collections::BTreeMap<String, String>>()
                    .unwrap(),
                settings
            );
            let actions = field(&converted, "actions")
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            assert_eq!(
                field(&actions, "mapping")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "candidates"
            );
        });
    }
}
