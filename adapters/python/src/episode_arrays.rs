//! Bound and stack ordinary/intermediate episode outputs before Python conversion.
use crate::{
    arrays,
    episode::{Output, check_capacity},
    episode_binding,
};
use numpy::{
    PyArray, PyArray1,
    ndarray::{ArrayD, IxDyn},
};
use pyo3::{
    exceptions::{PyMemoryError, PyValueError},
    prelude::*,
    types::PyDict,
};
use tag_rust::learning::{EncodedObservation, LearningSpec};

/// Preflight all numeric rows, including optional intermediate arrays.
/// `spec`/`names` borrow fixed descriptions; `count` includes inactive members.
/// `max_bytes` bounds final numeric data; `intermediate` reserves a second set.
/// Returns unit or an overflow/size error before any game is mutated.
pub(super) fn capacity(
    spec: &LearningSpec,
    names: &[String],
    count: usize,
    max_bytes: usize,
    intermediate: bool,
) -> PyResult<()> {
    let count = count
        .checked_mul(if intermediate { 2 } else { 1 })
        .ok_or_else(|| PyMemoryError::new_err("batch size overflow"))?;
    check_capacity(spec, names, count, max_bytes).map_err(PyMemoryError::new_err)
}

struct Stacked {
    payloads: Vec<(&'static str, ArrayD<f32>)>,
    masks: ArrayD<i8>,
}
/// Read optional observation rows and return owned arrays with zero placeholders.
/// `spec`/`names` define the checked layout; `rows` borrows independent outputs.
/// Returns stacked arrays or a layout error; no observation or game is changed.
fn stack(
    spec: &LearningSpec,
    names: &[String],
    rows: &[Option<&EncodedObservation>],
) -> Result<Stacked, String> {
    let count = rows.len();
    let mut payloads = Vec::with_capacity(names.len());
    for name in names {
        let tensor = spec
            .payloads
            .iter()
            .find(|p| p.name() == name)
            .and_then(|p| p.tensor())
            .ok_or("batch payload missing")?;
        let elements = tensor.shape.iter().product::<usize>();
        let mut data = vec![0f32; count * elements];
        for (index, row) in rows.iter().enumerate() {
            if let Some(row) = row {
                let source = row
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .ok_or("encoded payload missing")?;
                if source.shape != tensor.shape || source.values.len() != elements {
                    return Err("episode layout changed".into());
                }
                data[index * elements..(index + 1) * elements].copy_from_slice(&source.values);
            }
        }
        let mut shape = vec![count];
        shape.extend_from_slice(&tensor.shape);
        payloads.push((
            tensor.name,
            ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| e.to_string())?,
        ));
    }
    let mut masks = vec![0i8; count * spec.actions.size];
    for (index, row) in rows.iter().enumerate() {
        if let Some(row) = row {
            if row.action_mask.len() != spec.actions.size {
                return Err("episode mask layout changed".into());
            }
            for (dst, src) in masks[index * spec.actions.size..(index + 1) * spec.actions.size]
                .iter_mut()
                .zip(&row.action_mask)
            {
                *dst = i8::from(*src);
            }
        }
    }
    Ok(Stacked {
        payloads,
        masks: ArrayD::from_shape_vec(IxDyn(&[count, spec.actions.size]), masks)
            .map_err(|e| e.to_string())?,
    })
}

fn arrays<'py>(py: Python<'py>, stacked: Stacked) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (name, data) in stacked.payloads {
        out.set_item(name, PyArray::from_owned_array(py, data))?;
    }
    out.set_item("action_mask", PyArray::from_owned_array(py, stacked.masks))?;
    Ok(out)
}

/// Consume owned `rows` and return full-member arrays, metadata and errors.
/// `py` permits Python allocation after detached stacking; `spec` and `names`
/// borrow retained descriptions. `errors`/`statuses` transfer per-member diagnostics;
/// `intermediate` controls additional output allocations. Failed rows are None.
/// Layout/conversion errors return no result; callers enforce failed-step recovery.
pub(super) fn convert<'py>(
    py: Python<'py>,
    spec: &LearningSpec,
    names: &[String],
    mut rows: Vec<Option<Output>>,
    errors: Vec<Option<String>>,
    statuses: Vec<String>,
    intermediate: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (ordinary, extra, mut views) = py
        .detach(|| {
            let ordinary = stack(
                spec,
                names,
                &rows
                    .iter()
                    .map(|r| r.as_ref().map(|r| &r.observation))
                    .collect::<Vec<_>>(),
            )?;
            let views = rows
                .iter_mut()
                .map(|r| r.as_mut().and_then(|r| r.intermediate.take()))
                .collect::<Vec<_>>();
            let extra = if intermediate {
                Some(stack(
                    spec,
                    names,
                    &views.iter().map(Option::as_ref).collect::<Vec<_>>(),
                )?)
            } else {
                None
            };
            for row in rows.iter_mut().flatten() {
                row.observation.tensors.clear();
                row.observation.action_mask.clear();
                row.observation.offered_action_ids.clear();
            }
            Ok::<_, String>((ordinary, extra, views))
        })
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("observations", arrays(py, ordinary)?)?;
    out.set_item(
        "rewards",
        PyArray1::from_vec(
            py,
            rows.iter()
                .map(|r| r.as_ref().map_or(0., |r| r.reward))
                .collect(),
        ),
    )?;
    out.set_item(
        "terminated",
        PyArray1::from_vec(
            py,
            rows.iter()
                .map(|r| r.as_ref().is_some_and(|r| r.terminated))
                .collect(),
        ),
    )?;
    out.set_item(
        "truncated",
        PyArray1::from_vec(
            py,
            rows.iter()
                .map(|r| r.as_ref().is_some_and(|r| r.truncated))
                .collect(),
        ),
    )?;
    out.set_item(
        "valid",
        PyArray1::from_vec(py, rows.iter().map(Option::is_some).collect()),
    )?;
    if let Some(extra) = extra {
        out.set_item("intermediate_observations", arrays(py, extra)?)?;
        out.set_item(
            "intermediate_valid",
            PyArray1::from_vec(py, views.iter().map(Option::is_some).collect()),
        )?;
        let mut infos = Vec::with_capacity(views.len());
        for view in &mut views {
            infos.push(if let Some(view) = view.take() {
                arrays::observation_metadata(py, &view)?.into_any().unbind()
            } else {
                py.None()
            });
        }
        out.set_item("intermediate_infos", infos)?;
    } else {
        for name in [
            "intermediate_observations",
            "intermediate_valid",
            "intermediate_infos",
        ] {
            out.set_item(name, py.None())?;
        }
    }
    let mut infos = Vec::with_capacity(rows.len());
    for row in rows {
        infos.push(if let Some(row) = row {
            episode_binding::convert_info(py, row)?.into_any().unbind()
        } else {
            py.None()
        });
    }
    out.set_item("infos", infos)?;
    out.set_item("errors", errors)?;
    out.set_item("statuses", statuses)?;
    Ok(out)
}
