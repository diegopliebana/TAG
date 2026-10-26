//! Arithmetic for tree scores and seeded move selection.
use super::{BiasMode, Error, Result, TreePolicy};
use crate::core::random::AgentRng;

/// Compute the tree-policy score for an action that has already been visited.
/// `mean` is Q_i(s,a) for the deciding player i; `parent_visits` is N and
/// `visits` is n. `exploration` is K, and `preference` is the PUCT prior P(a).
/// `bias` supplies the selected progressive-bias formula, h(s,a) and bias visits.
pub(super) fn tree_score(
    policy: TreePolicy,
    exploration: f64,
    parent_visits: u64,
    visits: u64,
    mean: f64,
    preference: f64,
    bias: (BiasMode, f64, f64),
) -> Result<f64> {
    let (mode, heuristic, bias_visits) = bias;
    let n = visits as f64;
    let parent = parent_visits.max(1) as f64;
    let estimate = if mode == BiasMode::TagBlend {
        // This equivalent ratio avoids overflow when bias_visits is very large.
        let beta = (1.0 / (1.0 + 3.0 * n / bias_visits)).sqrt();
        (1.0 - beta) * mean + beta * heuristic
    } else {
        mean
    };
    let bonus = match policy {
        TreePolicy::Ucb1 => exploration * (parent.ln() / n).sqrt(),
        TreePolicy::Puct => exploration * preference * parent.sqrt() / (1.0 + n),
    };
    let score = estimate
        + bonus
        + if mode == BiasMode::Additive {
            heuristic / (1.0 + n)
        } else {
            0.0
        };
    if !score.is_finite() {
        return Err(Error::Evaluation("tree selection score overflowed".into()));
    }
    Ok(score)
}
pub(super) fn validate_scores(scores: &[f64], count: usize) -> Result<()> {
    if scores.len() != count || scores.iter().any(|v| !v.is_finite()) {
        return Err(Error::Evaluation(
            "action scores must be finite and match legal actions".into(),
        ));
    }
    Ok(())
}
/// Convert heuristic scores to probabilities proportional to exp(h(s,a) / T).
/// A smaller positive temperature T favours the highest scores more strongly;
/// a larger T moves the distribution towards uniform action selection.
/// Subtracting the maximum preserves the probabilities and avoids exp overflow.
pub(super) fn softmax(scores: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if scores.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::Evaluation(
            "softmax requires actions and a positive finite temperature".into(),
        ));
    }
    validate_scores(scores, scores.len())?;
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = scores
        .iter()
        .map(|score| ((score - maximum) / temperature).exp())
        .collect();
    probabilities(&weights)
}
/// Normalise nonnegative action weights to sum to one. Reject zero total weight.
pub(super) fn probabilities(weights: &[f64]) -> Result<Vec<f64>> {
    if weights.is_empty() || weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Error::Evaluation(
            "probabilities must be finite and nonnegative".into(),
        ));
    }
    let maximum = weights.iter().copied().fold(0.0, f64::max);
    if maximum == 0.0 {
        return Err(Error::Evaluation(
            "legal probabilities have zero mass".into(),
        ));
    }
    let scaled: Vec<_> = weights.iter().map(|v| v / maximum).collect();
    let total: f64 = scaled.iter().sum();
    Ok(scaled.iter().map(|v| v / total).collect())
}
/// Choose a highest-scoring action, breaking exact ties uniformly with the agent RNG.
pub(super) fn best(scores: &[f64], rng: &mut AgentRng) -> Result<usize> {
    validate_scores(scores, scores.len())?;
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ties: Vec<_> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == maximum).then_some(i))
        .collect();
    let chosen = rng
        .index(ties.len())
        .ok_or_else(|| Error::Invalid("cannot choose from empty scores".into()))?;
    Ok(ties[chosen])
}
pub(super) fn sample(weights: &[f64], rng: &mut AgentRng) -> Result<usize> {
    let probabilities = probabilities(weights)?;
    let positive: Vec<_> = probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v > 0.0).then_some(i))
        .collect();
    if positive.len() == 1 {
        return Ok(positive[0]);
    }
    let threshold = rng.unit_interval();
    let mut cumulative = 0.0;
    for (i, value) in probabilities.iter().enumerate() {
        cumulative += value;
        if threshold < cumulative {
            return Ok(i);
        }
    }
    Ok(*positive
        .last()
        .expect("validated probabilities have positive mass"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_terms_match_independent_numeric_fixtures() {
        let none = (BiasMode::None, 0.8, 4.0);
        let base = tree_score(TreePolicy::Ucb1, 1.0, 16, 4, 0.25, 0.6, none).unwrap();
        assert!((base - 1.0825546111576977).abs() < 1e-12);
        let additive = tree_score(
            TreePolicy::Ucb1,
            1.0,
            16,
            4,
            0.25,
            0.6,
            (BiasMode::Additive, 0.8, 4.0),
        )
        .unwrap();
        assert!((additive - 1.2425546111576977).abs() < 1e-12);
        let blend = tree_score(
            TreePolicy::Ucb1,
            1.0,
            16,
            4,
            0.25,
            0.6,
            (BiasMode::TagBlend, 0.8, 4.0),
        )
        .unwrap();
        assert!((blend - 1.3575546111576977).abs() < 1e-12);
        assert!(
            (tree_score(TreePolicy::Puct, 1.0, 16, 4, 0.25, 0.6, none).unwrap() - 0.73).abs()
                < 1e-12
        );
    }

    #[test]
    fn probability_conversion_is_stable_and_rejects_invalid_mass() {
        let result = softmax(&[1.0, 0.0], 1.0).unwrap();
        assert!((result[0] - 0.7310585786300049).abs() < 1e-12);
        assert_eq!(
            softmax(&[f64::MAX, -f64::MAX], 1.0).unwrap(),
            vec![1.0, 0.0]
        );
        assert_eq!(
            probabilities(&[f64::MAX, f64::MAX]).unwrap(),
            vec![0.5, 0.5]
        );
        assert!(probabilities(&[0.0, 0.0]).is_err());
        assert!(probabilities(&[-1.0, 1.0]).is_err());
        assert!(softmax(&[1.0, f64::NAN], 1.0).is_err());
    }
}
