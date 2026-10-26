//! Construct an agent or restore its saved state between decisions.
use super::*;

impl<F> MctsAgent<F>
where
    F: SimulationModel + 'static,
    F::State: Clone,
    F::Action: Clone + PartialEq,
    F::Error: std::fmt::Display,
{
    /// Build independent evaluators and resolve an automatic seed once.
    ///
    /// # Arguments
    /// `settings` is consumed. `registry` is borrowed only while its factories run.
    /// Each factory transfers ownership of a new evaluator to this agent.
    pub fn new(settings: Settings, registry: &EvaluatorRegistry<F>) -> Result<Self> {
        Self::with_seed_source(settings, registry, super::seed::system_seed)
    }
    /// Construct with a platform-provided automatic seed source, for example browser crypto.
    /// Explicit seeds do not call `source`. Failures never fall back to clock seeding.
    ///
    /// # Arguments
    /// `settings` transfers ownership of configuration. `registry` is read-only.
    /// `source` is consumed and called at most once, only for an automatic seed.
    ///
    /// # Errors
    /// Invalid settings, unknown evaluators, factory failures or a failed seed source.
    pub fn with_seed_source(
        mut settings: Settings,
        registry: &EvaluatorRegistry<F>,
        source: impl FnOnce() -> Result<u64>,
    ) -> Result<Self> {
        settings.validate()?;
        let state = registry.build_state(&settings.state_evaluator)?;
        let build = |spec: &Option<EvaluatorSpec>| {
            spec.as_ref()
                .map(|spec| registry.build_action(spec))
                .transpose()
        };
        let evaluators = Evaluators {
            state,
            bias: build(&settings.progressive_bias.evaluator)?,
            preferences: build(&settings.puct_preferences.evaluator)?,
            expansion: build(&settings.expansion.evaluator)?,
            rollout: build(&settings.rollout.evaluator)?,
        };
        let seed = match settings.seed {
            Some(seed) => seed,
            None => source()?,
        };
        settings.seed = Some(seed);
        Ok(Self {
            settings,
            evaluators,
            rng: AgentRng::new(seed),
            simulation_rng: SimulationRng::new(seed),
            last_report: None,
        })
    }
    pub fn settings(&self) -> &Settings {
        &self.settings
    }
    pub fn rng(&self) -> &AgentRng {
        &self.rng
    }
    pub fn last_report(&self) -> Option<&SearchReport<F::Action>> {
        self.last_report.as_ref()
    }

    /// Save evaluator state and exact RNG progress between decisions.
    ///
    /// # Arguments
    /// `self` is read-only; saving never advances the RNG or an evaluator.
    ///
    /// # Returns
    /// An independent owned snapshot, or an explicit error if any evaluator cannot save.
    pub fn snapshot(&self) -> Result<MctsSnapshot> {
        let save = |evaluator: &Option<Box<dyn ActionEvaluator<F>>>| {
            evaluator.as_ref().map(|e| e.checkpoint()).transpose()
        };
        Ok(MctsSnapshot {
            algorithm_version: 1,
            settings: self.settings.clone(),
            rng: self.rng.snapshot(),
            simulation_rng: self.simulation_rng.snapshot(),
            state_evaluator: self.evaluators.state.checkpoint()?,
            action_evaluators: [
                save(&self.evaluators.bias)?,
                save(&self.evaluators.preferences)?,
                save(&self.evaluators.expansion)?,
                save(&self.evaluators.rollout)?,
            ],
        })
    }
    /// Restore independent evaluator instances through the caller's compatible registry.
    ///
    /// # Arguments
    /// `snapshot` and `registry` are read-only. Their owned data remains with the caller.
    ///
    /// # Errors
    /// Unsupported versions, inconsistent seed/slots, or evaluator restoration failures.
    pub fn from_snapshot(snapshot: &MctsSnapshot, registry: &EvaluatorRegistry<F>) -> Result<Self> {
        if snapshot.algorithm_version != 1
            || snapshot.settings.seed != Some(snapshot.rng.initial_seed)
            || snapshot.settings.seed != Some(snapshot.simulation_rng.generator.initial_seed)
        {
            return Err(Error::Invalid(
                "checkpoint algorithm version or resolved seed mismatch".into(),
            ));
        }
        let mut agent = Self::new(snapshot.settings.clone(), registry)?;
        agent.evaluators.state.restore(&snapshot.state_evaluator)?;
        let slots = [
            &mut agent.evaluators.bias,
            &mut agent.evaluators.preferences,
            &mut agent.evaluators.expansion,
            &mut agent.evaluators.rollout,
        ];
        for (slot, bytes) in slots.into_iter().zip(&snapshot.action_evaluators) {
            match (slot.as_mut(), bytes) {
                (Some(evaluator), Some(bytes)) => evaluator.restore(bytes)?,
                (None, None) => {}
                _ => {
                    return Err(Error::Invalid(
                        "checkpoint evaluator slots differ from settings".into(),
                    ));
                }
            }
        }
        agent.rng =
            AgentRng::from_snapshot(&snapshot.rng).map_err(|e| Error::Invalid(e.to_string()))?;
        agent.simulation_rng = SimulationRng::from_snapshot(&snapshot.simulation_rng)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        Ok(agent)
    }
}
