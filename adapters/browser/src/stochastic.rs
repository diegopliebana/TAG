//! One controller and persistence implementation for sampled-world games.
macro_rules! stochastic_controllers {
    ($observation:ty, $action:ty, $heuristic:ty) => {
        #[derive(Clone, Debug, PartialEq)]
        pub(crate) struct Snapshot {
            setup: Value,
            rng: AgentRngSnapshot,
            simulation: SimulationRngSnapshot,
            search: Option<MctsSnapshot>,
            local: Option<LocalMctsModel>,
        }
        pub(crate) struct StochasticAgent {
            saved: Snapshot,
            rng: AgentRng,
        }
        impl Checkpoint for StochasticAgent {
            type Snapshot = Snapshot;
            type RestoreError = ModelError;
            fn snapshot(&self) -> Snapshot {
                let mut s = self.saved.clone();
                s.rng = self.rng.snapshot();
                if let Some(search) = &mut s.search {
                    search.rng = s.rng;
                }
                s
            }
            fn from_snapshot(s: &Snapshot) -> Result<Self, ModelError> {
                Ok(Self {
                    saved: s.clone(),
                    rng: AgentRng::from_snapshot(&s.rng).map_err(error)?,
                })
            }
        }
        fn registry(
            local: Option<&LocalMctsModel>,
            root: PlayerId,
        ) -> Result<tag_rust::search::mcts::EvaluatorRegistry<Game>, ModelError> {
            let mut r = Game::native_evaluators();
            if let Some(m) = local {
                m.register(&mut r, root).map_err(error)?;
            }
            Ok(r)
        }
        impl StochasticAgent {
            fn search(&self, actor: PlayerId) -> Result<MctsAgent<Game>, ModelError> {
                let mut s = self
                    .saved
                    .search
                    .clone()
                    .ok_or_else(|| error("missing search checkpoint"))?;
                s.rng = self.rng.snapshot();
                MctsAgent::from_snapshot(&s, &registry(self.saved.local.as_ref(), actor)?).map_err(error)
            }
            fn accept_search(&mut self, a: &MctsAgent<Game>) -> Result<(), ModelError> {
                let s = a.snapshot().map_err(error)?;
                self.rng = AgentRng::from_snapshot(&s.rng).map_err(error)?;
                self.saved.simulation = s.simulation_rng;
                self.saved.search = Some(s);
                Ok(())
            }
        }
        impl Agent<Game> for StochasticAgent {
            type Error = ModelError;
            fn rng(&self) -> &AgentRng {
                &self.rng
            }
            fn rng_mut(&mut self) -> &mut AgentRng {
                &mut self.rng
            }
            fn choose(
                &mut self,
                g: &Game,
                actor: PlayerId,
                o: &$observation,
                actions: &[$action],
            ) -> Result<usize, ModelError> {
                if actions.is_empty() {
                    return Err(error("empty action list"));
                }
                match self.saved.setup["kind"].as_str().unwrap_or("") {
                    "first_available" => Ok(0),
                    "random" => Ok(self.rng.index(actions.len()).expect("nonempty")),
                    "sampled_osla" => {
                        let count = self.saved.setup["repetitions_per_action"].as_u64().unwrap() as u32;
                        let mut agent =
                            SampledOslaAgent::<$heuristic>::from_snapshot(&SampledOslaSnapshot {
                                evaluator: <$heuristic>::default().snapshot(),
                                rng: self.rng.snapshot(),
                                simulation_rng: self.saved.simulation,
                                repetitions_per_action: std::num::NonZeroU32::new(count).unwrap(),
                            })
                            .map_err(error)?;
                        let choice = agent.choose(g, actor, o, actions).map_err(error)?;
                        let saved = agent.snapshot();
                        self.saved.simulation = saved.simulation_rng;
                        self.rng = AgentRng::from_snapshot(&saved.rng).map_err(error)?;
                        Ok(choice)
                    }
                    "mcts" | "neural_mcts" => {
                        let mut a = self.search(actor)?;
                        let r = a
                            .search_controlled(
                                g,
                                actor,
                                o,
                                actions,
                                &Default::default(),
                                &crate::mcts::BrowserClock::default(),
                                &|| false,
                            )
                            .map_err(error)?;
                        self.accept_search(&a)?;
                        Ok(r.selected_index)
                    }
                    _ => Err(error("unsupported stochastic game controller")),
                }
            }
            fn choose_async<'a>(
                &'a mut self,
                g: &'a Game,
                actor: PlayerId,
                o: &'a $observation,
                actions: &'a [$action],
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize, ModelError>> + 'a>> {
                Box::pin(async move {
                    if self.saved.search.is_none() {
                        return self.choose(g, actor, o, actions);
                    }
                    let mut a = self.search(actor)?;
                    let r = a
                        .search_controlled_async(
                            g,
                            actor,
                            o,
                            actions,
                            &Default::default(),
                            &crate::mcts::BrowserClock::default(),
                            &|| false,
                        )
                        .await
                        .map_err(error)?;
                    self.accept_search(&a)?;
                    Ok(r.selected_index)
                })
            }
        }
        fn controller(
            setup: SeatSetup,
            models: &Registry,
        ) -> Result<Controller<StochasticAgent>, BridgeError> {
            if matches!(setup, SeatSetup::Human) {
                return Ok(Controller::Human);
            }
            let setup = match setup {
                SeatSetup::Osla { seed } => SeatSetup::SampledOsla {
                    seed,
                    repetitions_per_action: 1,
                },
                s => s,
            };
            if GAME_ID == "loveletter" {
                if let SeatSetup::NeuralMcts { bindings, .. } = &setup {
                    if bindings.value_observers.is_some_and(|v| v != tag_rust::inference::ValuePerspective::DecidingPlayer) {
                        return Err(bridge("Love Letter requires deciding-player neural observations"));
                    }
                }
            }
            let mut value = serde_json::to_value(&setup).map_err(bridge)?;
            let seed = crate::runtime::integer(
                value["seed"]
                    .as_str()
                    .ok_or_else(|| bridge("missing seed"))?,
            )?;
            let rng = AgentRng::new(seed);
            let simulation = SimulationRng::new(seed).snapshot();
            let local = if let SeatSetup::NeuralMcts {
                model_id, bindings, ..
            } = &setup
            {
                let loaded = models
                    .get(model_id)
                    .ok_or_else(|| bridge("load the neural model first"))?;
                model_parameters(loaded).map_err(bridge)?;
                Some(LocalMctsModel::new(loaded.clone(), bindings.clone()).map_err(bridge)?)
            } else {
                None
            };
            let search = match &setup {
                SeatSetup::Mcts { configuration, .. } | SeatSetup::NeuralMcts { configuration, .. } => {
                    let mut settings = Settings::from_json(&configuration.to_string()).map_err(bridge)?;
                    settings.seed = Some(seed);
                    if GAME_ID == "loveletter" && settings.value_completion != tag_rust::search::mcts::ValueCompletion::RequireFullVector {
                        return Err(bridge("Love Letter requires full absolute-seat value vectors"));
                    }
                    if let Some(m) = &local {
                        m.validate_search(&settings).map_err(bridge)?;
                    }
                    let a = MctsAgent::new(
                        settings,
                        &registry(local.as_ref(), PlayerId::new(0)).map_err(bridge)?,
                    )
                    .map_err(bridge)?;
                    let mut configuration = crate::mcts::configuration_value(a.settings());
                    configuration["seed"] = Value::Null;
                    value["configuration"] = configuration;
                    Some(a.snapshot().map_err(bridge)?)
                }
                SeatSetup::SampledOsla {
                    repetitions_per_action,
                    ..
                } if (1..=128).contains(repetitions_per_action) => None,
                SeatSetup::Random { .. } | SeatSetup::FirstAvailable { .. } => None,
                _ => {
                    return Err(bridge(
                        "stochastic game supports Human, First Available, Random, sampled OSLA (1–128 repetitions), MCTS and Neural MCTS",
                    ));
                }
            };
            Ok(Controller::Ai(StochasticAgent {
                saved: Snapshot {
                    setup: value,
                    rng: rng.snapshot(),
                    simulation,
                    search,
                    local,
                },
                rng,
            }))
        }

        /// Portable metadata keeps all wide integers as strings, including private
        /// simulation progress. Evaluator bytes are checked by the native factories.
        #[derive(Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Policy {
            kind: String,
            version: u32,
            setup: Value,
            simulation: RngData,
            state_evaluator: Option<Vec<u8>>,
            action_evaluators: Option<[Option<Vec<u8>>; 4]>,
            model_id: Option<String>,
        }
        pub(crate) struct Codec<'a>(&'a Registry);
        impl ReplayCodec<StochasticAgent> for Codec<'_> {
            fn validate_fresh(&self, d: &SeatData) -> Result<(), PersistenceError> {
                if let SeatData::Extension { policy, .. } = d {
                    let data: RngData =
                        serde_json::from_value(policy["simulation"].clone()).map_err(invalid)?;
                    if data.snapshot()?.word_position != 0 {
                        return Err(invalid(
                            "replacement simulation RNG must start at its entered seed",
                        ));
                    }
                }
                Ok(())
            }
            fn encode(&self, s: &ControllerSnapshot<Snapshot>) -> SeatData {
                let ControllerSnapshot::Ai(s) = s else {
                    return SeatData::Human;
                };
                SeatData::Extension {
                    rng: RngData::from_snapshot(s.rng),
                    policy: serde_json::to_value(Policy {
                        kind: GAME_ID.into(),
                        version: 1,
                        setup: s.setup.clone(),
                        simulation: RngData::from_snapshot(s.simulation.generator),
                        state_evaluator: s.search.as_ref().map(|a| a.state_evaluator.clone()),
                        action_evaluators: s.search.as_ref().map(|a| a.action_evaluators.clone()),
                        model_id: s.local.as_ref().map(|m| m.model().id().into()),
                    })
                    .unwrap(),
                }
            }
            fn decode(&self, d: &SeatData) -> Result<Controller<StochasticAgent>, PersistenceError> {
                if matches!(d, SeatData::Human) {
                    return Ok(Controller::Human);
                }
                let SeatData::Extension { rng, policy } = d else {
                    return Err(invalid("unsupported stochastic game policy"));
                };
                let p: Policy = serde_json::from_value(policy.clone()).map_err(invalid)?;
                if p.kind != GAME_ID || p.version != 1 {
                    return Err(invalid("unsupported stochastic game policy version"));
                }
                let setup: SeatSetup = serde_json::from_value(p.setup).map_err(invalid)?;
                let Controller::Ai(mut a) = controller(setup, self.0).map_err(|e| invalid(e.message))?
                else {
                    return Err(invalid("AI requires AI setup"));
                };
                let random = rng.snapshot()?;
                let simulation = SimulationRngSnapshot {
                    generator: p.simulation.snapshot()?,
                };
                SimulationRng::from_snapshot(&simulation).map_err(invalid)?;
                if random.initial_seed != a.rng.initial_seed()
                    || random.stream != 0
                    || simulation.generator.initial_seed != random.initial_seed
                {
                    return Err(invalid("controller seed or stream disagrees"));
                }
                a.rng = AgentRng::from_snapshot(&random).map_err(invalid)?;
                a.saved.rng = random;
                a.saved.simulation = simulation;
                if let Some(s) = &mut a.saved.search {
                    s.rng = random;
                    s.simulation_rng = simulation;
                    s.state_evaluator = p
                        .state_evaluator
                        .ok_or_else(|| invalid("missing state evaluator checkpoint"))?;
                    s.action_evaluators = p
                        .action_evaluators
                        .ok_or_else(|| invalid("missing action evaluator checkpoints"))?;
                    a.search(PlayerId::new(0)).map_err(invalid)?;
                } else if p.state_evaluator.is_some() || p.action_evaluators.is_some() {
                    return Err(invalid("unexpected evaluator checkpoint"));
                }
                if self.encode(&ControllerSnapshot::Ai(a.snapshot())) != *d {
                    return Err(invalid("noncanonical controller metadata"));
                }
                Ok(Controller::Ai(a))
            }
            fn recorded_choice(&self, b: &SeatData, a: &SeatData) -> Result<bool, PersistenceError> {
                let (SeatData::Extension { rng: r, policy: p }, SeatData::Extension { rng: s, policy: q }) =
                    (b, a)
                else {
                    return Ok(false);
                };
                if !matches!(p["setup"]["kind"].as_str(), Some("mcts" | "neural_mcts")) {
                    return Ok(false);
                }
                let before = r.snapshot()?;
                let after = s.snapshot()?;
                let ps: RngData = serde_json::from_value(p["simulation"].clone()).map_err(invalid)?;
                let qs: RngData = serde_json::from_value(q["simulation"].clone()).map_err(invalid)?;
                let ps = ps.snapshot()?;
                let qs = qs.snapshot()?;
                let mut previous = p.clone();
                let mut next = q.clone();
                previous["simulation"] = Value::Null;
                next["simulation"] = Value::Null;
                if previous != next
                    || before.initial_seed != after.initial_seed
                    || before.stream != after.stream
                    || after.word_position < before.word_position
                    || ps.initial_seed != qs.initial_seed
                    || ps.stream != qs.stream
                    || qs.word_position < ps.word_position
                {
                    return Err(invalid("MCTS policy changed or RNG progress reversed"));
                }
                self.decode(a)?;
                Ok(true)
            }
            fn extra_work(&self, d: &SeatData) -> usize {
                if let SeatData::Extension { policy, .. } = d {
                    policy["setup"]["repetitions_per_action"]
                        .as_u64()
                        .unwrap_or(0) as usize
                        * 80
                        * 128
                } else {
                    0
                }
            }
        }
    };
}
pub(crate) use stochastic_controllers;
