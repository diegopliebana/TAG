//! Validate a complete experiment, then run fresh independent agents for each game.
use super::agents::{self, MatchGame};
use super::config::{GameConfig, Positions, ResolvedMatch, ResolvedPlayer};
use super::report::{
    ActionRecord, Failure, GameReport, MatchReport, SearchTotals, create, write_new,
};
use super::{Result, derive_seed};
use std::collections::BTreeSet;
use std::path::PathBuf;
use tag_rust::agents::Agent;
use tag_rust::core::{Decision, PlayerId, PlayerResult};
use tag_rust::core::{PublicInformationModel, checkpoint::Checkpoint};
use tag_rust::games::connect4::Connect4ForwardModel;
use tag_rust::games::tictactoe::TicTacToeForwardModel;
use tag_rust::replay::ReplayGame;
use tag_rust::replay::export_replay;
use tag_rust::session::{BranchId, BranchSession, Controller, Session};

/// A finished game's report and optional replay encoding result. Encoding errors
/// are handled after counting the finished game, separately from gameplay errors.
pub(super) type CompletedGame = (GameReport, Option<Result<String>>);

/// Each entry supplies all native match operations for one concrete Rust model.
/// Configuration validation and execution consult this same registration table.
struct GameRegistration {
    name: &'static str,
    run: fn(&ResolvedMatch) -> Result<MatchReport>,
    validate: fn(&GameConfig, usize, &[ResolvedPlayer]) -> Result<ValidatedGame>,
    scheduled: fn(&ResolvedMatch, &[u64]) -> Result<GameReport>,
    #[allow(dead_code)]
    export_actions: fn(&serde_json::Value, &GameReport) -> Result<Option<String>>,
}
impl GameRegistration {
    const fn new<F: MatchGame + Default>(name: &'static str) -> Self {
        Self {
            name,
            run: run_default::<F>,
            validate: validate_default::<F>,
            scheduled: scheduled_default::<F>,
            export_actions: F::action_replay,
        }
    }
}
const GAME_REGISTRY: &[GameRegistration] = &[
    GameRegistration::new::<TicTacToeForwardModel>("tictactoe"),
    GameRegistration::new::<Connect4ForwardModel>("connect4"),
    GameRegistration::new::<tag_rust::games::cantstop::CantStopForwardModel>("cantstop"),
    GameRegistration::new::<tag_rust::games::loveletter::LoveLetterForwardModel>("loveletter"),
];
fn registration(name: &str) -> Result<&'static GameRegistration> {
    GAME_REGISTRY
        .iter()
        .find(|game| game.name == name)
        .ok_or_else(|| {
            let names = GAME_REGISTRY
                .iter()
                .map(|game| game.name)
                .collect::<Vec<_>>()
                .join(", ");
            format!("unknown game '{name}'; registered games: {names}").into()
        })
}
pub fn validate_game_name(name: &str) -> Result<()> {
    registration(name).map(|_| ())
}
fn run_default<F: MatchGame + Default>(settings: &ResolvedMatch) -> Result<MatchReport> {
    run_registered(F::default(), settings)
}

/// Dispatch through the same factories used for configuration validation.
pub fn run(settings: &ResolvedMatch) -> Result<MatchReport> {
    (registration(&settings.config.game.name)?.run)(settings)
}

/// Effective game parameters and rule version for a validated tournament entry.
#[allow(dead_code)] // Read by tournaments; shared support also builds in the standalone match CLI.
pub struct ValidatedGame {
    pub parameters: serde_json::Value,
    pub rules_version: u32,
    pub replay_dimensions: Option<(usize, usize)>,
}
/// Validate every agent in a tournament pool against one game's configuration.
/// The pool can be larger than the player count. No actions or output writes occur.
#[allow(dead_code)] // Used by tournaments; this module also builds in the standalone match CLI.
pub fn validate_pool(
    game: &GameConfig,
    player_count: usize,
    pool: &[ResolvedPlayer],
) -> Result<ValidatedGame> {
    (registration(&game.name)?.validate)(game, player_count, pool)
}

fn validate_default<F: MatchGame + Default>(
    game: &GameConfig,
    count: usize,
    pool: &[ResolvedPlayer],
) -> Result<ValidatedGame> {
    let model = F::default();
    F::validate_ranking(game.ranking.as_deref())?;
    let parameters = parameters::<F>(game)?;
    model.init_state(&parameters, count)?;
    let registry = F::registry();
    for player in pool {
        #[cfg(feature = "inference")]
        if let Some(network) = &player.selected_model {
            model.validate_neural_model(&parameters, network)?;
        }
        agents::build::<F>(player, player.base_seed, &registry)?;
    }
    Ok(ValidatedGame {
        replay_dimensions: F::replay_dimensions(&parameters),
        parameters: serde_json::to_value(parameters)?,
        rules_version: <F as MatchGame>::RULES_VERSION,
    })
}
/// Execute one scheduled lineup with explicit per-seat seeds. Tournament scheduling
/// owns the ordering and seed derivation; this reuses ordinary native move execution.
#[allow(dead_code)] // Used by tournaments; this module also builds in the standalone match CLI.
pub fn run_scheduled(settings: &ResolvedMatch, agent_seeds: &[u64]) -> Result<GameReport> {
    if agent_seeds.len() != settings.players.len()
        || settings.config.positions != Positions::Fixed
        || settings.config.reporting.recordings.is_some()
    {
        return Err(
            "scheduled matches require fixed seats, one seed per seat and no replay export".into(),
        );
    }
    (registration(&settings.config.game.name)?.scheduled)(settings, agent_seeds)
}

fn scheduled_default<F: MatchGame + Default>(
    settings: &ResolvedMatch,
    seeds: &[u64],
) -> Result<GameReport> {
    let model = F::default();
    let mut parameters = parameters::<F>(&settings.config.game)?;
    if F::STOCHASTIC
        && settings
            .config
            .game
            .parameters
            .get("seed")
            .is_none_or(serde_json::Value::is_null)
    {
        parameters = F::with_game_seed(&parameters, settings.master_seed);
    }
    let (report, _) = play_one(
        &model,
        &parameters,
        &F::registry(),
        settings,
        0,
        seeds.to_vec(),
    )?;
    Ok(report)
}
fn parameters<F: MatchGame>(game: &GameConfig) -> Result<F::Parameters> {
    // Fill omitted rule parameters from the game's actual defaults. Deserialising
    // the merged object uses the game's existing unknown-field and type checks.
    let mut values = serde_json::to_value(F::Parameters::default())?;
    for (name, value) in game
        .parameters
        .as_object()
        .ok_or("parameters must be an object")?
    {
        if F::STOCHASTIC && name == "seed" && value.is_null() {
            continue;
        }
        values
            .as_object_mut()
            .ok_or("registered game parameters must be an object")?
            .insert(name.clone(), value.clone());
    }
    Ok(serde_json::from_value(values)?)
}

pub fn seats(settings: &ResolvedMatch, game: usize) -> Vec<usize> {
    let count = settings.players.len();
    let rotation = match settings.config.positions {
        Positions::Rotate => game % count,
        Positions::Fixed => 0,
    };
    (0..count).map(|agent| (agent + rotation) % count).collect()
}
fn seeds(settings: &ResolvedMatch, game: usize) -> Vec<u64> {
    settings
        .players
        .iter()
        .enumerate()
        .map(|(agent, player)| derive_seed(player.base_seed, game as u64, agent as u64 + 1))
        .collect()
}
fn recording_path(settings: &ResolvedMatch, game: usize) -> Option<PathBuf> {
    settings
        .config
        .reporting
        .recordings
        .as_ref()
        .map(|directory| directory.join(format!("game-{game:06}.json")))
}

/// Reject output conflicts before playing. Files are still opened with create_new
/// when written, so another process cannot make us overwrite an existing file.
fn check_outputs(settings: &ResolvedMatch) -> Result<()> {
    let mut paths = Vec::new();
    if let Some(path) = &settings.config.reporting.path {
        paths.push(path.clone());
    }
    for game in 0..settings.config.games {
        if let Some(path) = recording_path(settings, game) {
            paths.push(path);
        }
    }
    let mut destinations = BTreeSet::new();
    for path in &paths {
        if path.exists() {
            return Err(format!("output already exists: {}", path.display()).into());
        }
        // Normalise dot components for conflict checks without changing the path
        // used for writing. This catches aliases such as replays/../replays/file.
        let mut normal = PathBuf::new();
        for component in std::path::absolute(path)?.components() {
            match component {
                std::path::Component::ParentDir => {
                    normal.pop();
                }
                std::path::Component::CurDir => {}
                _ => normal.push(component),
            }
        }
        if !destinations.insert(normal) {
            return Err("report and recording destinations conflict".into());
        }
        for parent in path
            .ancestors()
            .skip(1)
            .filter(|p| !p.as_os_str().is_empty())
        {
            if parent.exists() && !parent.is_dir() {
                return Err(
                    format!("output parent is not a directory: {}", parent.display()).into(),
                );
            }
        }
    }
    for destination in &destinations {
        if destination
            .ancestors()
            .skip(1)
            .any(|parent| destinations.contains(parent))
        {
            return Err("an output file would also be used as an output directory".into());
        }
    }
    Ok(())
}

fn run_registered<F: MatchGame>(model: F, settings: &ResolvedMatch) -> Result<MatchReport> {
    F::validate_ranking(settings.config.game.ranking.as_deref())?;
    let mut parameters = parameters::<F>(&settings.config.game)?;
    if F::STOCHASTIC
        && settings
            .config
            .game
            .parameters
            .get("seed")
            .is_none_or(serde_json::Value::is_null)
    {
        parameters = F::with_game_seed(&parameters, settings.master_seed);
    }
    if settings.config.reporting.recordings.is_some() && !F::BROWSER_REPLAY {
        return Err(
            "this game has no browser replay capability; use action/chance research logs".into(),
        );
    }
    // Construct all evaluator roles before the first game, so unknown factories
    // or incompatible native settings cannot cause a partially started experiment.
    model.init_state(&parameters, settings.players.len())?;
    let registry = F::registry();
    let initial_seeds = seeds(settings, 0);
    for (index, player) in settings.players.iter().enumerate() {
        #[cfg(feature = "inference")]
        if let Some(network) = &player.selected_model {
            model.validate_neural_model(&parameters, network)?;
        }
        agents::build::<F>(player, initial_seeds[index], &registry)?;
    }
    check_outputs(settings)?;
    let report = create(settings, serde_json::to_value(&parameters)?)?;
    Ok(run_games(settings, report, |game| {
        play_one(
            &model,
            &parameters,
            &registry,
            settings,
            game,
            seeds(settings, game),
        )
    }))
}

/// Stop at the first runtime failure while retaining all completed games. Keeping
/// aggregation separate from game execution also allows controlled failure tests.
pub(super) fn run_games(
    settings: &ResolvedMatch,
    mut report: MatchReport,
    mut play: impl FnMut(usize) -> Result<CompletedGame>,
) -> MatchReport {
    for game in 0..settings.config.games {
        let outcome = play(game);
        let (completed, recording) = match outcome {
            Ok(value) => value,
            Err(error) => {
                report.failure = Some(Failure {
                    game_index: game,
                    message: error.to_string(),
                });
                break;
            }
        };
        for (agent, result) in completed.results.iter().enumerate() {
            report.agents[agent].combined.add(result);
            report.agents[agent].by_seat[completed.seats[agent]].add(result);
        }
        if completed.truncated {
            report.truncated_games += 1;
        } else {
            report.completed_games += 1;
        }
        report.games.push(completed);
        if let Some(text) = recording {
            let path = recording_path(settings, game).expect("recording was requested");
            if let Err(error) = text.and_then(|text| write_new(&path, &text)) {
                report.failure = Some(Failure {
                    game_index: game,
                    message: format!("game completed but recording failed: {error}"),
                });
                break;
            }
            report
                .games
                .last_mut()
                .expect("completed game was inserted")
                .recording = Some(path.to_string_lossy().into_owned());
        }
    }
    report
}

fn result_labels<F: MatchGame>(
    model: &F,
    state: &F::State,
    order: &[usize],
) -> Result<Vec<String>> {
    order
        .iter()
        .map(|seat| {
            Ok(match model.player_result(state, PlayerId::new(*seat))? {
                PlayerResult::Win => "win",
                PlayerResult::Draw => "draw",
                PlayerResult::Loss => "loss",
                PlayerResult::Ongoing => {
                    return Err("finished game has an ongoing player result".into());
                }
            }
            .into())
        })
        .collect()
}

fn final_ranks<F: MatchGame>(
    model: &F,
    state: &F::State,
    order: &[usize],
    rule: Option<&str>,
) -> Result<Option<Vec<usize>>> {
    model
        .final_ranks(state, rule)?
        .map(|ranks| {
            if ranks.len() != order.len() {
                return Err("game final-rank count differs from player count".into());
            }
            Ok(order.iter().map(|&seat| ranks[seat]).collect())
        })
        .transpose()
}

fn play_one<F: MatchGame>(
    model: &F,
    parameters: &F::Parameters,
    registry: &tag_rust::search::mcts::EvaluatorRegistry<F>,
    settings: &ResolvedMatch,
    game: usize,
    agent_seeds: Vec<u64>,
) -> Result<CompletedGame> {
    let order = seats(settings, game);
    let mut by_seat = vec![0; order.len()];
    for (agent, seat) in order.iter().enumerate() {
        by_seat[*seat] = agent;
    }
    let mut actions = Vec::new();
    let mut accepted_moves = 0;
    let mut totals: Vec<_> = (0..order.len()).map(|_| SearchTotals::default()).collect();
    let recording = None;
    if settings.config.reporting.recordings.is_some() {
        return F::recorded(settings, game, agent_seeds);
    }
    let game_seed = F::game_seed(parameters).map(|base| {
        derive_seed(
            base,
            settings
                .game_index_offset
                .checked_add(game)
                .expect("bounded schedule index") as u64,
            0x47414d45,
        )
    });
    let parameters = game_seed.map_or_else(
        || parameters.clone(),
        |seed| F::with_game_seed(parameters, seed),
    );
    let mut state = model.init_state(&parameters, order.len())?;
    let mut agents = by_seat
        .iter()
        .map(|agent| agents::build::<F>(&settings.players[*agent], agent_seeds[*agent], registry))
        .collect::<Result<Vec<_>>>()?;
    while let Decision::Player(actor) = model.next_player(&state) {
        if settings
            .config
            .max_actions
            .is_some_and(|limit| accepted_moves >= limit)
        {
            break;
        }
        let observation = model.observe(&state, actor)?;
        let offered = model.available_actions(&state, actor)?;
        if offered.is_empty() {
            return Err("nonterminal game offered no actions".into());
        }
        let agent = &mut agents[actor.index()];
        let selected = agent
            .choose(model, actor, &observation, &offered)
            .map_err(|e| e.to_string())?;
        let action = offered
            .get(selected)
            .ok_or("agent selected an invalid action index")?;
        model.next(&mut state, actor, action)?;
        accepted_moves += 1;
        if settings.config.reporting.include_actions {
            actions.push(ActionRecord {
                seat: actor.index(),
                action: serde_json::to_value(action)?,
                chance: F::chance_event(&state, action),
            });
        }
        if let Some(counters) = agent.last_counters() {
            let counts = &mut totals[by_seat[actor.index()]];
            counts.completed_iterations = counts
                .completed_iterations
                .checked_add(counters.completed_iterations)
                .ok_or("search total overflow")?;
            counts.forward_model_calls = counts
                .forward_model_calls
                .checked_add(counters.forward_model_calls)
                .ok_or("search total overflow")?;
        }
    }
    let truncated = model.next_player(&state) != Decision::Finished;
    let results = if truncated {
        vec!["truncated".into(); order.len()]
    } else {
        result_labels(model, &state, &order)?
    };
    let ranks = if truncated {
        None
    } else {
        final_ranks(
            model,
            &state,
            &order,
            settings.config.game.ranking.as_deref(),
        )?
    };
    Ok((
        GameReport {
            index: game,
            game_seed,
            ranks,
            truncated,
            max_actions: settings.config.max_actions,
            agent_seeds,
            seats: order,
            results,
            accepted_moves,
            mcts: totals,
            actions: settings.config.reporting.include_actions.then_some(actions),
            recording: None,
        },
        recording,
    ))
}

/// Optional replay capability, used only by registered deterministic browser games.
pub(super) fn play_recorded<F>(
    model: F,
    settings: &ResolvedMatch,
    game: usize,
    agent_seeds: Vec<u64>,
) -> Result<CompletedGame>
where
    F: MatchGame + PublicInformationModel + ReplayGame,
    F::SavedEvaluator:
        Default + Checkpoint<Snapshot = (), RestoreError: std::error::Error + 'static>,
    <F::SavedEvaluator as tag_rust::core::evaluation::Evaluator<F::State>>::Error:
        std::error::Error + 'static,
{
    let mut parameters = parameters::<F>(&settings.config.game)?;
    if F::STOCHASTIC
        && settings
            .config
            .game
            .parameters
            .get("seed")
            .is_none_or(serde_json::Value::is_null)
    {
        parameters = F::with_game_seed(&parameters, settings.master_seed);
    }
    let order = seats(settings, game);
    let mut by_seat = vec![0; order.len()];
    for (agent, seat) in order.iter().enumerate() {
        by_seat[*seat] = agent;
    }
    let mut actions = Vec::new();
    let mut accepted_moves = 0;
    let controllers = by_seat
        .iter()
        .map(|agent| {
            Ok(Controller::Ai(agents::basic::<F>(
                &settings.players[*agent],
                agent_seeds[*agent],
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut session = BranchSession::new(Session::create(
        model.clone(),
        &parameters,
        order.len(),
        controllers,
    )?);
    while let Decision::Player(_) = session.active().session().next_player() {
        if settings
            .config
            .max_actions
            .is_some_and(|limit| accepted_moves >= limit)
        {
            break;
        }
        let version = session.active().session().position_version();
        let accepted = session.advance_ai(version)?;
        accepted_moves += 1;
        if settings.config.reporting.include_actions {
            actions.push(ActionRecord {
                chance: None,
                seat: accepted.actor.index(),
                action: serde_json::to_value(accepted.action)?,
            });
        }
    }
    let truncated = session.active().session().next_player() != Decision::Finished;
    let results = if truncated {
        vec!["truncated".into(); order.len()]
    } else {
        result_labels(&model, session.active().session().state(), &order)?
    };
    let cursor = session.info(BranchId(0))?.move_count;
    let recording = Some(export_replay(&session, &parameters, cursor, 0).map_err(Into::into));

    let ranks = if truncated {
        None
    } else {
        final_ranks(
            &model,
            session.active().session().state(),
            &order,
            settings.config.game.ranking.as_deref(),
        )?
    };
    Ok((
        GameReport {
            index: game,
            game_seed: None,
            agent_seeds,
            seats: order,
            results,
            accepted_moves,
            ranks,
            truncated,
            max_actions: settings.config.max_actions,
            mcts: (0..settings.players.len())
                .map(|_| SearchTotals::default())
                .collect(),
            actions: settings.config.reporting.include_actions.then_some(actions),
            recording: None,
        },
        recording,
    ))
}

#[allow(dead_code)] // Tournament logging also uses this shared match registration.
pub fn export_action_replay(
    name: &str,
    parameters: &serde_json::Value,
    outcome: &GameReport,
) -> Result<Option<String>> {
    (registration(name)?.export_actions)(parameters, outcome)
}
pub(super) fn action_replay<
    F: MatchGame + tag_rust::core::PublicInformationModel + tag_rust::replay::ReplayGame,
>(
    model: F,
    parameters: &serde_json::Value,
    outcome: &crate::matches::report::GameReport,
) -> Result<String>
where
    F::Action: serde::de::DeserializeOwned,
    F::SavedEvaluator: Default
        + tag_rust::core::checkpoint::Checkpoint<
            Snapshot = (),
            RestoreError: std::error::Error + 'static,
        >,
    <F::SavedEvaluator as tag_rust::core::evaluation::Evaluator<F::State>>::Error:
        std::error::Error + 'static,
{
    let mut effective = serde_json::to_value(F::Parameters::default())?;
    for (key, value) in parameters
        .as_object()
        .ok_or("game parameters must be an object")?
    {
        effective
            .as_object_mut()
            .expect("registered parameters are objects")
            .insert(key.clone(), value.clone());
    }
    let parameters: F::Parameters = serde_json::from_value(effective)?;
    // Human controllers store the actual moves, independently of the algorithm
    // that selected them. Loading never re-runs a time-limited MCTS decision.
    let controllers: Vec<Controller<tag_rust::session::BuiltinAgent<F::SavedEvaluator>>> = (0
        ..outcome.seats.len())
        .map(|_| Controller::Human)
        .collect();
    let mut session = BranchSession::new(Session::create(
        model,
        &parameters,
        controllers.len(),
        controllers,
    )?);
    for action in outcome
        .actions
        .as_ref()
        .ok_or("missing actions for game log")?
    {
        let version = session.active().session().position_version();
        session.submit_human(
            version,
            PlayerId::new(action.seat),
            serde_json::from_value(action.action.clone())?,
        )?;
    }
    Ok(export_replay(
        &session,
        &parameters,
        outcome.accepted_moves,
        0,
    )?)
}

#[cfg(test)]
mod capabilities {
    use super::*;
    use tag_rust::core::ForwardModel;
    use tag_rust::games::tictactoe::*;
    #[derive(Clone)]
    struct Minimal;
    impl ForwardModel for Minimal {
        type State = TicTacToeState;
        type Observation = TicTacToeObservation;
        type Action = PlaceMark;
        type Parameters = TicTacToeParameters;
        type Error = TicTacToeError;
        fn init_state(
            &self,
            p: &Self::Parameters,
            n: usize,
        ) -> std::result::Result<Self::State, Self::Error> {
            TicTacToeForwardModel.init_state(p, n)
        }
        fn player_count(&self, s: &Self::State) -> usize {
            TicTacToeForwardModel.player_count(s)
        }
        fn next_player(&self, s: &Self::State) -> Decision {
            TicTacToeForwardModel.next_player(s)
        }
        fn observe(
            &self,
            s: &Self::State,
            p: PlayerId,
        ) -> std::result::Result<Self::Observation, Self::Error> {
            TicTacToeForwardModel.observe(s, p)
        }
        fn offered_actions(&self, o: &Self::Observation) -> Vec<Self::Action> {
            TicTacToeForwardModel.offered_actions(o)
        }
        fn available_actions(
            &self,
            s: &Self::State,
            p: PlayerId,
        ) -> std::result::Result<Vec<Self::Action>, Self::Error> {
            TicTacToeForwardModel.available_actions(s, p)
        }
        fn next(
            &self,
            s: &mut Self::State,
            p: PlayerId,
            a: &Self::Action,
        ) -> std::result::Result<(), Self::Error> {
            TicTacToeForwardModel.next(s, p, a)
        }
        fn player_result(
            &self,
            s: &Self::State,
            p: PlayerId,
        ) -> std::result::Result<PlayerResult, Self::Error> {
            TicTacToeForwardModel.player_result(s, p)
        }
        fn player_score(
            &self,
            s: &Self::State,
            p: PlayerId,
        ) -> std::result::Result<f64, Self::Error> {
            TicTacToeForwardModel.player_score(s, p)
        }
    }
    // Deliberately no simulation, heuristic, replay or learning implementations.
    impl MatchGame for Minimal {}
    #[test]
    fn random_matches_do_not_require_optional_game_capabilities() {
        let value = serde_json::json!({"schema_version":1,"game":{"name":"tictactoe"},"games":2,"seed":42,
            "players":[{"name":"a","agent":{"type":"random"}},{"name":"b","agent":{"type":"random"}}]});
        let resolve = |v| {
            serde_json::from_value::<super::super::config::MatchConfig>(v)
                .unwrap()
                .resolve(std::path::Path::new("."))
                .unwrap()
        };
        let report = run_registered(Minimal, &resolve(value.clone())).unwrap();
        assert_eq!(report.completed_games, 2);
        assert!(report.failure.is_none());
        for agent in [
            serde_json::json!({"type":"osla"}),
            serde_json::json!({"type":"mcts","config":{"schema_version":1,"budgets":{"iterations":1}}}),
        ] {
            let mut invalid = value.clone();
            invalid["players"][0]["agent"] = agent;
            assert!(run_registered(Minimal, &resolve(invalid)).is_err());
        }
    }
}
